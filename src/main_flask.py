"""
Flask Web Application for Interview Session
Supports both text and voice input/output with authentication
"""

from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, flash, stream_with_context
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from functools import wraps
import traceback
import asyncio
import threading
import os
import sys
import uuid
import argparse
import time
import logging
import secrets
# import hashlib  # unused when name-only login is active
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv(override=True)

# Your backend imports
from src.utils.speech.text_to_speech import TextToSpeechBase, create_tts_engine
from src.utils.speech.audio_player import AudioPlayerBase, create_audio_player
from src.utils.speech.speech_to_text import create_stt_engine
from src.interview_session.interview_session import InterviewSession

# =============================================================================
# CONFIGURATION
# =============================================================================

SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
START_TIME = time.time()

class AppConfig:
    """Application configuration"""
    def __init__(self):
        self.default_user_id = "web_user"
        self.host = "0.0.0.0"
        self.port = 5000
        self.debug = False
        self.restart = False
        self.max_turns = None
        self.additional_context_path = None

config = AppConfig()

# TTS/STT engines
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
TTS_VOICE = os.getenv('TTS_VOICE', 'alloy')
if TTS_PROVIDER and TTS_PROVIDER.lower() != 'none':
    tts_engine: TextToSpeechBase = create_tts_engine(provider=TTS_PROVIDER, voice=TTS_VOICE)
else:
    tts_engine = None
stt_engine = create_stt_engine()

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__,
            static_folder='web/static',
            template_folder='web/templates')

app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
CORS(app)

# =============================================================================
# AUTHENTICATION SETUP
# =============================================================================

REQUIRE_LOGIN = os.getenv('REQUIRE_LOGIN', 'true').lower() == 'true'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = None

USERS_FILE = os.path.join(os.getenv('DATA_DIR', 'data'), 'users.json')

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# def hash_password(password):
#     """Hash password using SHA-256"""
#     return hashlib.sha256(password.encode()).hexdigest()

_ANON_USER_ID = os.getenv('ANON_USER_ID', 'anon')
_ANON_USERNAME = os.getenv('ANON_USERNAME', 'anon')

if not REQUIRE_LOGIN:
    # Override Flask-Login's login_required to be a no-op
    def login_required(f):
        return f

def get_current_user():
    """Return current_user if login is required, otherwise return a fixed anon user."""
    if REQUIRE_LOGIN:
        return current_user
    if current_user.is_authenticated:
        return current_user
    return User(_ANON_USER_ID, _ANON_USERNAME)

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        return User(user_id, users[user_id]['username'])
    return None

# =============================================================================
# LOGGING SETUP
# =============================================================================

if not app.debug:
    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler(
        'logs/flask_app.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Interview application startup')

# =============================================================================
# ASYNC EVENT LOOP MANAGEMENT
# =============================================================================

loop = asyncio.new_event_loop()

def start_background_loop(loop):
    """Run async event loop in background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_background_loop, args=(loop,), daemon=True).start()

def run_async_task(coro):
    """Submit coroutine to background loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionWrapper:
    def __init__(self, session_token: str, interview_session: InterviewSession,
                 user_id: str):
        self.session_token = session_token
        self.interview_session = interview_session
        self.user_id = user_id
        self.created_at = time.time()

active_sessions: Dict[str, SessionWrapper] = {}
last_messages_by_session: Dict[str, Dict[str, str]] = {}
session_audio_cache: Dict[str, Dict[str, object]] = {}
# Tracks how many chat_history entries have been delivered for agent-mode sessions
chat_history_offsets: Dict[str, int] = {}

def create_interview_session(user_id: str, session_type: str = "intake",
                             interaction_mode: str = 'api',
                             available_time: int = None) -> tuple[InterviewSession, str]:
    """Create interview session with authenticated user_id.

    Args:
        user_id: User identifier (for agent mode, use the sample profile user_id).
        session_type: "intake" for initial profiling, "weekly" for recurring check-ins.
        interaction_mode: 'api' for human-via-web, 'agent' for simulated user.
        available_time: Participant's available time in minutes (from pre-interview question).
    """
    session_token = str(uuid.uuid4())

    if session_type == "weekly":
        interview_plan_path = os.getenv('INTERVIEW_PLAN_PATH_WEEKLY',
                                        'configs/topics_weekly.json')
        interview_description = "Weekly work check-in: tracking how your tasks and work are evolving"
    else:
        interview_plan_path = os.getenv('INTERVIEW_PLAN_PATH_INTAKE',
                                        'configs/topics_intake.json')
        interview_description = os.getenv(
            'INTERVIEW_DESCRIPTION',
            "Initial intake interview: understanding your role, tasks, and work patterns"
        )

    interview_session = InterviewSession(
        interaction_mode=interaction_mode,
        user_config={
            "user_id": user_id,
            "enable_voice": False,
            "restart": config.restart
        },
        interview_config={
            "enable_voice": False,
            "interview_description": interview_description,
            "interview_plan_path": interview_plan_path,
            "interview_evaluation": os.getenv('COMPLETION_METRIC'),
            "additional_context_path": config.additional_context_path,
            "initial_user_portrait_path": os.getenv('USER_PORTRAIT_PATH'),
            "session_type": session_type,
            "available_time": available_time,
        },
        max_turns=config.max_turns
    )
    
    wrapper = SessionWrapper(
        session_token=session_token,
        interview_session=interview_session,
        user_id=user_id,
    )
    active_sessions[session_token] = wrapper
    
    session_loop = asyncio.new_event_loop()
    def _start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    t = threading.Thread(target=_start_loop, args=(session_loop,), daemon=True)
    t.start()

    wrapper.loop = session_loop
    wrapper.loop_thread = t
    asyncio.run_coroutine_threadsafe(interview_session.run(), session_loop)
    
    return interview_session, session_token

def get_session(session_token: str) -> Optional[InterviewSession]:
    wrapper = active_sessions.get(session_token)
    return wrapper.interview_session if wrapper is not None else None

def get_session_wrapper(session_token: str) -> Optional[SessionWrapper]:
    return active_sessions.get(session_token)

def agent_or_login_required(f):
    """Allow access if login is disabled, logged in, or a valid session_token is present (agent mode)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not REQUIRE_LOGIN:
            return f(*args, **kwargs)
        if current_user.is_authenticated:
            return f(*args, **kwargs)
        token = (request.args.get('session_token') or
                 (request.get_json(silent=True) or {}).get('session_token'))
        if token and token in active_sessions:
            return f(*args, **kwargs)
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    return decorated

# =============================================================================
# AUTHENTICATION ROUTES (NO @login_required)
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if not REQUIRE_LOGIN:
        return redirect(url_for('index'))
    if current_user.is_authenticated:
        return redirect(url_for('index'))  # Changed: redirect to index with instructions
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()

        if not username:
            flash('Please enter your Stanford ID', 'error')
            return render_template('login.html')

        users = load_users()

        # Find existing user by Stanford ID, or create a new one
        user_id = None
        for uid, user_data in users.items():
            if user_data['username'] == username:
                user_id = uid
                break

        if not user_id:
            user_id = secrets.token_urlsafe(16)
            users[user_id] = {
                'username': username,
                'created_at': time.time()
            }
            save_users(users)
            os.makedirs(os.path.join(os.getenv('LOGS_DIR', 'logs'), user_id), exist_ok=True)
            os.makedirs(os.path.join(os.getenv('DATA_DIR', 'data'), user_id), exist_ok=True)
            app.logger.info(f"New user created: {username} ({user_id})")

        user = User(user_id, username)
        login_user(user)
        app.logger.info(f"User logged in: {username} ({user_id})")

        next_page = request.args.get('next')
        return redirect(next_page if next_page else url_for('index'))

        # --- password-based login (commented out) ---
        # password = request.form.get('password', '')
        # if user_id and users[user_id]['password'] == hash_password(password):
        #     user = User(user_id, username)
        #     login_user(user)
        #     app.logger.info(f"User logged in: {username} ({user_id})")
        #     next_page = request.args.get('next')
        #     return redirect(next_page if next_page else url_for('index'))
        # else:
        #     flash('Invalid username or password', 'error')

    return render_template('login.html')

# --- /register route (commented out — superseded by name-only login) ---
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     """Registration page"""
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))
#
#     if request.method == 'POST':
#         username = request.form.get('username', '').strip()
#         password = request.form.get('password', '')
#
#         if not username or not password:
#             flash('Username and password are required', 'error')
#             return render_template('register.html')
#
#         if len(password) < 6:
#             flash('Password must be at least 6 characters', 'error')
#             return render_template('register.html')
#
#         users = load_users()
#
#         for user_data in users.values():
#             if user_data['username'] == username:
#                 flash('Username already exists', 'error')
#                 return render_template('register.html')
#
#         user_id = secrets.token_urlsafe(16)
#         users[user_id] = {
#             'username': username,
#             'password': hash_password(password),
#             'created_at': time.time()
#         }
#         save_users(users)
#         os.makedirs(os.path.join(os.getenv('LOGS_DIR', 'logs'), user_id), exist_ok=True)
#         os.makedirs(os.path.join(os.getenv('DATA_DIR', 'data'), user_id), exist_ok=True)
#         app.logger.info(f"New user registered: {username} ({user_id})")
#         flash('Registration successful! Please login.', 'success')
#         return redirect(url_for('login'))
#
#     return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    username = get_current_user().username
    logout_user()
    app.logger.info(f"User logged out: {username}")
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# =============================================================================
# PAGE ROUTES - PROTECTED (REQUIRE LOGIN)
# =============================================================================

@app.route('/')
@login_required  # MUST BE LOGGED IN TO SEE INSTRUCTIONS
def index():
    """Landing page with instructions - shown after login.
    If the user already has a prior session (intake completed), skip to chat."""
    logs_dir = os.getenv("LOGS_DIR", "logs")
    user_logs = os.path.join(logs_dir, get_current_user().id, "execution_logs")
    if os.path.isdir(user_logs):
        session_dirs = [d for d in os.listdir(user_logs)
                        if d.startswith('session_') and
                        os.path.isdir(os.path.join(user_logs, d))]
        if session_dirs:
            return redirect(url_for('unified_chat'))
    return render_template('index.html', username=get_current_user().username)

@app.route('/chat')
@login_required  # MUST BE LOGGED IN
def unified_chat():
    """Unified chat interface"""
    return render_template('chat.html', username=get_current_user().username)

@app.route('/visualizer')
def visualizer():
    """Live interview + session state visualizer"""
    username = get_current_user().username if current_user.is_authenticated else 'agent'
    return render_template('visualizer.html', username=username)

# =============================================================================
# API ENDPOINTS - PROTECTED (REQUIRE LOGIN)
# All endpoints that handle interview data need @login_required
# =============================================================================

@app.route('/api/start-session', methods=['POST'])
@login_required  # PROTECTED
def start_session():
    """Initialize a new interview session using authenticated user's ID"""
    user_id = get_current_user().id
    
    # DEBUG: Log every request
    call_stack = ''.join(traceback.format_stack()[-3:-1])
    app.logger.info(f"[DEBUG] start-session called by user {get_current_user().username}")
    print(f"[DEBUG] start-session request from {get_current_user().username} (user_id: {user_id})")
    
    # Check if user already has an active in-progress session
    for token, wrapper in active_sessions.items():
        if wrapper.user_id == user_id and wrapper.interview_session.session_in_progress:
            # Return existing session instead of creating duplicate
            app.logger.info(f"Returning existing session {token} for user {get_current_user().username}")
            print(f"[Session] Reusing existing session {token} for {get_current_user().username}")
            return jsonify({
                'success': True,
                'session_token': token,
                'session_id': wrapper.interview_session.session_id,
                'user_id': user_id,
                'username': get_current_user().username,
                'message': 'Using existing session',
                'was_existing': True
            })
    
    # Create new session only if none exists
    data = request.get_json(silent=True) or {}
    session_type = data.get("session_type", os.getenv("SESSION_TYPE", "intake"))
    available_time = data.get("available_time")  # minutes, from pre-interview question
    interview_session, session_token = create_interview_session(user_id=user_id, session_type=session_type,
                                                                 available_time=available_time)

    app.logger.info(f"Session created: {session_token} | User: {get_current_user().username} ({user_id})")
    print(f"[Session] Created NEW session {session_token} for user {get_current_user().username}")

    return jsonify({
        'success': True,
        'session_token': session_token,
        'session_id': interview_session.session_id,
        'user_id': user_id,
        'username': get_current_user().username,
        'message': 'Session started successfully',
        'was_existing': False
    })

@app.route('/api/start-agent-session', methods=['POST'])
def start_agent_session():
    """Start an autonomous session with a simulated user (UserAgent) for testing/visualization."""
    data = request.get_json(silent=True) or {}
    session_type = data.get("session_type", os.getenv("SESSION_TYPE", "intake"))

    viewer = get_current_user().username if current_user.is_authenticated else 'agent'

    # sim_user_id must come from the request body; fall back to logged-in user id only if authenticated
    sim_user_id = data.get("sim_user_id") or (get_current_user().id if current_user.is_authenticated else None)
    if not sim_user_id:
        return jsonify({'success': False, 'error': 'sim_user_id is required'}), 400

    # Check if there's already an active agent session for this sim_user_id
    for token, wrapper in active_sessions.items():
        if wrapper.user_id == sim_user_id:
            app.logger.info(f"Returning existing agent session {token} for sim_user {sim_user_id}")
            return jsonify({
                'success': True,
                'session_token': token,
                'session_id': wrapper.interview_session.session_id,
                'user_id': sim_user_id,
                'username': viewer,
                'message': 'Using existing agent session',
                'was_existing': True,
                'agent_mode': True,
            })

    interview_session, session_token = create_interview_session(
        user_id=sim_user_id,
        session_type=session_type,
        interaction_mode='agent',
    )

    app.logger.info(f"Agent session created: {session_token} | sim_user: {sim_user_id} | watcher: {viewer}")
    print(f"[AgentSession] Created agent session {session_token} simulating user {sim_user_id}")

    return jsonify({
        'success': True,
        'session_token': session_token,
        'session_id': interview_session.session_id,
        'user_id': sim_user_id,
        'username': viewer,
        'message': 'Agent session started successfully',
        'was_existing': False,
        'agent_mode': True,
    })

@app.route('/api/agent-control', methods=['POST'])
@agent_or_login_required
def agent_control():
    """Pause, resume, or step the simulated user agent."""
    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    action = data.get('action')  # 'pause' | 'resume' | 'step'

    session = get_session(session_token)
    if not session:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    if action == 'pause':
        session.paused = True
    elif action == 'resume':
        session.paused = False
        session.step_requested = False
    elif action == 'step':
        session.paused = True   # stay paused after this step
        session.step_requested = True
    else:
        return jsonify({'success': False, 'error': f'Unknown action: {action}'}), 400

    return jsonify({'success': True, 'paused': session.paused})

@app.route('/api/send-message', methods=['POST'])
@login_required  # PROTECTED
def send_message():
    """Send a text message to the interview session"""
    data = request.json
    session_token = data.get('session_token')
    user_message = data.get('message')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not session.session_in_progress:
        return jsonify({
            'success': False,
            'error': 'Session has ended',
            'session_completed': True
        }), 400

    wrapper = get_session_wrapper(session_token)
    if wrapper and hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(wrapper.interview_session.user.add_user_message, user_message)
    else:
        session.user.add_user_message(user_message)

    bot_reply = wait_for_agent_response(session)

    return jsonify({
        'success': True,
        'message': 'Message sent successfully'
    })

@app.route('/api/send-voice', methods=['POST'])
@login_required  # PROTECTED
def send_voice():
    """Send a voice message to the interview session"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    orig_filename = audio_file.filename or 'recording.webm'
    ext = Path(orig_filename).suffix or '.webm'
    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}{ext}")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)

        transcribe_only = request.form.get('transcribe_only', 'false').lower() == 'true'
        if not transcribe_only:
            wrapper = get_session_wrapper(session_token)
            if wrapper and hasattr(wrapper, 'loop'):
                wrapper.loop.call_soon_threadsafe(wrapper.interview_session.user.add_user_message, transcribed_text)
            else:
                session.user.add_user_message(transcribed_text)

        return jsonify({
            'success': True,
            'transcribed_text': transcribed_text,
            'message': 'Transcription complete' if transcribe_only else 'Voice message processed successfully'
        })
    except Exception as e:
        app.logger.error(f"[send_voice] Transcription error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

def _prestart_tts(session_token: str, message_id: str, text: str, wrapper) -> None:
    """Kick off TTS generation in background immediately when a message is delivered."""
    cache = session_audio_cache.setdefault(session_token, {})
    if message_id in cache:
        return  # Already started or done
    cache[message_id] = {'status': 'pending', 'data': None, 'error': None, 'timestamp': time.time()}

    async def _synth():
        try:
            def _blocking():
                out = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
                try:
                    tts_engine.text_to_speech(text=text, output_path=str(out))
                    return out.read_bytes()
                finally:
                    if out.exists():
                        out.unlink(missing_ok=True)
            data = await wrapper.loop.run_in_executor(None, _blocking)
            cache[message_id] = {'status': 'ready', 'data': data, 'error': None, 'timestamp': time.time()}
            app.logger.info(f"TTS pre-generated {len(data)} bytes for message {message_id}")
        except Exception as e:
            cache[message_id] = {'status': 'failed', 'data': None, 'error': str(e), 'timestamp': time.time()}
            app.logger.error(f"TTS pre-generation failed for {message_id}: {e}")

    asyncio.run_coroutine_threadsafe(_synth(), wrapper.loop)


@app.route('/api/get-messages', methods=['GET'])
@agent_or_login_required
def get_messages():
    """Get new messages from the session (polling endpoint)"""
    session_token = request.args.get('session_token')

    session = get_session(session_token)
    if not session:
        print(f"[get_messages] Invalid session_token={session_token}")
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session',
            'active_sessions_count': len(active_sessions)
        }), 400

    messages = []
    full_history = request.args.get('full', 'false').lower() == 'true'
    has_user_buffer = session.user and (
        hasattr(session.user, 'get_new_messages') or
        hasattr(session.user, '_message_buffer') or
        hasattr(session.user, 'get_and_clear_messages')
    )
    if has_user_buffer and full_history:
        # Reconnect: replay full chat_history so the client can re-render everything
        messages = [
            {
                'id': m.id,
                'role': m.role,
                'content': m.content,
                'type': m.type,
                'timestamp': m.timestamp.isoformat(),
            }
            for m in session.chat_history
            if m.type == 'conversation'
        ]
    elif has_user_buffer:
        if hasattr(session.user, 'get_new_messages'):
            messages = session.user.get_new_messages() or []
        elif hasattr(session.user, '_message_buffer'):
            lock = getattr(session.user, '_lock', None)
            if lock:
                with lock:
                    messages = list(getattr(session.user, '_message_buffer', []))
            else:
                messages = list(getattr(session.user, '_message_buffer', []))
        elif hasattr(session.user, 'get_and_clear_messages'):
            messages = session.user.get_and_clear_messages() or []
    else:
        # Agent mode: UserAgent has no buffer — serve new messages directly from chat_history
        if full_history:
            chat_history_offsets[session_token] = 0
        offset = chat_history_offsets.get(session_token, 0)
        new_msgs = session.chat_history[offset:]
        chat_history_offsets[session_token] = offset + len(new_msgs)
        messages = [
            {
                'id': m.id,
                'role': m.role,
                'content': m.content,
                'type': m.type,
                'timestamp': m.timestamp.isoformat(),
            }
            for m in new_msgs
            if m.type == 'conversation'
        ]

    is_session_done = session.session_completed

    if not is_session_done:
        current_turns = getattr(session, 'turns', 0)
        max_turns = getattr(session, 'max_turns', float('inf'))
        if current_turns is not None and max_turns is not None and current_turns >= max_turns:
            is_session_done = True
        elif not session.session_in_progress and len(session.chat_history) > 0:
            is_session_done = True

    # Session completion is signaled via data.session_completed in the JSON response;
    # the frontend renders its own end-of-session banner.

    # Pre-start TTS generation for interviewer messages as soon as they are delivered,
    # so audio is ready (or nearly ready) by the time the client explicitly requests it.
    wrapper = get_session_wrapper(session_token)
    if wrapper and hasattr(wrapper, 'loop'):
        for msg in messages:
            if msg.get('role') in ('Interviewer', 'assistant') and msg.get('id') and msg.get('content'):
                _prestart_tts(session_token, msg['id'], msg['content'], wrapper)

    return jsonify({
        'success': True,
        'messages': messages,
        'session_active': session.session_in_progress,
        'session_completed': is_session_done
    })

@app.route('/api/acknowledge-messages', methods=['POST'])
@agent_or_login_required
def acknowledge_messages():
    """Mark messages as acknowledged by the client"""
    data = request.json
    session_token = data.get('session_token')
    message_ids = data.get('message_ids', [])

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if session.user and hasattr(session.user, '_message_buffer'):
        lock = getattr(session.user, '_lock', None)
        if lock:
            with lock:
                buffer = getattr(session.user, '_message_buffer', [])
                session.user._message_buffer = [
                    m for m in buffer 
                    if m.get('id') not in message_ids
                ]
        
    return jsonify({'success': True})

@app.route('/api/get-voice-response', methods=['GET'])
@login_required  # PROTECTED
def get_voice_response():
    """Get the latest interviewer message as voice audio"""
    session_token = request.args.get('session_token')
    message_id = request.args.get('message_id')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not message_id:
        return jsonify({
            'success': False,
            'error': 'message_id required'
        }), 400

    target_msg = None
    for m in session.chat_history:
        if hasattr(m, 'id') and m.id == message_id:
            target_msg = m
            break
    
    if not target_msg:
        return jsonify({
            'success': False,
            'error': 'Message not found'
        }), 404

    cache = session_audio_cache.setdefault(session_token, {})
    entry = cache.get(message_id)

    # Check cache entry status
    if isinstance(entry, dict):
        # New format: {'status': 'pending'|'ready'|'failed', 'data': bytes|None, 'error': str|None}
        status = entry.get('status')
        if status == 'ready' and entry.get('data'):
            return Response(entry['data'], mimetype='audio/mpeg')
        elif status == 'failed':
            return jsonify({
                'success': False,
                'error': f"TTS generation failed: {entry.get('error', 'Unknown error')}"
            }), 500
        elif status == 'pending':
            return ('', 202)  # Still generating
    elif isinstance(entry, (bytes, bytearray)):
        # Legacy format support
        return Response(entry, mimetype='audio/mpeg')
    elif entry == 'pending':
        # Legacy format support
        return ('', 202)

    # No cache entry exists — start generation (also triggered from get_messages,
    # but guard here in case the client requests TTS before the next poll cycle).
    wrapper = get_session_wrapper(session_token)

    if not wrapper or not hasattr(wrapper, 'loop'):
        cache[message_id] = {
            'status': 'failed',
            'data': None,
            'error': 'Session loop not available'
        }
        return jsonify({
            'success': False,
            'error': 'Session not properly initialized'
        }), 500

    _prestart_tts(session_token, message_id, target_msg.content, wrapper)

    return ('', 202)  # Tell client to poll again

@app.route('/api/stream-voice-response', methods=['GET'])
@login_required
def stream_voice_response():
    """Stream TTS audio for a message, starting playback before generation is complete."""
    from src.utils.speech.text_to_speech import CartesiaTTS
    session_token = request.args.get('session_token')
    message_id = request.args.get('message_id')

    session = get_session(session_token)
    if not session or not message_id:
        return jsonify({'success': False, 'error': 'Invalid request'}), 400

    cache = session_audio_cache.setdefault(session_token, {})
    entry = cache.get(message_id)

    # Serve from cache if already ready (avoids hitting Cartesia twice on replay)
    if isinstance(entry, dict) and entry.get('status') == 'ready' and entry.get('data'):
        mimetype = entry.get('mimetype', 'audio/mpeg')
        return Response(entry['data'], mimetype=mimetype)

    # Find message text
    target_msg = next(
        (m for m in session.chat_history if hasattr(m, 'id') and m.id == message_id),
        None
    )
    if not target_msg:
        return jsonify({'success': False, 'error': 'Message not found'}), 404

    if not isinstance(tts_engine, CartesiaTTS):
        # Non-Cartesia provider: kick off async generation, then stream bytes
        # once ready so the browser's <audio> element receives real audio data.
        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            _prestart_tts(session_token, message_id, target_msg.content, wrapper)

        def wait_and_serve():
            deadline = time.time() + 30
            while time.time() < deadline:
                entry = cache.get(message_id)
                if isinstance(entry, dict):
                    if entry.get('status') == 'ready' and entry.get('data'):
                        yield entry['data']
                        return
                    elif entry.get('status') == 'failed':
                        return
                time.sleep(0.2)

        return Response(stream_with_context(wait_and_serve()), mimetype='audio/mpeg')

    # Mark as streaming so _prestart_tts skips this message
    cache[message_id] = {'status': 'streaming', 'data': None, 'error': None, 'timestamp': time.time()}

    def generate():
        chunks = []
        try:
            for chunk in tts_engine.stream_tts_chunks(target_msg.content):
                chunks.append(chunk)
                yield chunk
            full_audio = b''.join(chunks)
            cache[message_id] = {
                'status': 'ready', 'data': full_audio,
                'mimetype': 'audio/mpeg', 'error': None, 'timestamp': time.time()
            }
            app.logger.info(f"Streamed and cached {len(full_audio)} bytes for message {message_id}")
        except Exception as e:
            cache[message_id] = {'status': 'failed', 'data': None, 'error': str(e), 'timestamp': time.time()}
            app.logger.error(f"Streaming TTS failed for {message_id}: {e}")

    return Response(stream_with_context(generate()), mimetype='audio/mpeg')

@app.route('/api/end-session', methods=['POST'])
@agent_or_login_required
def end_session():
    """End the interview session - background tasks will complete gracefully"""
    data = request.json
    session_token = data.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # End the session (marks as not in progress, stops new tasks)
    session.end_session()

    app.logger.info(f"Session {session_token} ended by user, background tasks will complete")

    # Don't remove from active_sessions yet - let background tasks complete
    # They'll be cleaned up when session.run() completes or by timeout cleanup

    return jsonify({
        'success': True,
        'message': 'Session ending, background tasks will complete shortly',
        'session_id': session.session_id,
        'user_id': session.user_id
    })

@app.route('/api/session-status', methods=['GET'])
@login_required  # PROTECTED
def session_status():
    """Get current session status including background task progress"""
    session_token = request.args.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # Get background task count if available
    background_tasks_count = 0
    if hasattr(session, '_background_tasks'):
        try:
            import asyncio
            # Try to get count safely
            if hasattr(session, '_background_tasks_lock'):
                # Can't acquire lock in sync context, just get len
                background_tasks_count = len(session._background_tasks)
        except:
            background_tasks_count = 0

    return jsonify({
        'success': True,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'background_tasks_remaining': background_tasks_count,
        'message_count': len(session.chat_history),
        'session_id': session.session_id,
        'user_id': session.user_id
    })

@app.route('/api/session-state', methods=['GET'])
@agent_or_login_required
def session_state():
    """Return serialized live session state for the visualizer"""
    session_token = request.args.get('session_token')
    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    agenda = getattr(iv, 'session_agenda', None)
    memory_bank = getattr(iv, 'memory_bank', None)

    # --- Topics & subtopics ---
    topics = []
    if agenda and agenda.interview_topic_manager:
        for topic in agenda.interview_topic_manager:
            subtopics = []
            for st in topic.required_subtopics.values():
                subtopics.append({
                    'id': st.subtopic_id,
                    'description': st.description,
                    'is_covered': st.is_covered,
                    'notes': list(st.notes),
                    'notes_count': len(st.notes),
                    'questions_count': len(st.questions),
                    'emergent_insights_count': len(st.emergent_insights),
                    'final_summary': st.final_summary,
                    'is_emergent': False,
                    'coverage_criteria': list(st.coverage_criteria),
                    'criteria_coverage': list(st.criteria_coverage),
                })
            for st in topic.emergent_subtopics.values():
                subtopics.append({
                    'id': st.subtopic_id,
                    'description': st.description,
                    'is_covered': st.is_covered,
                    'notes': list(st.notes),
                    'notes_count': len(st.notes),
                    'questions_count': len(st.questions),
                    'emergent_insights_count': len(st.emergent_insights),
                    'final_summary': st.final_summary,
                    'is_emergent': True,
                    'coverage_criteria': list(st.coverage_criteria),
                    'criteria_coverage': list(st.criteria_coverage),
                })
            topics.append({
                'id': topic.topic_id,
                'description': topic.description,
                'subtopics': subtopics,
            })

    # --- Recent memories (last 20) ---
    memories = []
    memory_count = 0
    if memory_bank:
        all_mems = memory_bank.memories
        memory_count = len(all_mems)
        for mem in all_mems[-20:]:
            memories.append({
                'id': mem.id,
                'title': mem.title,
                'text': mem.text[:250],
                'timestamp': mem.timestamp.isoformat() if hasattr(mem.timestamp, 'isoformat') else str(mem.timestamp),
                'subtopic_links': mem.subtopic_links,
                'transcript_references': [
                    {
                        'interview_response': ref.interview_response[:200],
                        'timestamp': ref.timestamp.isoformat() if ref.timestamp else None,
                    }
                    for ref in (mem.transcript_references or [])
                ],
            })

    # --- Chat history as lightweight turn list ---
    turns = []
    for msg in iv.chat_history:
        turns.append({
            'id': getattr(msg, 'id', ''),
            'role': getattr(msg, 'role', ''),
            'content': getattr(msg, 'content', ''),
            'timestamp': getattr(msg, 'timestamp', None).isoformat() if getattr(msg, 'timestamp', None) else '',
        })

    # --- Strategic priorities ---
    strategic_priorities = agenda.strategic_priorities or {} if agenda else {}

    # --- Rollout predictions (simulated exchanges) ---
    rollout_predictions = []
    strategic_planner = getattr(iv, 'strategic_planner', None)
    if strategic_planner:
        strategic_state = getattr(strategic_planner, 'strategic_state', None)
        if strategic_state:
            for rollout in (strategic_state.rollout_predictions or []):
                rollout_predictions.append({
                    'rollout_id': rollout.rollout_id,
                    'utility_score': round(rollout.utility_score, 3),
                    'expected_coverage_delta': round(rollout.expected_coverage_delta, 3),
                    'emergence_potential': round(rollout.emergence_potential, 3),
                    'cost_estimate': rollout.cost_estimate,
                    'predicted_turns': [
                        {
                            'turn_number': t.get('turn_number', '?'),
                            'question': str(t.get('question', '')),
                            'predicted_response': str(t.get('predicted_response', '')),
                            'subtopics_covered': t.get('subtopics_covered', []),
                            'emergence_potential': t.get('emergence_potential', 0.0),
                            'strategic_rationale': str(t.get('strategic_rationale', '')),
                        }
                        for t in rollout.predicted_turns
                    ],
                })

    # --- Emergent insights ---
    emergent_insights = []
    if agenda:
        for ins in (agenda.emergent_insights or []):
            if isinstance(ins, dict):
                emergent_insights.append(ins)
            else:
                emergent_insights.append(ins.to_dict() if hasattr(ins, 'to_dict') else str(ins))

    return jsonify({
        'success': True,
        'topics': topics,
        'memories': memories,
        'memory_count': memory_count,
        'turns': turns,
        'turn_count': len(iv.chat_history),
        'user_portrait': agenda.user_portrait if agenda else {},
        'last_week_snapshot': agenda.last_week_snapshot if agenda else {},
        'strategic_priorities': strategic_priorities,
        'rollout_predictions': rollout_predictions,
        'emergent_insights': emergent_insights,
        'session_type': getattr(iv, 'session_type', 'intake'),
        'session_completed': getattr(iv, 'session_completed', False),
        'session_in_progress': getattr(iv, 'session_in_progress', False),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    })

@app.route('/api/debug-session', methods=['GET'])
@login_required  # PROTECTED
def debug_session():
    """Development-only: return session internals"""
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session', 'active_sessions_count': len(active_sessions)}), 400

    session = wrapper.interview_session

    last_msgs = []
    for m in session.chat_history[-20:]:
        last_msgs.append({
            'id': getattr(m, 'id', None),
            'role': getattr(m, 'role', None),
            'content': getattr(m, 'content', None),
            'timestamp': getattr(m, 'timestamp', None).isoformat() if getattr(m, 'timestamp', None) else None,
        })

    user_buffer = []
    user = session.user
    if hasattr(user, '_message_buffer'):
        try:
            lock = getattr(user, '_lock', None)
            if lock:
                lock.acquire()
            user_buffer = list(getattr(user, '_message_buffer', []))
        finally:
            if lock:
                lock.release()

    return jsonify({
        'success': True,
        'session_id': session.session_id,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'message_count': len(session.chat_history),
        'chat_history': last_msgs,
        'user_buffer': user_buffer,
        'active_sessions_count': len(active_sessions)
    })

@app.route('/process_audio', methods=['POST'])
@login_required  # PROTECTED
def process_audio():
    """Compatibility route for older speech_chat.html template"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400

    user_id = get_current_user().id
    
    if not session_token:
        interview_session, session_token = create_interview_session(user_id=user_id)
    else:
        interview_session = get_session(session_token)
        if not interview_session:
            interview_session, session_token = create_interview_session(user_id=user_id)

    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}.wav")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)
        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(
                wrapper.interview_session.user.add_user_message, 
                transcribed_text
            )
        else:
            interview_session.user.add_user_message(transcribed_text)

        bot_reply = wait_for_agent_response(interview_session, timeout=15.0)
        
        last_messages_by_session[session_token] = {
            'user_message': transcribed_text,
            'bot_reply': bot_reply or ''
        }

        if bot_reply:
            out_path = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
            generate_speech_from_text(bot_reply, out_path)
            audio_bytes = out_path.read_bytes()
            out_path.unlink(missing_ok=True)
            return Response(audio_bytes, mimetype='audio/mpeg')

        return jsonify({
            'success': True, 
            'user_message': transcribed_text, 
            'bot_reply': bot_reply
        }), 200

    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

@app.route('/get_last_messages', methods=['GET'])
@login_required  # PROTECTED
def get_last_messages():
    """Get last messages for session"""
    session_token = request.args.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    msgs = last_messages_by_session.get(session_token, {})
    return jsonify({
        'success': True,
        'user_message': msgs.get('user_message', ''),
        'bot_reply': msgs.get('bot_reply', '')
    })

# =============================================================================
# HEALTH CHECK - NOT PROTECTED (for monitoring)
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - no login required for monitoring"""
    current_time = time.time()
    session_ages = []
    for wrapper in active_sessions.values():
        age_minutes = (current_time - wrapper.created_at) / 60
        session_ages.append(age_minutes)
    
    avg_age = sum(session_ages) / len(session_ages) if session_ages else 0
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'avg_session_age_minutes': round(avg_age, 2),
        'tts_provider': TTS_PROVIDER,
        'tts_voice': TTS_VOICE,
        'uptime_seconds': round(current_time - START_TIME, 2)
    })

# =============================================================================
# VOICE PROCESSING UTILITIES
# =============================================================================

def generate_speech_from_text(text: str, output_path: Path) -> Path:
    """Generate speech audio from text using TTS"""
    global tts_engine
    if tts_engine is None:
        raise NotImplementedError("TTS engine is not configured")
    os.makedirs(os.path.dirname(str(output_path)) or '.', exist_ok=True)
    result_path = tts_engine.text_to_speech(text=text, output_path=str(output_path))
    return Path(result_path)

def transcribe_audio_to_text(audio_path: Path) -> str:
    """Transcribe audio file to text using speech recognition"""
    global stt_engine
    if stt_engine is None:
        raise RuntimeError("STT engine is not initialized. Check speech_to_text setup.")
    return stt_engine.transcribe(str(audio_path))

def wait_for_agent_response(session, timeout: float = 60.0, poll_interval: float = 0.5):
    """Wait for the Interviewer/Agent to produce an output by peeking at the buffer
    without consuming it, so the frontend polling still receives the message."""
    elapsed = 0.0
    import time as _time

    while elapsed < timeout:
        try:
            lock = getattr(session.user, '_lock', None)
            if lock:
                with lock:
                    msgs = list(getattr(session.user, '_message_buffer', []))
            else:
                msgs = list(getattr(session.user, '_message_buffer', []))
            interviewer_msgs = [m for m in msgs if m.get('role') == 'Interviewer']
            if interviewer_msgs:
                return interviewer_msgs[-1].get('content')
        except Exception:
            pass
        _time.sleep(poll_interval)
        elapsed += poll_interval
    return None

# =============================================================================
# SESSION CLEANUP
# =============================================================================

def cleanup_old_sessions():
    """Remove sessions older than timeout threshold and clean stale audio cache"""
    current_time = time.time()
    to_remove = []

    for token, wrapper in list(active_sessions.items()):
        age = current_time - wrapper.created_at
        if age > SESSION_TIMEOUT_SECONDS:
            to_remove.append(token)
            session_audio_cache.pop(token, None)
            last_messages_by_session.pop(token, None)

    for token in to_remove:
        wrapper = active_sessions.pop(token, None)
        if wrapper:
            print(f"[Cleanup] Removed session {token} (age: {age/60:.1f}min, user: {wrapper.user_id})")

    if to_remove:
        print(f"[Cleanup] Removed {len(to_remove)} old sessions. Active: {len(active_sessions)}")

    # Clean up stale pending audio cache entries (older than 5 minutes)
    audio_cleaned = 0
    for session_token, cache in list(session_audio_cache.items()):
        for message_id, entry in list(cache.items()):
            if isinstance(entry, dict):
                entry_age = current_time - entry.get('timestamp', current_time)
                # Remove pending entries older than 5 minutes (likely failed)
                if entry.get('status') == 'pending' and entry_age > 300:
                    cache.pop(message_id, None)
                    audio_cleaned += 1
                # Remove failed entries older than 1 hour
                elif entry.get('status') == 'failed' and entry_age > 3600:
                    cache.pop(message_id, None)
                    audio_cleaned += 1

    if audio_cleaned > 0:
        print(f"[Cleanup] Removed {audio_cleaned} stale audio cache entries")

def start_cleanup_thread():
    """Start background thread for session cleanup"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                cleanup_old_sessions()
            except Exception as e:
                print(f"[Cleanup] Error: {e}")
    
    t = threading.Thread(target=cleanup_loop, daemon=True, name="SessionCleanup")
    t.start()
    print("[Cleanup] Started session cleanup thread")

# =============================================================================
# MAIN
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Flask Interview Session Web Application'
    )
    parser.add_argument('--user-id', type=str, help='Default user ID')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--additional_context_path', default=None)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--max_turns', type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.restart and args.user_id:
        os.system(f"rm -rf {os.getenv('LOGS_DIR')}/{args.user_id}")
        os.system(f"rm -rf {os.getenv('DATA_DIR')}/{args.user_id}")
        print(f"Cleared data for user {args.user_id}")
    
    config.default_user_id = args.user_id if args.user_id else "web_guest"
    config.host = args.host
    config.port = args.port
    config.debug = args.debug
    config.restart = args.restart
    config.max_turns = args.max_turns
    config.additional_context_path = args.additional_context_path
    
    start_cleanup_thread()
    
    print("\n" + "="*70)
    print("Flask Interview Session Server - Multi-User Mode")
    print("="*70)
    print(f"🌐 Host:              {config.host}")
    print(f"🔌 Port:              {config.port}")
    print(f"🐛 Debug:             {config.debug}")
    print(f"🔐 Authentication:    Enabled")
    print(f"🧹 Session Cleanup:   Every 5 minutes (timeout: {SESSION_TIMEOUT_SECONDS/60:.0f} min)")
    print(f"🗣️  TTS Provider:      {TTS_PROVIDER} ({TTS_VOICE})")
    print("="*70)
    print(f"\n📍 Login at: http://{config.host}:{config.port}/login")
    print(f"📊 Health check: http://{config.host}:{config.port}/health")
    print("="*70 + "\n")
    
    if not config.debug:
        print("⚠️  For production, use: gunicorn -w 2 --threads 4 -b 0.0.0.0:8080 flask_app:app\n")
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        use_reloader=False,
        threaded=True
    )
