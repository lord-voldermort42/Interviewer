
import os
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import VertexAI

from src.utils.llm.models.data import ModelResponse
from src.utils.llm.models.claude import ClaudeVertexEngine, claude_vertex_model_mapping
from src.utils.llm.models.gemini import GeminiVertexEngine, gemini_models
from src.utils.llm.models.deepseek import DeepSeekEngine, deepseek_models
from src.utils.llm.models.vllm import VLLMEngine



engine_constructor = {
    "gpt-4.1": ChatOpenAI,
    "gpt-4.1-mini": ChatOpenAI,
    "gpt-4.1-nano": ChatOpenAI,
    "gpt-5.1": ChatOpenAI,
    "gpt-5-mini": ChatOpenAI,
    "gpt-5.4-mini": ChatOpenAI,
    "gpt-4o-mini-2024-07-18": ChatOpenAI,
    "gpt-3.5-turbo-0125": ChatOpenAI,
    "gpt-4o": ChatOpenAI,
    "meta-llama/Llama-3.1-8B-Instruct": ChatTogether,
    "meta-llama/Llama-3.1-70B-Instruct": ChatTogether
}

azure_engine_constructor = {
    "gpt-4.1-mini": AzureChatOpenAI,
    "gpt-4.1": AzureChatOpenAI,
    "gpt-4.1-nano": AzureChatOpenAI,
    "gpt-4o": AzureChatOpenAI,
    "gpt-4o-mini-2024-07-18": AzureChatOpenAI,
    "gpt-3.5-turbo-0125": AzureChatOpenAI,
}

def get_engine(model_name, **kwargs):
    """
    Creates and returns a language model engine based on the specified model name.

    Args:
        model_name (str): Name of the model to initialize. Supported models:
            - OpenAI models: gpt-4o-mini, gpt-3.5-turbo-0125, gpt-4o
            - Llama models: meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.1-70B-Instruct
            - DeepSeek models: deepseek-ai/DeepSeek-V3 (671B parameter model)
            - Claude models: via Vertex AI
            - Gemini models: via Vertex AI
            - vLLM models: prefix with "vllm:" (e.g., vllm:meta-llama/Llama-3.1-8B-Instruct)
                Requires VLLM_BASE_URL environment variable
        **kwargs: Additional keyword arguments to pass to the model constructor.
            - temperature: Float between 0 and 1 (default: 0.0)
            - max_tokens/max_output_tokens: Maximum number of tokens in the response (default: 4096)
                Note: This will be mapped to the appropriate parameter name for each model:
                - OpenAI/Llama/DeepSeek/vLLM: max_tokens
                - Gemini: max_output_tokens
                - Claude: max_tokens_to_sample

    Returns:
        LangChain chat model instance or custom engine configured with the specified parameters
        All engines now return ModelResponse objects with token usage metadata populated
    """
    # Set default temperature if not provided
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.0

    # Standardize max token handling
    max_tokens = kwargs.pop("max_tokens", None)
    max_output_tokens = kwargs.pop("max_output_tokens", None)
    max_tokens_to_sample = kwargs.pop("max_tokens_to_sample", None)
    
    # Use the first non-None value in order of precedence
    token_limit = max_output_tokens or max_tokens or max_tokens_to_sample or 8192
        
    if model_name == "gpt-4o-mini":
        model_name = "gpt-4o-mini-2024-07-18"
    
    # Handle Claude models via Vertex AI
    if model_name in claude_vertex_model_mapping or "claude" in model_name:
        kwargs["max_tokens_to_sample"] = token_limit
        return ClaudeVertexEngine(model_name=model_name, **kwargs)
    
    # Handle Gemini models via Vertex AI
    if model_name in gemini_models or "gemini" in model_name:
        kwargs["max_output_tokens"] = token_limit
        return GeminiVertexEngine(model_name=model_name, **kwargs)
        
    # Handle DeepSeek models
    if model_name in deepseek_models or "deepseek" in model_name.lower():
        kwargs["max_tokens"] = token_limit
        return DeepSeekEngine(model_name=model_name, **kwargs)

    # Handle vLLM models (identified by vllm: prefix)
    if model_name.startswith("vllm:"):
        # Extract the actual model name after the vllm: prefix
        actual_model_name = model_name[5:]  # Remove "vllm:" prefix
        kwargs["max_tokens"] = token_limit
        return VLLMEngine(model_name=actual_model_name, **kwargs)

    # gpt-5.x models require max_completion_tokens instead of max_tokens
    # and only support the default temperature of 1
    if model_name.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = token_limit
        kwargs["temperature"] = 1
    else:
        kwargs["max_tokens"] = token_limit
    kwargs["model_name"] = model_name

    # Use AzureOpenAI if endpoint env var is set
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint and model_name in azure_engine_constructor:
        kwargs["azure_deployment"] = model_name
        kwargs["azure_endpoint"] = azure_endpoint
        kwargs["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        kwargs["api_version"] = os.getenv("OPENAI_API_VERSION", "2024-02-01")
        kwargs.pop("model_name", None)  # AzureChatOpenAI uses azure_deployment, not model_name
        return azure_engine_constructor[model_name](**kwargs)

    return engine_constructor[model_name](**kwargs)

def invoke_engine(engine, prompt, **kwargs) -> ModelResponse:
    """
    Simple wrapper to invoke a language model engine and return its response.

    Args:
        engine: The language model engine to use
        prompt: The input prompt to send to the model
        **kwargs: Additional keyword arguments for the model invocation

    Returns:
        ModelResponse: The model's response with content and token usage metadata
    """
    response = engine.invoke(prompt, **kwargs)

    # If the engine is a custom engine (returns ModelResponse already), return it directly
    if isinstance(response, ModelResponse):
        return response

    # For LangChain models, wrap the response and extract token usage
    # Newer OpenAI models may use native JSON tool calling even without tools bound,
    # resulting in empty response.content. Convert native tool_calls to XML format.
    content = response.content
    if not content and hasattr(response, 'tool_calls') and response.tool_calls:
        xml_parts = ["<tool_calls>"]
        for tc in response.tool_calls:
            tool_name = tc.get('name', '')
            args = tc.get('args', {})
            xml_parts.append(f"<{tool_name}>")
            for k, v in args.items():
                xml_parts.append(f"<{k}>{v}</{k}>")
            xml_parts.append(f"</{tool_name}>")
        xml_parts.append("</tool_calls>")
        content = "\n".join(xml_parts)
    model_response = ModelResponse(content)

    # Extract token usage from LangChain response metadata if available
    if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
        model_response.response_metadata = {
            'token_usage': response.response_metadata['token_usage']
        }

    return model_response
