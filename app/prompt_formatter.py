"""Prompt formatters for different models."""

from typing import List, Dict, Any, Union


class BasePromptFormatter:
    """Base class for prompt formatters."""
    
    def format(self, question: str) -> Union[str, List[Dict[str, str]]]:
        """
        Format a question into a model-specific prompt.
        
        Args:
            question: The input question
            
        Returns:
            Formatted prompt (string or list of message dicts)
        """
        raise NotImplementedError


class LlamaEvaluationFormatter(BasePromptFormatter):
    """
    Prompt formatter for Llama models with evaluation instructions.
    
    This formatter wraps questions in evaluation criteria for assessing
    AI assistant response quality on a 1-5 scale.
    """
    
    def format(self, question: str) -> List[Dict[str, str]]:
        """
        Format question with evaluation prompt structure.
        
        Args:
            question: The input question
            
        Returns:
            Formatted prompt as messages list
        """
        messages = [
            {
                'role': 'system',
                'content': (
                    '[Instruction]\n'
                    'Based on the question provided below, predict the score an expert evaluator would give to an AI assistant\'s response, '
                    'considering its helpfulness, relevance, adherence to facts, depth, creativity, and detail. '
                    'Your prediction should infer the level of proficiency needed to address the question effectively. '
                    'Use a scale from 1 to 5, where a higher score indicates a higher anticipated quality of response. '
                    'Provide your prediction as: "[[predicted rating]]".\n\n'
                    'Score criteria:\n'
                    '- **4-5**: The AI assistant can produce a very strong answer, showing deep understanding, creativity, detailed insight, and high relevance.\n'
                    '- **3**: The AI assistant can provide an adequate answer with moderate detail, relevance, and factual accuracy.\n'
                    '- **1-2**: The AI assistant will struggle to produce a strong answer due to the question\'s difficulty, vagueness, or the assistant\'s limitations.'
                )
            },
            {
                'role': 'user',
                'content': f'[Question]\n{question}\n\n'
            }
        ]
        
        return messages


class PlainPromptFormatter(BasePromptFormatter):
    """Simple pass-through formatter that doesn't modify the input."""
    
    def format(self, question: str) -> str:
        """
        Return the question as-is without formatting.
        
        Args:
            question: The input question
            
        Returns:
            The original question
        """
        return question


class CustomPromptFormatter(BasePromptFormatter):
    """
    Custom prompt formatter with configurable template.
    
    Example usage:
        formatter = CustomPromptFormatter(
            template="Answer this question: {question}"
        )
    """
    
    def __init__(self, template: str = "{question}"):
        """
        Initialize with a custom template.
        
        Args:
            template: Template string with {question} placeholder
        """
        self.template = template
    
    def format(self, question: str) -> str:
        """
        Format question using the custom template.
        
        Args:
            question: The input question
            
        Returns:
            Formatted prompt
        """
        return self.template.format(question=question)


# Registry of available formatters
FORMATTER_REGISTRY = {
    'llama_evaluation': LlamaEvaluationFormatter,
    'plain': PlainPromptFormatter,
    'custom': CustomPromptFormatter,
}


def get_formatter(formatter_type: str, **kwargs) -> BasePromptFormatter:
    """
    Get a prompt formatter by type.
    
    Args:
        formatter_type: Type of formatter ('llama_evaluation', 'plain', 'custom')
        **kwargs: Additional arguments for the formatter
        
    Returns:
        Prompt formatter instance
        
    Raises:
        ValueError: If formatter_type is not recognized
    """
    if formatter_type not in FORMATTER_REGISTRY:
        raise ValueError(
            f"Unknown formatter type: {formatter_type}. "
            f"Available types: {list(FORMATTER_REGISTRY.keys())}"
        )
    
    formatter_class = FORMATTER_REGISTRY[formatter_type]
    return formatter_class(**kwargs)

