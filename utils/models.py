def is_gemma_family(model_type: str) -> bool:
    """
    Checks if a given Hugging Face transformers model_type string
    belongs to the Gemma family of models.

    The model type is typically obtained from the `config.model_type`
    attribute after loading a model configuration (e.g., using AutoConfig).

    Args:
        model_type (str): The model type string (e.g., 'gemma', 'gemma2', 'llama').

    Returns:
        bool: True if the model is part of the Gemma lineage, False otherwise.
    """
    if not isinstance(model_type, str):
        return False

    # Standard model types used for Gemma and related architectures in Hugging Face
    # 'gemma': Original Gemma 1
    # 'gemma2': Gemma 2 models
    # 'gemma3': Gemma 3 models
    # 'paligemma': PaliGemma Vision-Language Models (VLM)
    gemma_family_types = {"gemma", "gemma2", "gemma3", "paligemma"}

    # The check is case-insensitive for robustness
    return model_type.lower() in gemma_family_types