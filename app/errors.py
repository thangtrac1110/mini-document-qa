"""Expected application errors. The API layer maps each subclass to an HTTP status."""


class AppError(Exception):
    """Base class for errors the application expects and reports to the caller."""


class PDFNotFoundError(AppError):
    pass


class InvalidPDFError(AppError):
    """The input is not a readable, unencrypted PDF."""


class NoExtractableTextError(AppError):
    """The PDF opened correctly but no page contains extractable text (e.g. a scanned PDF)."""


class DuplicateDocumentError(AppError):
    def __init__(self, existing_document_id: str):
        super().__init__(f"Identical file already ingested as document {existing_document_id}")
        self.existing_document_id = existing_document_id


class DocumentNotFoundError(AppError):
    def __init__(self, document_id: str):
        super().__init__(f"Document not found: {document_id}")
        self.document_id = document_id


class EmptyQueryError(AppError):
    """The query contains no searchable tokens."""


class InvalidParameterError(AppError):
    pass


class LLMError(AppError):
    """Base class for answer-generation failures. Messages never include credentials."""


class LLMNotConfiguredError(LLMError):
    pass


class LLMConfigurationError(LLMError):
    """The provider rejected our credentials or permissions."""


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMProviderError(LLMError):
    """Network failure or an unexpected error status from the provider."""


class LLMRefusalError(LLMError):
    pass


class MalformedLLMResponseError(LLMError):
    """The model output was truncated, not JSON, or did not match the answer schema."""
