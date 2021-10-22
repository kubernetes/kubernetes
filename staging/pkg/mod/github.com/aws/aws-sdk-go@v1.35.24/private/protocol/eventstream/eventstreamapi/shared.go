package eventstreamapi

// EventStream headers with specific meaning to async API functionality.
const (
	ChunkSignatureHeader = `:chunk-signature` // chunk signature for message
	DateHeader           = `:date`            // Date header for signature

	// Message header and values
	MessageTypeHeader    = `:message-type` // Identifies type of message.
	EventMessageType     = `event`
	ErrorMessageType     = `error`
	ExceptionMessageType = `exception`

	// Message Events
	EventTypeHeader = `:event-type` // Identifies message event type e.g. "Stats".

	// Message Error
	ErrorCodeHeader    = `:error-code`
	ErrorMessageHeader = `:error-message`

	// Message Exception
	ExceptionTypeHeader = `:exception-type`
)
