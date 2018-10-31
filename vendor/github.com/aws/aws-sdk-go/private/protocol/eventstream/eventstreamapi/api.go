package eventstreamapi

import (
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
)

// Unmarshaler provides the interface for unmarshaling a EventStream
// message into a SDK type.
type Unmarshaler interface {
	UnmarshalEvent(protocol.PayloadUnmarshaler, eventstream.Message) error
}

// EventStream headers with specific meaning to async API functionality.
const (
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

// EventReader provides reading from the EventStream of an reader.
type EventReader struct {
	reader  io.ReadCloser
	decoder *eventstream.Decoder

	unmarshalerForEventType func(string) (Unmarshaler, error)
	payloadUnmarshaler      protocol.PayloadUnmarshaler

	payloadBuf []byte
}

// NewEventReader returns a EventReader built from the reader and unmarshaler
// provided.  Use ReadStream method to start reading from the EventStream.
func NewEventReader(
	reader io.ReadCloser,
	payloadUnmarshaler protocol.PayloadUnmarshaler,
	unmarshalerForEventType func(string) (Unmarshaler, error),
) *EventReader {
	return &EventReader{
		reader:                  reader,
		decoder:                 eventstream.NewDecoder(reader),
		payloadUnmarshaler:      payloadUnmarshaler,
		unmarshalerForEventType: unmarshalerForEventType,
		payloadBuf:              make([]byte, 10*1024),
	}
}

// UseLogger instructs the EventReader to use the logger and log level
// specified.
func (r *EventReader) UseLogger(logger aws.Logger, logLevel aws.LogLevelType) {
	if logger != nil && logLevel.Matches(aws.LogDebugWithEventStreamBody) {
		r.decoder.UseLogger(logger)
	}
}

// ReadEvent attempts to read a message from the EventStream and return the
// unmarshaled event value that the message is for.
//
// For EventStream API errors check if the returned error satisfies the
// awserr.Error interface to get the error's Code and Message components.
//
// EventUnmarshalers called with EventStream messages must take copies of the
// message's Payload. The payload will is reused between events read.
func (r *EventReader) ReadEvent() (event interface{}, err error) {
	msg, err := r.decoder.Decode(r.payloadBuf)
	if err != nil {
		return nil, err
	}
	defer func() {
		// Reclaim payload buffer for next message read.
		r.payloadBuf = msg.Payload[0:0]
	}()

	typ, err := GetHeaderString(msg, MessageTypeHeader)
	if err != nil {
		return nil, err
	}

	switch typ {
	case EventMessageType:
		return r.unmarshalEventMessage(msg)
	case ErrorMessageType:
		return nil, r.unmarshalErrorMessage(msg)
	default:
		return nil, fmt.Errorf("unknown eventstream message type, %v", typ)
	}
}

func (r *EventReader) unmarshalEventMessage(
	msg eventstream.Message,
) (event interface{}, err error) {
	eventType, err := GetHeaderString(msg, EventTypeHeader)
	if err != nil {
		return nil, err
	}

	ev, err := r.unmarshalerForEventType(eventType)
	if err != nil {
		return nil, err
	}

	err = ev.UnmarshalEvent(r.payloadUnmarshaler, msg)
	if err != nil {
		return nil, err
	}

	return ev, nil
}

func (r *EventReader) unmarshalErrorMessage(msg eventstream.Message) (err error) {
	var msgErr messageError

	msgErr.code, err = GetHeaderString(msg, ErrorCodeHeader)
	if err != nil {
		return err
	}

	msgErr.msg, err = GetHeaderString(msg, ErrorMessageHeader)
	if err != nil {
		return err
	}

	return msgErr
}

// Close closes the EventReader's EventStream reader.
func (r *EventReader) Close() error {
	return r.reader.Close()
}

// GetHeaderString returns the value of the header as a string. If the header
// is not set or the value is not a string an error will be returned.
func GetHeaderString(msg eventstream.Message, headerName string) (string, error) {
	headerVal := msg.Headers.Get(headerName)
	if headerVal == nil {
		return "", fmt.Errorf("error header %s not present", headerName)
	}

	v, ok := headerVal.Get().(string)
	if !ok {
		return "", fmt.Errorf("error header value is not a string, %T", headerVal)
	}

	return v, nil
}
