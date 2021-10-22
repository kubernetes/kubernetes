package eventstreamapi

import (
	"fmt"

	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
)

// Unmarshaler provides the interface for unmarshaling a EventStream
// message into a SDK type.
type Unmarshaler interface {
	UnmarshalEvent(protocol.PayloadUnmarshaler, eventstream.Message) error
}

// EventReader provides reading from the EventStream of an reader.
type EventReader struct {
	decoder *eventstream.Decoder

	unmarshalerForEventType func(string) (Unmarshaler, error)
	payloadUnmarshaler      protocol.PayloadUnmarshaler

	payloadBuf []byte
}

// NewEventReader returns a EventReader built from the reader and unmarshaler
// provided.  Use ReadStream method to start reading from the EventStream.
func NewEventReader(
	decoder *eventstream.Decoder,
	payloadUnmarshaler protocol.PayloadUnmarshaler,
	unmarshalerForEventType func(string) (Unmarshaler, error),
) *EventReader {
	return &EventReader{
		decoder:                 decoder,
		payloadUnmarshaler:      payloadUnmarshaler,
		unmarshalerForEventType: unmarshalerForEventType,
		payloadBuf:              make([]byte, 10*1024),
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
	case ExceptionMessageType:
		return nil, r.unmarshalEventException(msg)
	case ErrorMessageType:
		return nil, r.unmarshalErrorMessage(msg)
	default:
		return nil, &UnknownMessageTypeError{
			Type: typ, Message: msg.Clone(),
		}
	}
}

// UnknownMessageTypeError provides an error when a message is received from
// the stream, but the reader is unable to determine what kind of message it is.
type UnknownMessageTypeError struct {
	Type    string
	Message eventstream.Message
}

func (e *UnknownMessageTypeError) Error() string {
	return "unknown eventstream message type, " + e.Type
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

func (r *EventReader) unmarshalEventException(
	msg eventstream.Message,
) (err error) {
	eventType, err := GetHeaderString(msg, ExceptionTypeHeader)
	if err != nil {
		return err
	}

	ev, err := r.unmarshalerForEventType(eventType)
	if err != nil {
		return err
	}

	err = ev.UnmarshalEvent(r.payloadUnmarshaler, msg)
	if err != nil {
		return err
	}

	var ok bool
	err, ok = ev.(error)
	if !ok {
		err = messageError{
			code: "SerializationError",
			msg: fmt.Sprintf(
				"event stream exception %s mapped to non-error %T, %v",
				eventType, ev, ev,
			),
		}
	}

	return err
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
