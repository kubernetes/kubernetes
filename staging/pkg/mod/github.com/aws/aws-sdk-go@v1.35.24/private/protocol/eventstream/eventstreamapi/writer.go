package eventstreamapi

import (
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
)

// Marshaler provides a marshaling interface for event types to event stream
// messages.
type Marshaler interface {
	MarshalEvent(protocol.PayloadMarshaler) (eventstream.Message, error)
}

// Encoder is an stream encoder that will encode an event stream message for
// the transport.
type Encoder interface {
	Encode(eventstream.Message) error
}

// EventWriter provides a wrapper around the underlying event stream encoder
// for an io.WriteCloser.
type EventWriter struct {
	encoder          Encoder
	payloadMarshaler protocol.PayloadMarshaler
	eventTypeFor     func(Marshaler) (string, error)
}

// NewEventWriter returns a new event stream writer, that will write to the
// writer provided. Use the WriteEvent method to write an event to the stream.
func NewEventWriter(encoder Encoder, pm protocol.PayloadMarshaler, eventTypeFor func(Marshaler) (string, error),
) *EventWriter {
	return &EventWriter{
		encoder:          encoder,
		payloadMarshaler: pm,
		eventTypeFor:     eventTypeFor,
	}
}

// WriteEvent writes an event to the stream. Returns an error if the event
// fails to marshal into a message, or writing to the underlying writer fails.
func (w *EventWriter) WriteEvent(event Marshaler) error {
	msg, err := w.marshal(event)
	if err != nil {
		return err
	}

	return w.encoder.Encode(msg)
}

func (w *EventWriter) marshal(event Marshaler) (eventstream.Message, error) {
	eventType, err := w.eventTypeFor(event)
	if err != nil {
		return eventstream.Message{}, err
	}

	msg, err := event.MarshalEvent(w.payloadMarshaler)
	if err != nil {
		return eventstream.Message{}, err
	}

	msg.Headers.Set(EventTypeHeader, eventstream.StringValue(eventType))
	return msg, nil
}

//type EventEncoder struct {
//	encoder           Encoder
//	ppayloadMarshaler protocol.PayloadMarshaler
//	eventTypeFor      func(Marshaler) (string, error)
//}
//
//func (e EventEncoder) Encode(event Marshaler) error {
//	msg, err := e.marshal(event)
//	if err != nil {
//		return err
//	}
//
//	return w.encoder.Encode(msg)
//}
//
//func (e EventEncoder) marshal(event Marshaler) (eventstream.Message, error) {
//	eventType, err := w.eventTypeFor(event)
//	if err != nil {
//		return eventstream.Message{}, err
//	}
//
//	msg, err := event.MarshalEvent(w.payloadMarshaler)
//	if err != nil {
//		return eventstream.Message{}, err
//	}
//
//	msg.Headers.Set(EventTypeHeader, eventstream.StringValue(eventType))
//	return msg, nil
//}
//
//func (w *EventWriter) marshal(event Marshaler) (eventstream.Message, error) {
//	eventType, err := w.eventTypeFor(event)
//	if err != nil {
//		return eventstream.Message{}, err
//	}
//
//	msg, err := event.MarshalEvent(w.payloadMarshaler)
//	if err != nil {
//		return eventstream.Message{}, err
//	}
//
//	msg.Headers.Set(EventTypeHeader, eventstream.StringValue(eventType))
//	return msg, nil
//}
//
