package eventstreamapi

import (
	"bytes"
	"fmt"
	"io"
	"testing"

	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
	"github.com/aws/aws-sdk-go/private/protocol/restjson"
)

var eventMessageTypeHeader = eventstream.Header{
	Name:  MessageTypeHeader,
	Value: eventstream.StringValue(EventMessageType),
}

func TestEventReader(t *testing.T) {
	stream := createStream(
		eventstream.Message{
			Headers: eventstream.Headers{
				eventMessageTypeHeader,
				eventstream.Header{
					Name:  EventTypeHeader,
					Value: eventstream.StringValue("eventABC"),
				},
			},
		},
		eventstream.Message{
			Headers: eventstream.Headers{
				eventMessageTypeHeader,
				eventstream.Header{
					Name:  EventTypeHeader,
					Value: eventstream.StringValue("eventEFG"),
				},
			},
		},
	)

	var unmarshalers request.HandlerList
	unmarshalers.PushBackNamed(restjson.UnmarshalHandler)

	decoder := eventstream.NewDecoder(stream)
	eventReader := NewEventReader(decoder,
		protocol.HandlerPayloadUnmarshal{
			Unmarshalers: unmarshalers,
		},
		unmarshalerForEventType,
	)

	event, err := eventReader.ReadEvent()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if event == nil {
		t.Fatalf("expect event got none")
	}

	event, err = eventReader.ReadEvent()
	if err == nil {
		t.Fatalf("expect error for unknown event, got none")
	}

	if event != nil {
		t.Fatalf("expect no event, got %T, %v", event, event)
	}
}

func TestEventReader_Error(t *testing.T) {
	stream := createStream(
		eventstream.Message{
			Headers: eventstream.Headers{
				eventstream.Header{
					Name:  MessageTypeHeader,
					Value: eventstream.StringValue(ErrorMessageType),
				},
				eventstream.Header{
					Name:  ErrorCodeHeader,
					Value: eventstream.StringValue("errorCode"),
				},
				eventstream.Header{
					Name:  ErrorMessageHeader,
					Value: eventstream.StringValue("error message occur"),
				},
			},
		},
	)

	var unmarshalers request.HandlerList
	unmarshalers.PushBackNamed(restjson.UnmarshalHandler)

	decoder := eventstream.NewDecoder(stream)
	eventReader := NewEventReader(decoder,
		protocol.HandlerPayloadUnmarshal{
			Unmarshalers: unmarshalers,
		},
		unmarshalerForEventType,
	)

	event, err := eventReader.ReadEvent()
	if err == nil {
		t.Fatalf("expect error got none")
	}

	if event != nil {
		t.Fatalf("expect no event, got %v", event)
	}

	if e, a := "errorCode: error message occur", err.Error(); e != a {
		t.Errorf("expect %v error, got %v", e, a)
	}
}

func TestEventReader_Exception(t *testing.T) {
	eventMsgs := []eventstream.Message{
		{
			Headers: eventstream.Headers{
				eventstream.Header{
					Name:  MessageTypeHeader,
					Value: eventstream.StringValue(ExceptionMessageType),
				},
				eventstream.Header{
					Name:  ExceptionTypeHeader,
					Value: eventstream.StringValue("exception"),
				},
			},
			Payload: []byte(`{"message":"exception message"}`),
		},
	}
	stream := createStream(eventMsgs...)

	var unmarshalers request.HandlerList
	unmarshalers.PushBackNamed(restjson.UnmarshalHandler)

	decoder := eventstream.NewDecoder(stream)
	eventReader := NewEventReader(decoder,
		protocol.HandlerPayloadUnmarshal{
			Unmarshalers: unmarshalers,
		},
		unmarshalerForEventType,
	)

	event, err := eventReader.ReadEvent()
	if err == nil {
		t.Fatalf("expect error got none")
	}

	if event != nil {
		t.Fatalf("expect no event, got %v", event)
	}

	et := err.(*exceptionType)
	if e, a := string(eventMsgs[0].Payload), string(et.Payload); e != a {
		t.Errorf("expect %v payload, got %v", e, a)
	}
}

func BenchmarkEventReader(b *testing.B) {
	var buf bytes.Buffer
	encoder := eventstream.NewEncoder(&buf)
	msg := eventstream.Message{
		Headers: eventstream.Headers{
			eventMessageTypeHeader,
			eventstream.Header{
				Name:  EventTypeHeader,
				Value: eventstream.StringValue("eventStructured"),
			},
		},
		Payload: []byte(`{"String":"stringfield","Number":123,"Nested":{"String":"fieldstring","Number":321}}`),
	}
	if err := encoder.Encode(msg); err != nil {
		b.Fatalf("failed to encode message, %v", err)
	}
	stream := bytes.NewReader(buf.Bytes())

	var unmarshalers request.HandlerList
	unmarshalers.PushBackNamed(restjson.UnmarshalHandler)

	decoder := eventstream.NewDecoder(stream)
	eventReader := NewEventReader(decoder,
		protocol.HandlerPayloadUnmarshal{
			Unmarshalers: unmarshalers,
		},
		unmarshalerForEventType,
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream.Seek(0, 0)

		event, err := eventReader.ReadEvent()
		if err != nil {
			b.Fatalf("expect no error, got %v", err)
		}
		if event == nil {
			b.Fatalf("expect event got none")
		}
	}
}

func unmarshalerForEventType(eventType string) (Unmarshaler, error) {
	switch eventType {
	case "eventABC":
		return &eventABC{}, nil
	case "eventStructured":
		return &eventStructured{}, nil
	case "exception":
		return &exceptionType{}, nil
	default:
		return nil, fmt.Errorf("unknown event type, %v", eventType)
	}
}

type eventABC struct {
	_ struct{}

	HeaderField string
	Payload     []byte
}

func (e *eventABC) UnmarshalEvent(
	unmarshaler protocol.PayloadUnmarshaler,
	msg eventstream.Message,
) error {
	return nil
}

func createStream(msgs ...eventstream.Message) io.Reader {
	w := bytes.NewBuffer(nil)

	encoder := eventstream.NewEncoder(w)

	for _, msg := range msgs {
		if err := encoder.Encode(msg); err != nil {
			panic("createStream failed, " + err.Error())
		}
	}

	return w
}

type exceptionType struct {
	Payload []byte
}

func (e exceptionType) Error() string {
	return fmt.Sprintf("exception error message")
}

func (e *exceptionType) UnmarshalEvent(
	unmarshaler protocol.PayloadUnmarshaler,
	msg eventstream.Message,
) error {
	e.Payload = msg.Payload
	return nil
}
