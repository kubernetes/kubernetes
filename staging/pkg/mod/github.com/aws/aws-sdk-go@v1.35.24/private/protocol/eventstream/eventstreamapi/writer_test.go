// +build go1.7

package eventstreamapi

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream/eventstreamtest"
	"github.com/aws/aws-sdk-go/private/protocol/restjson"
)

func TestEventWriter(t *testing.T) {
	cases := map[string]struct {
		Event         Marshaler
		EncodeWrapper func(e Encoder) Encoder
		TimeFunc      func() time.Time
		Expect        eventstream.Message
		NestedExpect  *eventstream.Message
	}{
		"structured event": {
			Event: &eventStructured{
				String: aws.String("stringfield"),
				Number: aws.Int64(123),
				Nested: &eventStructured{
					String: aws.String("fieldstring"),
					Number: aws.Int64(321),
				},
			},
			Expect: eventstream.Message{
				Headers: eventstream.Headers{
					eventMessageTypeHeader,
					eventstream.Header{
						Name:  EventTypeHeader,
						Value: eventstream.StringValue("eventStructured"),
					},
				},
				Payload: []byte(`{"String":"stringfield","Number":123,"Nested":{"String":"fieldstring","Number":321}}`),
			},
		},
		"signed event": {
			Event: &eventStructured{
				String: aws.String("stringfield"),
				Number: aws.Int64(123),
				Nested: &eventStructured{
					String: aws.String("fieldstring"),
					Number: aws.Int64(321),
				},
			},
			EncodeWrapper: func(e Encoder) Encoder {
				return NewSignEncoder(
					&mockChunkSigner{
						signature: "524f1d03d1d81e94a099042736d40bd9681b867321443ff58a4568e274dbd83bff",
					},
					e,
				)
			},
			TimeFunc: func() time.Time {
				return time.Date(2019, 1, 27, 22, 37, 54, 0, time.UTC)
			},
			Expect: eventstream.Message{
				Headers: eventstream.Headers{
					{
						Name:  DateHeader,
						Value: eventstream.TimestampValue(time.Date(2019, 1, 27, 22, 37, 54, 0, time.UTC)),
					},
					{
						Name: ChunkSignatureHeader,
						Value: eventstream.BytesValue(mustDecodeBytes(
							hex.DecodeString("524f1d03d1d81e94a099042736d40bd9681b867321443ff58a4568e274dbd83bff"),
						)),
					},
				},
				Payload: mustDecodeBytes(base64.StdEncoding.DecodeString(
					`AAAAmAAAADSl4EcNDTptZXNzYWdlLXR5cGUHAAVldmVudAs6ZXZlbnQtdHlwZQcAD2V2ZW50U3RydWN0dXJlZHsiU3RyaW5nIjoic3RyaW5nZmllbGQiLCJOdW1iZXIiOjEyMywiTmVzdGVkIjp7IlN0cmluZyI6ImZpZWxkc3RyaW5nIiwiTnVtYmVyIjozMjF9fdVW3Ow=`,
				)),
			},
		},
	}

	var marshalers request.HandlerList
	marshalers.PushBackNamed(restjson.BuildHandler)

	var stream bytes.Buffer

	decodeBuf := make([]byte, 1024)
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			defer swapTimeNow(c.TimeFunc)()

			stream.Reset()

			var encoder Encoder
			encoder = eventstream.NewEncoder(&stream, eventstream.EncodeWithLogger(t))
			if c.EncodeWrapper != nil {
				encoder = c.EncodeWrapper(encoder)
			}

			eventWriter := NewEventWriter(encoder,
				protocol.HandlerPayloadMarshal{
					Marshalers: marshalers,
				},
				func(event Marshaler) (string, error) {
					return "eventStructured", nil
				},
			)

			decoder := eventstream.NewDecoder(&stream)

			if err := eventWriter.WriteEvent(c.Event); err != nil {
				t.Fatalf("expect no write error, got %v", err)
			}

			msg, err := decoder.Decode(decodeBuf)
			if err != nil {
				t.Fatalf("expect no decode error got, %v", err)
			}

			eventstreamtest.AssertMessageEqual(t, c.Expect, msg)
		})
	}
}

func BenchmarkEventWriter(b *testing.B) {
	var marshalers request.HandlerList
	marshalers.PushBackNamed(restjson.BuildHandler)

	var stream bytes.Buffer
	encoder := eventstream.NewEncoder(&stream)
	eventWriter := NewEventWriter(encoder,
		protocol.HandlerPayloadMarshal{
			Marshalers: marshalers,
		},
		func(event Marshaler) (string, error) {
			return "eventStructured", nil
		},
	)

	event := &eventStructured{
		String: aws.String("stringfield"),
		Number: aws.Int64(123),
		Nested: &eventStructured{
			String: aws.String("fieldstring"),
			Number: aws.Int64(321),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := eventWriter.WriteEvent(event); err != nil {
			b.Fatalf("expect no write error, got %v", err)
		}
	}
}
