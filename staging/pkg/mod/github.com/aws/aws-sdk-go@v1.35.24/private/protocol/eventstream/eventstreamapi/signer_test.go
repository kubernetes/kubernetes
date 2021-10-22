// +build go1.7

package eventstreamapi

import (
	"bytes"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream/eventstreamtest"
)

func TestSignEncoder(t *testing.T) {
	currentTime := time.Date(2019, 1, 27, 22, 37, 54, 0, time.UTC)

	cases := map[string]struct {
		Signer       StreamSigner
		Input        eventstream.Message
		Expect       eventstream.Message
		NestedExpect *eventstream.Message
		Err          string
	}{
		"sign message": {
			Signer: mockChunkSigner{
				signature: "524f1d03d1d81e94a099042736d40bd9681b867321443ff58a4568e274dbd83bff",
			},
			Input: eventstream.Message{
				Headers: eventstream.Headers{
					{
						Name:  "header_name",
						Value: eventstream.StringValue("header value"),
					},
				},
				Payload: []byte("payload"),
			},
			Expect: eventstream.Message{
				Headers: eventstream.Headers{
					{
						Name:  ":date",
						Value: eventstream.TimestampValue(currentTime),
					},
					{
						Name: ":chunk-signature",
						Value: eventstream.BytesValue(mustDecodeBytes(
							hex.DecodeString("524f1d03d1d81e94a099042736d40bd9681b867321443ff58a4568e274dbd83bff"),
						)),
					},
				},
				Payload: mustDecodeBytes(
					base64.StdEncoding.DecodeString(
						`AAAAMgAAABs0pv1jC2hlYWRlcl9uYW1lBwAMaGVhZGVyIHZhbHVlcGF5bG9hZH4tKFg=`,
					),
				),
			},
			NestedExpect: &eventstream.Message{
				Headers: eventstream.Headers{
					{
						Name:  "header_name",
						Value: eventstream.StringValue("header value"),
					},
				},
				Payload: []byte(`payload`),
			},
		},
		"signing error": {
			Signer: mockChunkSigner{err: fmt.Errorf("signing error")},
			Input: eventstream.Message{
				Headers: []eventstream.Header{
					{
						Name:  "header_name",
						Value: eventstream.StringValue("header value"),
					},
				},
				Payload: []byte("payload"),
			},
			Err: "signing error",
		},
	}

	origNowFn := timeNow
	timeNow = func() time.Time { return currentTime }
	defer func() { timeNow = origNowFn }()

	decodeBuf := make([]byte, 1024)
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			encoder := &mockEncoder{}
			signer := NewSignEncoder(c.Signer, encoder)

			err := signer.Encode(c.Input)
			if err == nil && len(c.Err) > 0 {
				t.Fatalf("expected error, but got nil")
			} else if err != nil && len(c.Err) == 0 {
				t.Fatalf("expected no error, but got %v", err)
			} else if err != nil && len(c.Err) > 0 && !strings.Contains(err.Error(), c.Err) {
				t.Fatalf("expected %v, but got %v", c.Err, err)
			} else if len(c.Err) > 0 {
				return
			}

			eventstreamtest.AssertMessageEqual(t, c.Expect, encoder.msgs[0], "envelope msg")

			if c.NestedExpect != nil {
				nested := eventstream.NewDecoder(bytes.NewReader(encoder.msgs[0].Payload))
				nestedMsg, err := nested.Decode(decodeBuf)
				if err != nil {
					t.Fatalf("expect no decode error got, %v", err)
				}

				eventstreamtest.AssertMessageEqual(t, *c.NestedExpect, nestedMsg, "nested msg")
			}
		})
	}
}

type mockEncoder struct {
	msgs []eventstream.Message
}

func (m *mockEncoder) Encode(msg eventstream.Message) error {
	m.msgs = append(m.msgs, msg)
	return nil
}
