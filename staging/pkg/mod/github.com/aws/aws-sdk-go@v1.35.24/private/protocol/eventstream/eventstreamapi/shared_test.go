package eventstreamapi

import (
	"bytes"
	"encoding/hex"
	"time"

	"github.com/aws/aws-sdk-go/private/protocol"
	"github.com/aws/aws-sdk-go/private/protocol/eventstream"
)

type mockChunkSigner struct {
	signature string
	err       error
}

func (m mockChunkSigner) GetSignature(_, _ []byte, _ time.Time) ([]byte, error) {
	return mustDecodeBytes(hex.DecodeString(m.signature)), m.err
}

type eventStructured struct {
	_ struct{} `type:"structure"`

	String *string          `type:"string"`
	Number *int64           `type:"long"`
	Nested *eventStructured `type:"structure"`
}

func (e *eventStructured) MarshalEvent(pm protocol.PayloadMarshaler) (eventstream.Message, error) {
	var msg eventstream.Message
	msg.Headers.Set(MessageTypeHeader, eventstream.StringValue(EventMessageType))

	var buf bytes.Buffer
	if err := pm.MarshalPayload(&buf, e); err != nil {
		return eventstream.Message{}, err
	}

	msg.Payload = buf.Bytes()

	return msg, nil
}

func (e *eventStructured) UnmarshalEvent(pm protocol.PayloadUnmarshaler, msg eventstream.Message) error {
	return pm.UnmarshalPayload(bytes.NewReader(msg.Payload), e)
}

func mustDecodeBytes(b []byte, err error) []byte {
	if err != nil {
		panic(err)
	}

	return b
}

func swapTimeNow(f func() time.Time) func() {
	if f == nil {
		return func() {}
	}

	timeNow = f
	return func() {
		timeNow = time.Now
	}
}
