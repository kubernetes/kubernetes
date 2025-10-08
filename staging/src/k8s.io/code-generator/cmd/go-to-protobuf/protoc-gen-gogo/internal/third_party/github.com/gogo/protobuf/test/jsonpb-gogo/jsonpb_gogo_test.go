package jsonpb_gogo

import (
	"testing"

	"github.com/gogo/protobuf/jsonpb"
)

// customFieldMessage implements protobuf.Message but is not a normal generated message type.
type customFieldMessage struct {
	someField string //this is not a proto field
}

func (m *customFieldMessage) Reset() {
	m.someField = "hello"
}

func (m *customFieldMessage) String() string {
	return m.someField
}

func (m *customFieldMessage) ProtoMessage() {
}

func TestUnmarshalWithJSONPBUnmarshaler(t *testing.T) {
	rawJson := `{}`
	marshaler := &jsonpb.Marshaler{}
	msg := &customFieldMessage{someField: "Ignore me"}
	str, err := marshaler.MarshalToString(msg)
	if err != nil {
		t.Errorf("an unexpected error occurred when marshaling message: %v", err)
	}
	if str != rawJson {
		t.Errorf("marshaled JSON was incorrect: got %s, wanted %s", str, rawJson)
	}
}
