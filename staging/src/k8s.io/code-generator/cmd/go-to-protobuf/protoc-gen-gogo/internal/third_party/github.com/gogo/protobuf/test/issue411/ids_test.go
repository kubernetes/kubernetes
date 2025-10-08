package issue411_test

import (
	"bytes"
	"testing"

	"github.com/gogo/protobuf/jsonpb"
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/test/issue411"
)

// Thanks to @yurishkuro for reporting this issue (#411) and providing this test case

// TraceID/SpanID fields are defined as bytes in proto, backed by custom types in Go.
// Unfortunately, that means they require manual implementations of proto & json serialization.
// To ensure that it's the same as the default protobuf serialization, file jaeger_test.proto
// contains a copy of SpanRef message without any gogo options. This test file is compiled with
// plain protoc -go_out (without gogo). This test performs proto and JSON marshaling/unmarshaling
// to ensure that the outputs of manual and standard serialization are identical.
func TestTraceSpanIDMarshalProto(t *testing.T) {
	testCases := []struct {
		name      string
		marshal   func(proto.Message) ([]byte, error)
		unmarshal func([]byte, proto.Message) error
		expected  string
	}{
		{
			name:      "protobuf",
			marshal:   proto.Marshal,
			unmarshal: proto.Unmarshal,
		},
		{
			name: "JSON",
			marshal: func(m proto.Message) ([]byte, error) {
				out := new(bytes.Buffer)
				err := new(jsonpb.Marshaler).Marshal(out, m)
				if err != nil {
					return nil, err
				}
				return out.Bytes(), nil
			},
			unmarshal: func(in []byte, m proto.Message) error {
				return jsonpb.Unmarshal(bytes.NewReader(in), m)
			},
			expected: `{"traceId":"AAAAAAAAAAIAAAAAAAAAAw==","spanId":"AAAAAAAAAAs="}`,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			o1 := issue411.Span{TraceID: issue411.NewTraceID(2, 3), SpanID: issue411.NewSpanID(11)}
			d1, err := testCase.marshal(&o1)
			if err != nil {
				t.Fatalf("marshal error: %v", err)
			}
			// test unmarshal
			var o2 issue411.Span
			err = testCase.unmarshal(d1, &o2)
			if err != nil {
				t.Fatalf("umarshal error: %v", err)
			}
			if o1.TraceID != o2.TraceID {
				t.Fatalf("TraceID: expected %v but got %v", o1, o2)
			}
			if o1.SpanID != o2.SpanID {
				t.Fatalf("SpanID: expected %v but got %v", o1, o2)
			}
		})
	}
}
