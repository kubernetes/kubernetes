package eventstream

import (
	"bytes"
	"encoding/hex"
	"reflect"
	"testing"
)

func TestEncoder_Encode(t *testing.T) {
	cases, err := readPositiveTests("testdata")
	if err != nil {
		t.Fatalf("failed to load positive tests, %v", err)
	}

	for _, c := range cases {
		var w bytes.Buffer
		encoder := NewEncoder(&w)

		err = encoder.Encode(c.Decoded.Message())
		if err != nil {
			t.Fatalf("%s, failed to encode message, %v", c.Name, err)
		}

		if e, a := c.Encoded, w.Bytes(); !reflect.DeepEqual(e, a) {
			t.Errorf("%s, expect:\n%v\nactual:\n%v\n", c.Name,
				hex.Dump(e), hex.Dump(a))
		}
	}
}

func BenchmarkEncode(b *testing.B) {
	var w bytes.Buffer
	encoder := NewEncoder(&w)
	msg := Message{
		Headers: Headers{
			{Name: "event-id", Value: Int16Value(123)},
		},
		Payload: []byte(`{"abc":123}`),
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		err := encoder.Encode(msg)
		if err != nil {
			b.Fatal(err)
		}
	}
}
