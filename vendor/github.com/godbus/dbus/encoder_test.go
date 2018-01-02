package dbus

import (
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"
)

func TestEncodeArrayOfMaps(t *testing.T) {
	tests := []struct {
		name string
		vs   []interface{}
	}{
		{
			"aligned at 8 at start of array",
			[]interface{}{
				"12345",
				[]map[string]Variant{
					{
						"abcdefg": MakeVariant("foo"),
						"cdef":    MakeVariant(uint32(2)),
					},
				},
			},
		},
		{
			"not aligned at 8 for start of array",
			[]interface{}{
				"1234567890",
				[]map[string]Variant{
					{
						"abcdefg": MakeVariant("foo"),
						"cdef":    MakeVariant(uint32(2)),
					},
				},
			},
		},
	}
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		for _, tt := range tests {
			buf := new(bytes.Buffer)
			enc := newEncoder(buf, order)
			enc.Encode(tt.vs...)

			dec := newDecoder(buf, order)
			v, err := dec.Decode(SignatureOf(tt.vs...))
			if err != nil {
				t.Errorf("%q: decode (%v) failed: %v", tt.name, order, err)
				continue
			}
			if !reflect.DeepEqual(v, tt.vs) {
				t.Errorf("%q: (%v) not equal: got '%v', want '%v'", tt.name, order, v, tt.vs)
				continue
			}
		}
	}
}
