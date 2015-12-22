package escape

import (
	"reflect"
	"testing"
)

func TestUnescape(t *testing.T) {
	tests := []struct {
		in  []byte
		out []byte
	}{
		{
			[]byte(nil),
			[]byte(nil),
		},

		{
			[]byte(""),
			[]byte(nil),
		},

		{
			[]byte("\\,\\\"\\ \\="),
			[]byte(",\" ="),
		},

		{
			[]byte("\\\\"),
			[]byte("\\\\"),
		},

		{
			[]byte("plain and simple"),
			[]byte("plain and simple"),
		},
	}

	for ii, tt := range tests {
		got := Unescape(tt.in)
		if !reflect.DeepEqual(got, tt.out) {
			t.Errorf("[%d] Unescape(%#v) = %#v, expected %#v", ii, string(tt.in), string(got), string(tt.out))
		}
	}
}
