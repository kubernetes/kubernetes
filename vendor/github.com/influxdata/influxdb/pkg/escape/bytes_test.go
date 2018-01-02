package escape

import (
	"bytes"
	"reflect"
	"strings"
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

func TestAppendUnescaped(t *testing.T) {
	cases := strings.Split(strings.TrimSpace(`
normal
inv\alid
goo\"d
sp\ ace
\,\"\ \=
f\\\ x
`), "\n")

	for _, c := range cases {
		exp := Unescape([]byte(c))
		got := AppendUnescaped(nil, []byte(c))

		if !bytes.Equal(got, exp) {
			t.Errorf("AppendUnescaped failed for %#q: got %#q, exp %#q", c, got, exp)
		}
	}

}
