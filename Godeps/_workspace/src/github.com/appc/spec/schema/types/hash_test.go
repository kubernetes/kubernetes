package types

import (
	"encoding/json"
	"testing"
)

func TestMarshalHash(t *testing.T) {
	tests := []struct {
		typ string
		val string

		wout string
	}{
		{
			"sha512",
			"abcdefghi",

			`"sha512-abcdefghi"`,
		},
		{
			"sha512",
			"06c733b1838136838e6d2d3e8fa5aea4c7905e92",

			`"sha512-06c733b1838136838e6d2d3e8fa5aea4c7905e92"`,
		},
	}
	for i, tt := range tests {
		h := Hash{
			typ: tt.typ,
			Val: tt.val,
		}
		b, err := json.Marshal(h)
		if err != nil {
			t.Errorf("#%d: unexpected err=%v", i, err)
		}
		if g := string(b); g != tt.wout {
			t.Errorf("#%d: got string=%v, want %v", i, g, tt.wout)
		}
	}
}

func TestMarshalHashBad(t *testing.T) {
	tests := []struct {
		typ string
		val string
	}{
		{
			// empty value
			"sha512",
			"",
		},
		{
			// bad type
			"sha1",
			"abcdef",
		},
		{
			// empty type
			"",
			"abcdef",
		},
		{
			// empty empty
			"",
			"",
		},
	}
	for i, tt := range tests {
		h := Hash{
			typ: tt.typ,
			Val: tt.val,
		}
		g, err := json.Marshal(h)
		if err == nil {
			t.Errorf("#%d: unexpected nil err", i)
		}
		if g != nil {
			t.Errorf("#%d: unexpected non-nil bytes: %v", i, g)
		}
	}
}
