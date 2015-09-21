package types

import (
	"testing"
)

func TestEnumParserBool(t *testing.T) {
	for _, tt := range []struct {
		val string
		res bool
		ok  bool
	}{
		{val: "tRuE", res: true, ok: true},
		{val: "False", res: false, ok: true},
		{val: "t", ok: false},
	} {
		b, err := ParseBool(tt.val)
		switch {
		case tt.ok && err != nil:
			t.Errorf("%q: got error %v, want %v", tt.val, err, tt.res)
		case !tt.ok && err == nil:
			t.Errorf("%q: got %v, want error", tt.val, b)
		case tt.ok && b != tt.res:
			t.Errorf("%q: got %v, want %v", tt.val, b, tt.res)
		default:
			t.Logf("%q: got %v, %v", tt.val, b, err)
		}
	}
}
