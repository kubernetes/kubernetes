package types

import (
	"reflect"
	"testing"
)

func elem(p interface{}) interface{} {
	return reflect.ValueOf(p).Elem().Interface()
}

func TestParseInt(t *testing.T) {
	for _, tt := range []struct {
		val  string
		mode IntMode
		exp  interface{}
		ok   bool
	}{
		{"0", Dec, int(0), true},
		{"10", Dec, int(10), true},
		{"-10", Dec, int(-10), true},
		{"x", Dec, int(0), false},
		{"0xa", Hex, int(0xa), true},
		{"a", Hex, int(0xa), true},
		{"10", Hex, int(0x10), true},
		{"-0xa", Hex, int(-0xa), true},
		{"0x", Hex, int(0x0), true},  // Scanf doesn't require digit behind 0x
		{"-0x", Hex, int(0x0), true}, // Scanf doesn't require digit behind 0x
		{"-a", Hex, int(-0xa), true},
		{"-10", Hex, int(-0x10), true},
		{"x", Hex, int(0), false},
		{"10", Oct, int(010), true},
		{"010", Oct, int(010), true},
		{"-10", Oct, int(-010), true},
		{"-010", Oct, int(-010), true},
		{"10", Dec | Hex, int(10), true},
		{"010", Dec | Hex, int(10), true},
		{"0x10", Dec | Hex, int(0x10), true},
		{"10", Dec | Oct, int(10), true},
		{"010", Dec | Oct, int(010), true},
		{"0x10", Dec | Oct, int(0), false},
		{"10", Hex | Oct, int(0), false}, // need prefix to distinguish Hex/Oct
		{"010", Hex | Oct, int(010), true},
		{"0x10", Hex | Oct, int(0x10), true},
		{"10", Dec | Hex | Oct, int(10), true},
		{"010", Dec | Hex | Oct, int(010), true},
		{"0x10", Dec | Hex | Oct, int(0x10), true},
	} {
		typ := reflect.TypeOf(tt.exp)
		res := reflect.New(typ).Interface()
		err := ParseInt(res, tt.val, tt.mode)
		switch {
		case tt.ok && err != nil:
			t.Errorf("ParseInt(%v, %#v, %v): fail; got error %v, want ok",
				typ, tt.val, tt.mode, err)
		case !tt.ok && err == nil:
			t.Errorf("ParseInt(%v, %#v, %v): fail; got %v, want error",
				typ, tt.val, tt.mode, elem(res))
		case tt.ok && !reflect.DeepEqual(elem(res), tt.exp):
			t.Errorf("ParseInt(%v, %#v, %v): fail; got %v, want %v",
				typ, tt.val, tt.mode, elem(res), tt.exp)
		default:
			t.Logf("ParseInt(%v, %#v, %s): pass; got %v, error %v",
				typ, tt.val, tt.mode, elem(res), err)
		}
	}
}
