package types

import (
	"reflect"
	"testing"
)

func TestScanFully(t *testing.T) {
	for _, tt := range []struct {
		val  string
		verb byte
		res  interface{}
		ok   bool
	}{
		{"a", 'v', int(0), false},
		{"0x", 'v', int(0), true},
		{"0x", 'd', int(0), false},
	} {
		d := reflect.New(reflect.TypeOf(tt.res)).Interface()
		err := ScanFully(d, tt.val, tt.verb)
		switch {
		case tt.ok && err != nil:
			t.Errorf("ScanFully(%T, %q, '%c'): want ok, got error %v",
				d, tt.val, tt.verb, err)
		case !tt.ok && err == nil:
			t.Errorf("ScanFully(%T, %q, '%c'): want error, got %v",
				d, tt.val, tt.verb, elem(d))
		case tt.ok && err == nil && !reflect.DeepEqual(tt.res, elem(d)):
			t.Errorf("ScanFully(%T, %q, '%c'): want %v, got %v",
				d, tt.val, tt.verb, tt.res, elem(d))
		default:
			t.Logf("ScanFully(%T, %q, '%c') = %v; *ptr==%v",
				d, tt.val, tt.verb, err, elem(d))
		}
	}
}
