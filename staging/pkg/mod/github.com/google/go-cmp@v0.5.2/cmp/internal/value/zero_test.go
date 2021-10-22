// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package value

import (
	"archive/tar"
	"math"
	"reflect"
	"testing"
)

func TestIsZero(t *testing.T) {
	tests := []struct {
		in   interface{}
		want bool
	}{
		{0, true},
		{1, false},
		{"", true},
		{"foo", false},
		{[]byte(nil), true},
		{[]byte{}, false},
		{map[string]bool(nil), true},
		{map[string]bool{}, false},
		{tar.Header{}, true},
		{&tar.Header{}, false},
		{tar.Header{Name: "foo"}, false},
		{(chan bool)(nil), true},
		{make(chan bool), false},
		{(func(*testing.T))(nil), true},
		{TestIsZero, false},
		{[...]int{0, 0, 0}, true},
		{[...]int{0, 1, 0}, false},
		{math.Copysign(0, +1), true},
		{math.Copysign(0, -1), false},
		{complex(math.Copysign(0, +1), math.Copysign(0, +1)), true},
		{complex(math.Copysign(0, -1), math.Copysign(0, +1)), false},
		{complex(math.Copysign(0, +1), math.Copysign(0, -1)), false},
		{complex(math.Copysign(0, -1), math.Copysign(0, -1)), false},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := IsZero(reflect.ValueOf(tt.in))
			if got != tt.want {
				t.Errorf("IsZero(%v) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
