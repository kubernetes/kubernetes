// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package function

import (
	"bytes"
	"reflect"
	"testing"
)

type myType struct{ bytes.Buffer }

func (myType) valueMethod() {}
func (myType) ValueMethod() {}

func (*myType) pointerMethod() {}
func (*myType) PointerMethod() {}

func TestNameOf(t *testing.T) {
	tests := []struct {
		fnc  interface{}
		want string
	}{
		{TestNameOf, "function.TestNameOf"},
		{func() {}, "function.TestNameOf.func1"},
		{(myType).valueMethod, "function.myType.valueMethod"},
		{(myType).ValueMethod, "function.myType.ValueMethod"},
		{(myType{}).valueMethod, "function.myType.valueMethod"},
		{(myType{}).ValueMethod, "function.myType.ValueMethod"},
		{(*myType).valueMethod, "function.myType.valueMethod"},
		{(*myType).ValueMethod, "function.myType.ValueMethod"},
		{(&myType{}).valueMethod, "function.myType.valueMethod"},
		{(&myType{}).ValueMethod, "function.myType.ValueMethod"},
		{(*myType).pointerMethod, "function.myType.pointerMethod"},
		{(*myType).PointerMethod, "function.myType.PointerMethod"},
		{(&myType{}).pointerMethod, "function.myType.pointerMethod"},
		{(&myType{}).PointerMethod, "function.myType.PointerMethod"},
		{(*myType).Write, "function.myType.Write"},
		{(&myType{}).Write, "bytes.Buffer.Write"},
	}
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := NameOf(reflect.ValueOf(tt.fnc))
			if got != tt.want {
				t.Errorf("NameOf() = %v, want %v", got, tt.want)
			}
		})
	}
}
