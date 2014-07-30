/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package api

import (
	"fmt"
	"testing"
)

func TestConverter(t *testing.T) {
	type A struct {
		Foo string
	}
	type B struct {
		Bar string
	}
	type C struct{}
	c := NewConverter()
	err := c.Register(func(in *A, out *B) error {
		out.Bar = in.Foo
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	err = c.Register(func(in *B, out *A) error {
		out.Foo = in.Bar
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	x := A{"hello, intrepid test reader!"}
	y := B{}

	err = c.Convert(&x, &y)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := x.Foo, y.Bar; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	z := B{"all your test are belong to us"}
	w := A{}

	err = c.Convert(&z, &w)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := z.Bar, w.Foo; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	err = c.Register(func(in *A, out *C) error {
		return fmt.Errorf("C can't store an A, silly")
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = c.Convert(&A{}, &C{})
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}
