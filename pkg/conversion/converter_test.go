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

package conversion

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/google/gofuzz"
)

func TestConverter_CallsRegisteredFunctions(t *testing.T) {
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

	err = c.Convert(&x, &y, 0)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := x.Foo, y.Bar; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	z := B{"all your test are belong to us"}
	w := A{}

	err = c.Convert(&z, &w, 0)
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

	err = c.Convert(&A{}, &C{}, 0)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestConverter_fuzz(t *testing.T) {
	newAnonType := func() interface{} {
		return reflect.New(reflect.TypeOf(externalTypeReturn())).Interface()
	}
	// Use the same types from the scheme test.
	table := []struct {
		from, to, check interface{}
	}{
		{&TestType1{}, newAnonType(), &TestType1{}},
		{newAnonType(), &TestType1{}, newAnonType()},
	}

	f := fuzz.New().NilChance(.5).NumElements(0, 100)
	c := NewConverter()

	for i, item := range table {
		for j := 0; j < *fuzzIters; j++ {
			f.Fuzz(item.from)
			err := c.Convert(item.from, item.to, 0)
			if err != nil {
				t.Errorf("(%v, %v): unexpected error: %v", i, j, err)
				continue
			}
			err = c.Convert(item.to, item.check, 0)
			if err != nil {
				t.Errorf("(%v, %v): unexpected error: %v", i, j, err)
				continue
			}
			if e, a := item.from, item.check; !reflect.DeepEqual(e, a) {
				t.Errorf("(%v, %v): unexpected diff: %v", i, j, objDiff(e, a))
			}
		}
	}
}

func TestConverter_flags(t *testing.T) {
	type Foo struct{ A string }
	type Bar struct{ A string }
	table := []struct {
		from, to      interface{}
		flags         FieldMatchingFlags
		shouldSucceed bool
	}{
		// Check that DestFromSource allows extra fields only in source.
		{
			from:          &struct{ A string }{},
			to:            &struct{ A, B string }{},
			flags:         DestFromSource,
			shouldSucceed: false,
		}, {
			from:          &struct{ A, B string }{},
			to:            &struct{ A string }{},
			flags:         DestFromSource,
			shouldSucceed: true,
		},

		// Check that SourceToDest allows for extra fields only in dest.
		{
			from:          &struct{ A string }{},
			to:            &struct{ A, B string }{},
			flags:         SourceToDest,
			shouldSucceed: true,
		}, {
			from:          &struct{ A, B string }{},
			to:            &struct{ A string }{},
			flags:         SourceToDest,
			shouldSucceed: false,
		},

		// Check that IgnoreMissingFields makes the above failure cases pass.
		{
			from:          &struct{ A string }{},
			to:            &struct{ A, B string }{},
			flags:         DestFromSource | IgnoreMissingFields,
			shouldSucceed: true,
		}, {
			from:          &struct{ A, B string }{},
			to:            &struct{ A string }{},
			flags:         SourceToDest | IgnoreMissingFields,
			shouldSucceed: true,
		},

		// Check that the field type name must match unless
		// AllowDifferentFieldTypeNames is specified.
		{
			from:          &struct{ A, B Foo }{},
			to:            &struct{ A Bar }{},
			flags:         DestFromSource,
			shouldSucceed: false,
		}, {
			from:          &struct{ A Foo }{},
			to:            &struct{ A, B Bar }{},
			flags:         SourceToDest,
			shouldSucceed: false,
		}, {
			from:          &struct{ A, B Foo }{},
			to:            &struct{ A Bar }{},
			flags:         DestFromSource | AllowDifferentFieldTypeNames,
			shouldSucceed: true,
		}, {
			from:          &struct{ A Foo }{},
			to:            &struct{ A, B Bar }{},
			flags:         SourceToDest | AllowDifferentFieldTypeNames,
			shouldSucceed: true,
		},
	}
	f := fuzz.New().NilChance(.5).NumElements(0, 100)
	c := NewConverter()

	for i, item := range table {
		for j := 0; j < *fuzzIters; j++ {
			f.Fuzz(item.from)
			err := c.Convert(item.from, item.to, item.flags)
			if item.shouldSucceed && err != nil {
				t.Errorf("(%v, %v): unexpected error: %v", i, j, err)
				continue
			}
			if !item.shouldSucceed && err == nil {
				t.Errorf("(%v, %v): unexpected non-error", i, j)
				continue
			}
		}
	}
}
