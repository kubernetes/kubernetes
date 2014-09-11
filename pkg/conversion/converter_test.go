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
		Baz int
	}
	type B struct {
		Bar string
		Baz int
	}
	type C struct{}
	c := NewConverter()
	c.Debug = t
	err := c.Register(func(in *A, out *B, s Scope) error {
		out.Bar = in.Foo
		return s.Convert(&in.Baz, &out.Baz, 0)
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	err = c.Register(func(in *B, out *A, s Scope) error {
		out.Foo = in.Bar
		return s.Convert(&in.Baz, &out.Baz, 0)
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	x := A{"hello, intrepid test reader!", 3}
	y := B{}

	err = c.Convert(&x, &y, 0, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := x.Foo, y.Bar; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := x.Baz, y.Baz; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	z := B{"all your test are belong to us", 42}
	w := A{}

	err = c.Convert(&z, &w, 0, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := z.Bar, w.Foo; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
	if e, a := z.Baz, w.Baz; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	err = c.Register(func(in *A, out *C, s Scope) error {
		return fmt.Errorf("C can't store an A, silly")
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = c.Convert(&A{}, &C{}, 0, nil)
	if err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestConverter_fuzz(t *testing.T) {
	// Use the same types from the scheme test.
	table := []struct {
		from, to, check interface{}
	}{
		{&TestType1{}, &ExternalTestType1{}, &TestType1{}},
		{&ExternalTestType1{}, &TestType1{}, &ExternalTestType1{}},
	}

	f := fuzz.New().NilChance(.5).NumElements(0, 100)
	c := NewConverter()
	c.NameFunc = func(t reflect.Type) string {
		// Hide the fact that we don't have separate packages for these things.
		return map[reflect.Type]string{
			reflect.TypeOf(TestType1{}):         "TestType1",
			reflect.TypeOf(ExternalTestType1{}): "TestType1",
			reflect.TypeOf(TestType2{}):         "TestType2",
			reflect.TypeOf(ExternalTestType2{}): "TestType2",
		}[t]
	}
	c.Debug = t

	for i, item := range table {
		for j := 0; j < *fuzzIters; j++ {
			f.Fuzz(item.from)
			err := c.Convert(item.from, item.to, 0, nil)
			if err != nil {
				t.Errorf("(%v, %v): unexpected error: %v", i, j, err)
				continue
			}
			err = c.Convert(item.to, item.check, 0, nil)
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

func TestConverter_tags(t *testing.T) {
	type Foo struct {
		A string `test:"foo"`
	}
	type Bar struct {
		A string `test:"bar"`
	}
	c := NewConverter()
	c.Debug = t
	err := c.Register(
		func(in *string, out *string, s Scope) error {
			if e, a := "foo", s.SrcTag().Get("test"); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
			if e, a := "bar", s.DestTag().Get("test"); e != a {
				t.Errorf("expected %v, got %v", e, a)
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	c.Convert(&Foo{}, &Bar{}, 0, nil)
}

func TestConverter_meta(t *testing.T) {
	type Foo struct{ A string }
	type Bar struct{ A string }
	c := NewConverter()
	c.Debug = t
	checks := 0
	err := c.Register(
		func(in *Foo, out *Bar, s Scope) error {
			if s.Meta() == nil || s.Meta().SrcVersion != "test" || s.Meta().DestVersion != "passes" {
				t.Errorf("Meta did not get passed!")
			}
			checks++
			s.Convert(&in.A, &out.A, 0)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	err = c.Register(
		func(in *string, out *string, s Scope) error {
			if s.Meta() == nil || s.Meta().SrcVersion != "test" || s.Meta().DestVersion != "passes" {
				t.Errorf("Meta did not get passed a second time!")
			}
			checks++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	err = c.Convert(&Foo{}, &Bar{}, 0, &Meta{SrcVersion: "test", DestVersion: "passes"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if checks != 2 {
		t.Errorf("Registered functions did not get called.")
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
	c.Debug = t

	for i, item := range table {
		for j := 0; j < *fuzzIters; j++ {
			f.Fuzz(item.from)
			err := c.Convert(item.from, item.to, item.flags, nil)
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
