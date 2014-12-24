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
	"strconv"
	"testing"

	"github.com/google/gofuzz"
)

func TestConverter_DefaultConvert(t *testing.T) {
	type A struct {
		Foo string
		Baz int
	}
	type B struct {
		Bar string
		Baz int
	}
	c := NewConverter()
	c.Debug = t
	c.NameFunc = func(t reflect.Type) string { return "MyType" }

	// Ensure conversion funcs can call DefaultConvert to get default behavior,
	// then fixup remaining fields manually
	err := c.Register(func(in *A, out *B, s Scope) error {
		if err := s.DefaultConvert(in, out, IgnoreMissingFields); err != nil {
			return err
		}
		out.Bar = in.Foo
		return nil
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
}

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

func TestConverter_MapElemAddr(t *testing.T) {
	type Foo struct {
		A map[int]int
	}
	type Bar struct {
		A map[string]string
	}
	c := NewConverter()
	c.Debug = t
	err := c.Register(
		func(in *int, out *string, s Scope) error {
			*out = fmt.Sprintf("%v", *in)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	err = c.Register(
		func(in *string, out *int, s Scope) error {
			if str, err := strconv.Atoi(*in); err != nil {
				return err
			} else {
				*out = str
				return nil
			}
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	f := fuzz.New().NilChance(0).NumElements(3, 3)
	first := Foo{}
	second := Bar{}
	f.Fuzz(&first)
	err = c.Convert(&first, &second, AllowDifferentFieldTypeNames, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	third := Foo{}
	err = c.Convert(&second, &third, AllowDifferentFieldTypeNames, nil)
	if e, a := first, third; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected diff: %v", objDiff(e, a))
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
	err = c.Convert(&Foo{}, &Bar{}, AllowDifferentFieldTypeNames, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
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

func TestConverter_FieldRename(t *testing.T) {
	type WeirdMeta struct {
		Name string
		Type string
	}
	type NameMeta struct {
		Name string
	}
	type TypeMeta struct {
		Type string
	}
	type A struct {
		WeirdMeta
	}
	type B struct {
		TypeMeta
		NameMeta
	}

	c := NewConverter()
	err := c.SetStructFieldCopy(WeirdMeta{}, "WeirdMeta", TypeMeta{}, "TypeMeta")
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	err = c.SetStructFieldCopy(WeirdMeta{}, "WeirdMeta", NameMeta{}, "NameMeta")
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	err = c.SetStructFieldCopy(TypeMeta{}, "TypeMeta", WeirdMeta{}, "WeirdMeta")
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	err = c.SetStructFieldCopy(NameMeta{}, "NameMeta", WeirdMeta{}, "WeirdMeta")
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	c.Debug = t

	aVal := &A{
		WeirdMeta: WeirdMeta{
			Name: "Foo",
			Type: "Bar",
		},
	}

	bVal := &B{
		TypeMeta: TypeMeta{"Bar"},
		NameMeta: NameMeta{"Foo"},
	}

	table := map[string]struct {
		from, to, expect interface{}
		flags            FieldMatchingFlags
	}{
		"to": {
			aVal,
			&B{},
			bVal,
			AllowDifferentFieldTypeNames | SourceToDest | IgnoreMissingFields,
		},
		"from": {
			bVal,
			&A{},
			aVal,
			AllowDifferentFieldTypeNames | SourceToDest,
		},
		"toDestFirst": {
			aVal,
			&B{},
			bVal,
			AllowDifferentFieldTypeNames,
		},
		"fromDestFirst": {
			bVal,
			&A{},
			aVal,
			AllowDifferentFieldTypeNames | IgnoreMissingFields,
		},
	}

	for name, item := range table {
		err := c.Convert(item.from, item.to, item.flags, nil)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", name, err)
			continue
		}
		if e, a := item.expect, item.to; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected diff: %v", name, objDiff(e, a))
		}
	}
}
