/*
Copyright 2014 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/google/gofuzz"
	flag "github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/diff"
)

var fuzzIters = flag.Int("fuzz-iters", 50, "How many fuzzing iterations to do.")

// Test a weird version/kind embedding format.
type MyWeirdCustomEmbeddedVersionKindField struct {
	ID         string `json:"ID,omitempty"`
	APIVersion string `json:"myVersionKey,omitempty"`
	ObjectKind string `json:"myKindKey,omitempty"`
	Z          string `json:"Z,omitempty"`
	Y          uint64 `json:"Y,omitempty"`
}

type TestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string               `json:"A,omitempty"`
	B                                     int                  `json:"B,omitempty"`
	C                                     int8                 `json:"C,omitempty"`
	D                                     int16                `json:"D,omitempty"`
	E                                     int32                `json:"E,omitempty"`
	F                                     int64                `json:"F,omitempty"`
	G                                     uint                 `json:"G,omitempty"`
	H                                     uint8                `json:"H,omitempty"`
	I                                     uint16               `json:"I,omitempty"`
	J                                     uint32               `json:"J,omitempty"`
	K                                     uint64               `json:"K,omitempty"`
	L                                     bool                 `json:"L,omitempty"`
	M                                     map[string]int       `json:"M,omitempty"`
	N                                     map[string]TestType2 `json:"N,omitempty"`
	O                                     *TestType2           `json:"O,omitempty"`
	P                                     []TestType2          `json:"Q,omitempty"`
}

type TestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}

type ExternalTestType2 struct {
	A string `json:"A,omitempty"`
	B int    `json:"B,omitempty"`
}
type ExternalTestType1 struct {
	MyWeirdCustomEmbeddedVersionKindField `json:",inline"`
	A                                     string                       `json:"A,omitempty"`
	B                                     int                          `json:"B,omitempty"`
	C                                     int8                         `json:"C,omitempty"`
	D                                     int16                        `json:"D,omitempty"`
	E                                     int32                        `json:"E,omitempty"`
	F                                     int64                        `json:"F,omitempty"`
	G                                     uint                         `json:"G,omitempty"`
	H                                     uint8                        `json:"H,omitempty"`
	I                                     uint16                       `json:"I,omitempty"`
	J                                     uint32                       `json:"J,omitempty"`
	K                                     uint64                       `json:"K,omitempty"`
	L                                     bool                         `json:"L,omitempty"`
	M                                     map[string]int               `json:"M,omitempty"`
	N                                     map[string]ExternalTestType2 `json:"N,omitempty"`
	O                                     *ExternalTestType2           `json:"O,omitempty"`
	P                                     []ExternalTestType2          `json:"Q,omitempty"`
}

func testLogger(t *testing.T) DebugLogger {
	// We don't set logger to eliminate rubbish logs in tests.
	// If you want to switch it, simply switch it to: "return t"
	return nil
}

func TestConverter_byteSlice(t *testing.T) {
	c := NewConverter(DefaultNameFunc)
	src := []byte{1, 2, 3}
	dest := []byte{}
	err := c.Convert(&src, &dest, 0, nil)
	if err != nil {
		t.Fatalf("expected no error")
	}
	if e, a := src, dest; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %#v, got %#v", e, a)
	}
}

func TestConverter_MismatchedTypes(t *testing.T) {
	c := NewConverter(DefaultNameFunc)

	err := c.RegisterConversionFunc(
		func(in *[]string, out *int, s Scope) error {
			if str, err := strconv.Atoi((*in)[0]); err != nil {
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

	src := []string{"5"}
	var dest *int
	err = c.Convert(&src, &dest, 0, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := 5, *dest; e != a {
		t.Errorf("expected %#v, got %#v", e, a)
	}
}

func TestConverter_DefaultConvert(t *testing.T) {
	type A struct {
		Foo string
		Baz int
	}
	type B struct {
		Bar string
		Baz int
	}
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	c.nameFunc = func(t reflect.Type) string { return "MyType" }

	// Ensure conversion funcs can call DefaultConvert to get default behavior,
	// then fixup remaining fields manually
	err := c.RegisterConversionFunc(func(in *A, out *B, s Scope) error {
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

func TestConverter_DeepCopy(t *testing.T) {
	type A struct {
		Foo *string
		Bar []string
		Baz interface{}
		Qux map[string]string
	}
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)

	foo, baz := "foo", "baz"
	x := A{
		Foo: &foo,
		Bar: []string{"bar"},
		Baz: &baz,
		Qux: map[string]string{"qux": "qux"},
	}
	y := A{}

	if err := c.Convert(&x, &y, 0, nil); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	*x.Foo = "foo2"
	x.Bar[0] = "bar2"
	*x.Baz.(*string) = "baz2"
	x.Qux["qux"] = "qux2"
	if e, a := *x.Foo, *y.Foo; e == a {
		t.Errorf("expected difference between %v and %v", e, a)
	}
	if e, a := x.Bar, y.Bar; reflect.DeepEqual(e, a) {
		t.Errorf("expected difference between %v and %v", e, a)
	}
	if e, a := *x.Baz.(*string), *y.Baz.(*string); e == a {
		t.Errorf("expected difference between %v and %v", e, a)
	}
	if e, a := x.Qux, y.Qux; reflect.DeepEqual(e, a) {
		t.Errorf("expected difference between %v and %v", e, a)
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
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	err := c.RegisterConversionFunc(func(in *A, out *B, s Scope) error {
		out.Bar = in.Foo
		return s.Convert(&in.Baz, &out.Baz, 0)
	})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	err = c.RegisterConversionFunc(func(in *B, out *A, s Scope) error {
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

	err = c.RegisterConversionFunc(func(in *A, out *C, s Scope) error {
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

func TestConverter_IgnoredConversion(t *testing.T) {
	type A struct{}
	type B struct{}

	count := 0
	c := NewConverter(DefaultNameFunc)
	if err := c.RegisterConversionFunc(func(in *A, out *B, s Scope) error {
		count++
		return nil
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if err := c.RegisterIgnoredConversion(&A{}, &B{}); err != nil {
		t.Fatal(err)
	}
	a := A{}
	b := B{}
	if err := c.Convert(&a, &b, 0, nil); err != nil {
		t.Errorf("%v", err)
	}
	if count != 0 {
		t.Errorf("unexpected number of conversion invocations")
	}
}

func TestConverter_IgnoredConversionNested(t *testing.T) {
	type C string
	type A struct {
		C C
	}
	type B struct {
		C C
	}

	c := NewConverter(DefaultNameFunc)
	typed := C("")
	if err := c.RegisterIgnoredConversion(&typed, &typed); err != nil {
		t.Fatal(err)
	}
	a := A{C: C("test")}
	b := B{C: C("other")}
	if err := c.Convert(&a, &b, AllowDifferentFieldTypeNames, nil); err != nil {
		t.Errorf("%v", err)
	}
	if b.C != C("other") {
		t.Errorf("expected no conversion of field C: %#v", b)
	}
}

func TestConverter_GeneratedConversionOverridden(t *testing.T) {
	type A struct{}
	type B struct{}
	c := NewConverter(DefaultNameFunc)
	if err := c.RegisterConversionFunc(func(in *A, out *B, s Scope) error {
		return nil
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if err := c.RegisterGeneratedConversionFunc(func(in *A, out *B, s Scope) error {
		return fmt.Errorf("generated function should be overridden")
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	a := A{}
	b := B{}
	if err := c.Convert(&a, &b, 0, nil); err != nil {
		t.Errorf("%v", err)
	}
}

func TestConverter_WithConversionOverridden(t *testing.T) {
	type A struct{}
	type B struct{}
	c := NewConverter(DefaultNameFunc)
	if err := c.RegisterConversionFunc(func(in *A, out *B, s Scope) error {
		return fmt.Errorf("conversion function should be overridden")
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if err := c.RegisterGeneratedConversionFunc(func(in *A, out *B, s Scope) error {
		return fmt.Errorf("generated function should be overridden")
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	ext := NewConversionFuncs()
	ext.Add(func(in *A, out *B, s Scope) error {
		return nil
	})
	newc := c.WithConversions(ext)

	a := A{}
	b := B{}
	if err := c.Convert(&a, &b, 0, nil); err == nil || err.Error() != "conversion function should be overridden" {
		t.Errorf("unexpected error: %v", err)
	}
	if err := newc.Convert(&a, &b, 0, nil); err != nil {
		t.Errorf("%v", err)
	}
}

func TestConverter_MapsStringArrays(t *testing.T) {
	type A struct {
		Foo   string
		Baz   int
		Other string
	}
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	if err := c.RegisterConversionFunc(func(input *[]string, out *string, s Scope) error {
		if len(*input) == 0 {
			*out = ""
		}
		*out = (*input)[0]
		return nil
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	x := map[string][]string{
		"Foo":   {"bar"},
		"Baz":   {"1"},
		"Other": {"", "test"},
		"other": {"wrong"},
	}
	y := A{"test", 2, "something"}

	if err := c.Convert(&x, &y, AllowDifferentFieldTypeNames, nil); err == nil {
		t.Error("unexpected non-error")
	}

	if err := c.RegisterConversionFunc(func(input *[]string, out *int, s Scope) error {
		if len(*input) == 0 {
			*out = 0
		}
		str := (*input)[0]
		i, err := strconv.Atoi(str)
		if err != nil {
			return err
		}
		*out = i
		return nil
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	if err := c.Convert(&x, &y, AllowDifferentFieldTypeNames, nil); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(y, A{"bar", 1, ""}) {
		t.Errorf("unexpected result: %#v", y)
	}
}

func TestConverter_MapsStringArraysWithMappingKey(t *testing.T) {
	type A struct {
		Foo   string `json:"test"`
		Baz   int
		Other string
	}
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	if err := c.RegisterConversionFunc(func(input *[]string, out *string, s Scope) error {
		if len(*input) == 0 {
			*out = ""
		}
		*out = (*input)[0]
		return nil
	}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	x := map[string][]string{
		"Foo":  {"bar"},
		"test": {"baz"},
	}
	y := A{"", 0, ""}

	if err := c.Convert(&x, &y, AllowDifferentFieldTypeNames|IgnoreMissingFields, &Meta{}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(y, A{"bar", 0, ""}) {
		t.Errorf("unexpected result: %#v", y)
	}

	mapping := func(key string, sourceTag, destTag reflect.StructTag) (source string, dest string) {
		if s := destTag.Get("json"); len(s) > 0 {
			return strings.SplitN(s, ",", 2)[0], key
		}
		return key, key
	}

	if err := c.Convert(&x, &y, AllowDifferentFieldTypeNames|IgnoreMissingFields, &Meta{KeyNameMapping: mapping}); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !reflect.DeepEqual(y, A{"baz", 0, ""}) {
		t.Errorf("unexpected result: %#v", y)
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
	c := NewConverter(DefaultNameFunc)
	c.nameFunc = func(t reflect.Type) string {
		// Hide the fact that we don't have separate packages for these things.
		return map[reflect.Type]string{
			reflect.TypeOf(TestType1{}):         "TestType1",
			reflect.TypeOf(ExternalTestType1{}): "TestType1",
			reflect.TypeOf(TestType2{}):         "TestType2",
			reflect.TypeOf(ExternalTestType2{}): "TestType2",
		}[t]
	}
	c.Debug = testLogger(t)

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
				t.Errorf("(%v, %v): unexpected diff: %v", i, j, diff.ObjectDiff(e, a))
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
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	err := c.RegisterConversionFunc(
		func(in *int, out *string, s Scope) error {
			*out = fmt.Sprintf("%v", *in)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	err = c.RegisterConversionFunc(
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
		t.Errorf("Unexpected diff: %v", diff.ObjectDiff(e, a))
	}
}

func TestConverter_tags(t *testing.T) {
	type Foo struct {
		A string `test:"foo"`
	}
	type Bar struct {
		A string `test:"bar"`
	}
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	err := c.RegisterConversionFunc(
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
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)
	checks := 0
	err := c.RegisterConversionFunc(
		func(in *Foo, out *Bar, s Scope) error {
			if s.Meta() == nil {
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
	err = c.RegisterConversionFunc(
		func(in *string, out *string, s Scope) error {
			if s.Meta() == nil {
				t.Errorf("Meta did not get passed a second time!")
			}
			checks++
			return nil
		},
	)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	err = c.Convert(&Foo{}, &Bar{}, 0, &Meta{})
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
	c := NewConverter(DefaultNameFunc)
	c.Debug = testLogger(t)

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
