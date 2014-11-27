package mapstructure

import (
	"errors"
	"reflect"
	"testing"
)

func TestComposeDecodeHookFunc(t *testing.T) {
	f1 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		return data.(string) + "foo", nil
	}

	f2 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		return data.(string) + "bar", nil
	}

	f := ComposeDecodeHookFunc(f1, f2)

	result, err := f(reflect.String, reflect.Slice, "")
	if err != nil {
		t.Fatalf("bad: %s", err)
	}
	if result.(string) != "foobar" {
		t.Fatalf("bad: %#v", result)
	}
}

func TestComposeDecodeHookFunc_err(t *testing.T) {
	f1 := func(reflect.Kind, reflect.Kind, interface{}) (interface{}, error) {
		return nil, errors.New("foo")
	}

	f2 := func(reflect.Kind, reflect.Kind, interface{}) (interface{}, error) {
		panic("NOPE")
	}

	f := ComposeDecodeHookFunc(f1, f2)

	_, err := f(reflect.String, reflect.Slice, 42)
	if err.Error() != "foo" {
		t.Fatalf("bad: %s", err)
	}
}

func TestComposeDecodeHookFunc_kinds(t *testing.T) {
	var f2From reflect.Kind

	f1 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		return int(42), nil
	}

	f2 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		f2From = f
		return data, nil
	}

	f := ComposeDecodeHookFunc(f1, f2)

	_, err := f(reflect.String, reflect.Slice, "")
	if err != nil {
		t.Fatalf("bad: %s", err)
	}
	if f2From != reflect.Int {
		t.Fatalf("bad: %#v", f2From)
	}
}

func TestStringToSliceHookFunc(t *testing.T) {
	f := StringToSliceHookFunc(",")

	cases := []struct {
		f, t   reflect.Kind
		data   interface{}
		result interface{}
		err    bool
	}{
		{reflect.Slice, reflect.Slice, 42, 42, false},
		{reflect.String, reflect.String, 42, 42, false},
		{
			reflect.String,
			reflect.Slice,
			"foo,bar,baz",
			[]string{"foo", "bar", "baz"},
			false,
		},
		{
			reflect.String,
			reflect.Slice,
			"",
			[]string{},
			false,
		},
	}

	for i, tc := range cases {
		actual, err := f(tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}

func TestWeaklyTypedHook(t *testing.T) {
	var f DecodeHookFunc = WeaklyTypedHook

	cases := []struct {
		f, t   reflect.Kind
		data   interface{}
		result interface{}
		err    bool
	}{
		// TO STRING
		{
			reflect.Bool,
			reflect.String,
			false,
			"0",
			false,
		},

		{
			reflect.Bool,
			reflect.String,
			true,
			"1",
			false,
		},

		{
			reflect.Float32,
			reflect.String,
			float32(7),
			"7",
			false,
		},

		{
			reflect.Int,
			reflect.String,
			int(7),
			"7",
			false,
		},

		{
			reflect.Slice,
			reflect.String,
			[]uint8("foo"),
			"foo",
			false,
		},

		{
			reflect.Uint,
			reflect.String,
			uint(7),
			"7",
			false,
		},
	}

	for i, tc := range cases {
		actual, err := f(tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}
