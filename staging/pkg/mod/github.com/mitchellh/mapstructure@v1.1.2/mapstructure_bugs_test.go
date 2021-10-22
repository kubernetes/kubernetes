package mapstructure

import (
	"reflect"
	"testing"
)

// GH-1, GH-10, GH-96
func TestDecode_NilValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		in         interface{}
		target     interface{}
		out        interface{}
		metaKeys   []string
		metaUnused []string
	}{
		{
			"all nil",
			&map[string]interface{}{
				"vfoo":   nil,
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "", Vother: nil},
			[]string{"Vfoo", "Vother"},
			[]string{},
		},
		{
			"partial nil",
			&map[string]interface{}{
				"vfoo":   "baz",
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "baz", Vother: nil},
			[]string{"Vfoo", "Vother"},
			[]string{},
		},
		{
			"partial decode",
			&map[string]interface{}{
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "foo", Vother: nil},
			[]string{"Vother"},
			[]string{},
		},
		{
			"unused values",
			&map[string]interface{}{
				"vbar":   "bar",
				"vfoo":   nil,
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "", Vother: nil},
			[]string{"Vfoo", "Vother"},
			[]string{"vbar"},
		},
		{
			"map interface all nil",
			&map[interface{}]interface{}{
				"vfoo":   nil,
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "", Vother: nil},
			[]string{"Vfoo", "Vother"},
			[]string{},
		},
		{
			"map interface partial nil",
			&map[interface{}]interface{}{
				"vfoo":   "baz",
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "baz", Vother: nil},
			[]string{"Vfoo", "Vother"},
			[]string{},
		},
		{
			"map interface partial decode",
			&map[interface{}]interface{}{
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "foo", Vother: nil},
			[]string{"Vother"},
			[]string{},
		},
		{
			"map interface unused values",
			&map[interface{}]interface{}{
				"vbar":   "bar",
				"vfoo":   nil,
				"vother": nil,
			},
			&Map{Vfoo: "foo", Vother: map[string]string{"foo": "bar"}},
			&Map{Vfoo: "", Vother: nil},
			[]string{"Vfoo", "Vother"},
			[]string{"vbar"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := &DecoderConfig{
				Metadata:   new(Metadata),
				Result:     tc.target,
				ZeroFields: true,
			}

			decoder, err := NewDecoder(config)
			if err != nil {
				t.Fatalf("should not error: %s", err)
			}

			err = decoder.Decode(tc.in)
			if err != nil {
				t.Fatalf("should not error: %s", err)
			}

			if !reflect.DeepEqual(tc.out, tc.target) {
				t.Fatalf("%q: TestDecode_NilValue() expected: %#v, got: %#v", tc.name, tc.out, tc.target)
			}

			if !reflect.DeepEqual(tc.metaKeys, config.Metadata.Keys) {
				t.Fatalf("%q: Metadata.Keys mismatch expected: %#v, got: %#v", tc.name, tc.metaKeys, config.Metadata.Keys)
			}

			if !reflect.DeepEqual(tc.metaUnused, config.Metadata.Unused) {
				t.Fatalf("%q: Metadata.Unused mismatch expected: %#v, got: %#v", tc.name, tc.metaUnused, config.Metadata.Unused)
			}
		})
	}
}

// #48
func TestNestedTypePointerWithDefaults(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": map[string]interface{}{
			"vstring": "foo",
			"vint":    42,
			"vbool":   true,
		},
	}

	result := NestedPointer{
		Vbar: &Basic{
			Vuint: 42,
		},
	}
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Vfoo != "foo" {
		t.Errorf("vfoo value should be 'foo': %#v", result.Vfoo)
	}

	if result.Vbar.Vstring != "foo" {
		t.Errorf("vstring value should be 'foo': %#v", result.Vbar.Vstring)
	}

	if result.Vbar.Vint != 42 {
		t.Errorf("vint value should be 42: %#v", result.Vbar.Vint)
	}

	if result.Vbar.Vbool != true {
		t.Errorf("vbool value should be true: %#v", result.Vbar.Vbool)
	}

	if result.Vbar.Vextra != "" {
		t.Errorf("vextra value should be empty: %#v", result.Vbar.Vextra)
	}

	// this is the error
	if result.Vbar.Vuint != 42 {
		t.Errorf("vuint value should be 42: %#v", result.Vbar.Vuint)
	}

}

type NestedSlice struct {
	Vfoo   string
	Vbars  []Basic
	Vempty []Basic
}

// #48
func TestNestedTypeSliceWithDefaults(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbars": []map[string]interface{}{
			{"vstring": "foo", "vint": 42, "vbool": true},
			{"vint": 42, "vbool": true},
		},
		"vempty": []map[string]interface{}{
			{"vstring": "foo", "vint": 42, "vbool": true},
			{"vint": 42, "vbool": true},
		},
	}

	result := NestedSlice{
		Vbars: []Basic{
			{Vuint: 42},
			{Vstring: "foo"},
		},
	}
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Vfoo != "foo" {
		t.Errorf("vfoo value should be 'foo': %#v", result.Vfoo)
	}

	if result.Vbars[0].Vstring != "foo" {
		t.Errorf("vstring value should be 'foo': %#v", result.Vbars[0].Vstring)
	}
	// this is the error
	if result.Vbars[0].Vuint != 42 {
		t.Errorf("vuint value should be 42: %#v", result.Vbars[0].Vuint)
	}
}

// #48 workaround
func TestNestedTypeWithDefaults(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": map[string]interface{}{
			"vstring": "foo",
			"vint":    42,
			"vbool":   true,
		},
	}

	result := Nested{
		Vbar: Basic{
			Vuint: 42,
		},
	}
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Vfoo != "foo" {
		t.Errorf("vfoo value should be 'foo': %#v", result.Vfoo)
	}

	if result.Vbar.Vstring != "foo" {
		t.Errorf("vstring value should be 'foo': %#v", result.Vbar.Vstring)
	}

	if result.Vbar.Vint != 42 {
		t.Errorf("vint value should be 42: %#v", result.Vbar.Vint)
	}

	if result.Vbar.Vbool != true {
		t.Errorf("vbool value should be true: %#v", result.Vbar.Vbool)
	}

	if result.Vbar.Vextra != "" {
		t.Errorf("vextra value should be empty: %#v", result.Vbar.Vextra)
	}

	// this is the error
	if result.Vbar.Vuint != 42 {
		t.Errorf("vuint value should be 42: %#v", result.Vbar.Vuint)
	}

}

// #67 panic() on extending slices (decodeSlice with disabled ZeroValues)
func TestDecodeSliceToEmptySliceWOZeroing(t *testing.T) {
	t.Parallel()

	type TestStruct struct {
		Vfoo []string
	}

	decode := func(m interface{}, rawVal interface{}) error {
		config := &DecoderConfig{
			Metadata:   nil,
			Result:     rawVal,
			ZeroFields: false,
		}

		decoder, err := NewDecoder(config)
		if err != nil {
			return err
		}

		return decoder.Decode(m)
	}

	{
		input := map[string]interface{}{
			"vfoo": []string{"1"},
		}

		result := &TestStruct{}

		err := decode(input, &result)
		if err != nil {
			t.Fatalf("got an err: %s", err.Error())
		}
	}

	{
		input := map[string]interface{}{
			"vfoo": []string{"1"},
		}

		result := &TestStruct{
			Vfoo: []string{},
		}

		err := decode(input, &result)
		if err != nil {
			t.Fatalf("got an err: %s", err.Error())
		}
	}

	{
		input := map[string]interface{}{
			"vfoo": []string{"2", "3"},
		}

		result := &TestStruct{
			Vfoo: []string{"1"},
		}

		err := decode(input, &result)
		if err != nil {
			t.Fatalf("got an err: %s", err.Error())
		}
	}
}

// #70
func TestNextSquashMapstructure(t *testing.T) {
	data := &struct {
		Level1 struct {
			Level2 struct {
				Foo string
			} `mapstructure:",squash"`
		} `mapstructure:",squash"`
	}{}
	err := Decode(map[interface{}]interface{}{"foo": "baz"}, &data)
	if err != nil {
		t.Fatalf("should not error: %s", err)
	}
	if data.Level1.Level2.Foo != "baz" {
		t.Fatal("value should be baz")
	}
}

type ImplementsInterfacePointerReceiver struct {
	Name string
}

func (i *ImplementsInterfacePointerReceiver) DoStuff() {}

type ImplementsInterfaceValueReceiver string

func (i ImplementsInterfaceValueReceiver) DoStuff() {}

// GH-140 Type error when using DecodeHook to decode into interface
func TestDecode_DecodeHookInterface(t *testing.T) {
	t.Parallel()

	type Interface interface {
		DoStuff()
	}
	type DecodeIntoInterface struct {
		Test Interface
	}

	testData := map[string]string{"test": "test"}

	stringToPointerInterfaceDecodeHook := func(from, to reflect.Type, data interface{}) (interface{}, error) {
		if from.Kind() != reflect.String {
			return data, nil
		}

		if to != reflect.TypeOf((*Interface)(nil)).Elem() {
			return data, nil
		}
		// Ensure interface is satisfied
		var impl Interface = &ImplementsInterfacePointerReceiver{data.(string)}
		return impl, nil
	}

	stringToValueInterfaceDecodeHook := func(from, to reflect.Type, data interface{}) (interface{}, error) {
		if from.Kind() != reflect.String {
			return data, nil
		}

		if to != reflect.TypeOf((*Interface)(nil)).Elem() {
			return data, nil
		}
		// Ensure interface is satisfied
		var impl Interface = ImplementsInterfaceValueReceiver(data.(string))
		return impl, nil
	}

	{
		decodeInto := new(DecodeIntoInterface)

		decoder, _ := NewDecoder(&DecoderConfig{
			DecodeHook: stringToPointerInterfaceDecodeHook,
			Result:     decodeInto,
		})

		err := decoder.Decode(testData)
		if err != nil {
			t.Fatalf("Decode returned error: %s", err)
		}

		expected := &ImplementsInterfacePointerReceiver{"test"}
		if !reflect.DeepEqual(decodeInto.Test, expected) {
			t.Fatalf("expected: %#v (%T), got: %#v (%T)", decodeInto.Test, decodeInto.Test, expected, expected)
		}
	}

	{
		decodeInto := new(DecodeIntoInterface)

		decoder, _ := NewDecoder(&DecoderConfig{
			DecodeHook: stringToValueInterfaceDecodeHook,
			Result:     decodeInto,
		})

		err := decoder.Decode(testData)
		if err != nil {
			t.Fatalf("Decode returned error: %s", err)
		}

		expected := ImplementsInterfaceValueReceiver("test")
		if !reflect.DeepEqual(decodeInto.Test, expected) {
			t.Fatalf("expected: %#v (%T), got: %#v (%T)", decodeInto.Test, decodeInto.Test, expected, expected)
		}
	}
}
