package mapstructure

import "testing"

// GH-1
func TestDecode_NilValue(t *testing.T) {
	input := map[string]interface{}{
		"vfoo":   nil,
		"vother": nil,
	}

	var result Map
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("should not error: %s", err)
	}

	if result.Vfoo != "" {
		t.Fatalf("value should be default: %s", result.Vfoo)
	}

	if result.Vother != nil {
		t.Fatalf("Vother should be nil: %s", result.Vother)
	}
}

// GH-10
func TestDecode_mapInterfaceInterface(t *testing.T) {
	input := map[interface{}]interface{}{
		"vfoo":   nil,
		"vother": nil,
	}

	var result Map
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("should not error: %s", err)
	}

	if result.Vfoo != "" {
		t.Fatalf("value should be default: %s", result.Vfoo)
	}

	if result.Vother != nil {
		t.Fatalf("Vother should be nil: %s", result.Vother)
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
