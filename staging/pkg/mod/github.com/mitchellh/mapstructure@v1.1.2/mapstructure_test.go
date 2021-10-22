package mapstructure

import (
	"encoding/json"
	"io"
	"reflect"
	"sort"
	"strings"
	"testing"
)

type Basic struct {
	Vstring     string
	Vint        int
	Vuint       uint
	Vbool       bool
	Vfloat      float64
	Vextra      string
	vsilent     bool
	Vdata       interface{}
	VjsonInt    int
	VjsonFloat  float64
	VjsonNumber json.Number
}

type BasicPointer struct {
	Vstring     *string
	Vint        *int
	Vuint       *uint
	Vbool       *bool
	Vfloat      *float64
	Vextra      *string
	vsilent     *bool
	Vdata       *interface{}
	VjsonInt    *int
	VjsonFloat  *float64
	VjsonNumber *json.Number
}

type BasicSquash struct {
	Test Basic `mapstructure:",squash"`
}

type Embedded struct {
	Basic
	Vunique string
}

type EmbeddedPointer struct {
	*Basic
	Vunique string
}

type EmbeddedSquash struct {
	Basic   `mapstructure:",squash"`
	Vunique string
}

type SliceAlias []string

type EmbeddedSlice struct {
	SliceAlias `mapstructure:"slice_alias"`
	Vunique    string
}

type ArrayAlias [2]string

type EmbeddedArray struct {
	ArrayAlias `mapstructure:"array_alias"`
	Vunique    string
}

type SquashOnNonStructType struct {
	InvalidSquashType int `mapstructure:",squash"`
}

type Map struct {
	Vfoo   string
	Vother map[string]string
}

type MapOfStruct struct {
	Value map[string]Basic
}

type Nested struct {
	Vfoo string
	Vbar Basic
}

type NestedPointer struct {
	Vfoo string
	Vbar *Basic
}

type NilInterface struct {
	W io.Writer
}

type NilPointer struct {
	Value *string
}

type Slice struct {
	Vfoo string
	Vbar []string
}

type SliceOfAlias struct {
	Vfoo string
	Vbar SliceAlias
}

type SliceOfStruct struct {
	Value []Basic
}

type SlicePointer struct {
	Vbar *[]string
}

type Array struct {
	Vfoo string
	Vbar [2]string
}

type ArrayOfStruct struct {
	Value [2]Basic
}

type Func struct {
	Foo func() string
}

type Tagged struct {
	Extra string `mapstructure:"bar,what,what"`
	Value string `mapstructure:"foo"`
}

type TypeConversionResult struct {
	IntToFloat         float32
	IntToUint          uint
	IntToBool          bool
	IntToString        string
	UintToInt          int
	UintToFloat        float32
	UintToBool         bool
	UintToString       string
	BoolToInt          int
	BoolToUint         uint
	BoolToFloat        float32
	BoolToString       string
	FloatToInt         int
	FloatToUint        uint
	FloatToBool        bool
	FloatToString      string
	SliceUint8ToString string
	StringToSliceUint8 []byte
	ArrayUint8ToString string
	StringToInt        int
	StringToUint       uint
	StringToBool       bool
	StringToFloat      float32
	StringToStrSlice   []string
	StringToIntSlice   []int
	StringToStrArray   [1]string
	StringToIntArray   [1]int
	SliceToMap         map[string]interface{}
	MapToSlice         []interface{}
	ArrayToMap         map[string]interface{}
	MapToArray         [1]interface{}
}

func TestBasicTypes(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring":     "foo",
		"vint":        42,
		"Vuint":       42,
		"vbool":       true,
		"Vfloat":      42.42,
		"vsilent":     true,
		"vdata":       42,
		"vjsonInt":    json.Number("1234"),
		"vjsonFloat":  json.Number("1234.5"),
		"vjsonNumber": json.Number("1234.5"),
	}

	var result Basic
	err := Decode(input, &result)
	if err != nil {
		t.Errorf("got an err: %s", err.Error())
		t.FailNow()
	}

	if result.Vstring != "foo" {
		t.Errorf("vstring value should be 'foo': %#v", result.Vstring)
	}

	if result.Vint != 42 {
		t.Errorf("vint value should be 42: %#v", result.Vint)
	}

	if result.Vuint != 42 {
		t.Errorf("vuint value should be 42: %#v", result.Vuint)
	}

	if result.Vbool != true {
		t.Errorf("vbool value should be true: %#v", result.Vbool)
	}

	if result.Vfloat != 42.42 {
		t.Errorf("vfloat value should be 42.42: %#v", result.Vfloat)
	}

	if result.Vextra != "" {
		t.Errorf("vextra value should be empty: %#v", result.Vextra)
	}

	if result.vsilent != false {
		t.Error("vsilent should not be set, it is unexported")
	}

	if result.Vdata != 42 {
		t.Error("vdata should be valid")
	}

	if result.VjsonInt != 1234 {
		t.Errorf("vjsonint value should be 1234: %#v", result.VjsonInt)
	}

	if result.VjsonFloat != 1234.5 {
		t.Errorf("vjsonfloat value should be 1234.5: %#v", result.VjsonFloat)
	}

	if !reflect.DeepEqual(result.VjsonNumber, json.Number("1234.5")) {
		t.Errorf("vjsonnumber value should be '1234.5': %T, %#v", result.VjsonNumber, result.VjsonNumber)
	}
}

func TestBasic_IntWithFloat(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vint": float64(42),
	}

	var result Basic
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}
}

func TestBasic_Merge(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vint": 42,
	}

	var result Basic
	result.Vuint = 100
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	expected := Basic{
		Vint:  42,
		Vuint: 100,
	}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("bad: %#v", result)
	}
}

// Test for issue #46.
func TestBasic_Struct(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vdata": map[string]interface{}{
			"vstring": "foo",
		},
	}

	var result, inner Basic
	result.Vdata = &inner
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}
	expected := Basic{
		Vdata: &Basic{
			Vstring: "foo",
		},
	}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("bad: %#v", result)
	}
}

func TestDecode_BasicSquash(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": "foo",
	}

	var result BasicSquash
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Test.Vstring != "foo" {
		t.Errorf("vstring value should be 'foo': %#v", result.Test.Vstring)
	}
}

func TestDecodeFrom_BasicSquash(t *testing.T) {
	t.Parallel()

	var v interface{}
	var ok bool

	input := BasicSquash{
		Test: Basic{
			Vstring: "foo",
		},
	}

	var result map[string]interface{}
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if _, ok = result["Test"]; ok {
		t.Error("test should not be present in map")
	}

	v, ok = result["Vstring"]
	if !ok {
		t.Error("vstring should be present in map")
	} else if !reflect.DeepEqual(v, "foo") {
		t.Errorf("vstring value should be 'foo': %#v", v)
	}
}

func TestDecode_Embedded(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": "foo",
		"Basic": map[string]interface{}{
			"vstring": "innerfoo",
		},
		"vunique": "bar",
	}

	var result Embedded
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Vstring != "innerfoo" {
		t.Errorf("vstring value should be 'innerfoo': %#v", result.Vstring)
	}

	if result.Vunique != "bar" {
		t.Errorf("vunique value should be 'bar': %#v", result.Vunique)
	}
}

func TestDecode_EmbeddedPointer(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": "foo",
		"Basic": map[string]interface{}{
			"vstring": "innerfoo",
		},
		"vunique": "bar",
	}

	var result EmbeddedPointer
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := EmbeddedPointer{
		Basic: &Basic{
			Vstring: "innerfoo",
		},
		Vunique: "bar",
	}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("bad: %#v", result)
	}
}

func TestDecode_EmbeddedSlice(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"slice_alias": []string{"foo", "bar"},
		"vunique":     "bar",
	}

	var result EmbeddedSlice
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if !reflect.DeepEqual(result.SliceAlias, SliceAlias([]string{"foo", "bar"})) {
		t.Errorf("slice value: %#v", result.SliceAlias)
	}

	if result.Vunique != "bar" {
		t.Errorf("vunique value should be 'bar': %#v", result.Vunique)
	}
}

func TestDecode_EmbeddedArray(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"array_alias": [2]string{"foo", "bar"},
		"vunique":     "bar",
	}

	var result EmbeddedArray
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if !reflect.DeepEqual(result.ArrayAlias, ArrayAlias([2]string{"foo", "bar"})) {
		t.Errorf("array value: %#v", result.ArrayAlias)
	}

	if result.Vunique != "bar" {
		t.Errorf("vunique value should be 'bar': %#v", result.Vunique)
	}
}

func TestDecode_EmbeddedSquash(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": "foo",
		"vunique": "bar",
	}

	var result EmbeddedSquash
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Vstring != "foo" {
		t.Errorf("vstring value should be 'foo': %#v", result.Vstring)
	}

	if result.Vunique != "bar" {
		t.Errorf("vunique value should be 'bar': %#v", result.Vunique)
	}
}

func TestDecodeFrom_EmbeddedSquash(t *testing.T) {
	t.Parallel()

	var v interface{}
	var ok bool

	input := EmbeddedSquash{
		Basic: Basic{
			Vstring: "foo",
		},
		Vunique: "bar",
	}

	var result map[string]interface{}
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if _, ok = result["Basic"]; ok {
		t.Error("basic should not be present in map")
	}

	v, ok = result["Vstring"]
	if !ok {
		t.Error("vstring should be present in map")
	} else if !reflect.DeepEqual(v, "foo") {
		t.Errorf("vstring value should be 'foo': %#v", v)
	}

	v, ok = result["Vunique"]
	if !ok {
		t.Error("vunique should be present in map")
	} else if !reflect.DeepEqual(v, "bar") {
		t.Errorf("vunique value should be 'bar': %#v", v)
	}
}

func TestDecode_SquashOnNonStructType(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"InvalidSquashType": 42,
	}

	var result SquashOnNonStructType
	err := Decode(input, &result)
	if err == nil {
		t.Fatal("unexpected success decoding invalid squash field type")
	} else if !strings.Contains(err.Error(), "unsupported type for squash") {
		t.Fatalf("unexpected error message for invalid squash field type: %s", err)
	}
}

func TestDecode_DecodeHook(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vint": "WHAT",
	}

	decodeHook := func(from reflect.Kind, to reflect.Kind, v interface{}) (interface{}, error) {
		if from == reflect.String && to != reflect.String {
			return 5, nil
		}

		return v, nil
	}

	var result Basic
	config := &DecoderConfig{
		DecodeHook: decodeHook,
		Result:     &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if result.Vint != 5 {
		t.Errorf("vint should be 5: %#v", result.Vint)
	}
}

func TestDecode_DecodeHookType(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vint": "WHAT",
	}

	decodeHook := func(from reflect.Type, to reflect.Type, v interface{}) (interface{}, error) {
		if from.Kind() == reflect.String &&
			to.Kind() != reflect.String {
			return 5, nil
		}

		return v, nil
	}

	var result Basic
	config := &DecoderConfig{
		DecodeHook: decodeHook,
		Result:     &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if result.Vint != 5 {
		t.Errorf("vint should be 5: %#v", result.Vint)
	}
}

func TestDecode_Nil(t *testing.T) {
	t.Parallel()

	var input interface{}
	result := Basic{
		Vstring: "foo",
	}

	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if result.Vstring != "foo" {
		t.Fatalf("bad: %#v", result.Vstring)
	}
}

func TestDecode_NilInterfaceHook(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"w": "",
	}

	decodeHook := func(f, t reflect.Type, v interface{}) (interface{}, error) {
		if t.String() == "io.Writer" {
			return nil, nil
		}

		return v, nil
	}

	var result NilInterface
	config := &DecoderConfig{
		DecodeHook: decodeHook,
		Result:     &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if result.W != nil {
		t.Errorf("W should be nil: %#v", result.W)
	}
}

func TestDecode_NilPointerHook(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"value": "",
	}

	decodeHook := func(f, t reflect.Type, v interface{}) (interface{}, error) {
		if typed, ok := v.(string); ok {
			if typed == "" {
				return nil, nil
			}
		}
		return v, nil
	}

	var result NilPointer
	config := &DecoderConfig{
		DecodeHook: decodeHook,
		Result:     &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if result.Value != nil {
		t.Errorf("W should be nil: %#v", result.Value)
	}
}

func TestDecode_FuncHook(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"foo": "baz",
	}

	decodeHook := func(f, t reflect.Type, v interface{}) (interface{}, error) {
		if t.Kind() != reflect.Func {
			return v, nil
		}
		val := v.(string)
		return func() string { return val }, nil
	}

	var result Func
	config := &DecoderConfig{
		DecodeHook: decodeHook,
		Result:     &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if result.Foo() != "baz" {
		t.Errorf("Foo call result should be 'baz': %s", result.Foo())
	}
}

func TestDecode_NonStruct(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"foo": "bar",
		"bar": "baz",
	}

	var result map[string]string
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if result["foo"] != "bar" {
		t.Fatal("foo is not bar")
	}
}

func TestDecode_StructMatch(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vbar": Basic{
			Vstring: "foo",
		},
	}

	var result Nested
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err.Error())
	}

	if result.Vbar.Vstring != "foo" {
		t.Errorf("bad: %#v", result)
	}
}

func TestDecode_TypeConversion(t *testing.T) {
	input := map[string]interface{}{
		"IntToFloat":         42,
		"IntToUint":          42,
		"IntToBool":          1,
		"IntToString":        42,
		"UintToInt":          42,
		"UintToFloat":        42,
		"UintToBool":         42,
		"UintToString":       42,
		"BoolToInt":          true,
		"BoolToUint":         true,
		"BoolToFloat":        true,
		"BoolToString":       true,
		"FloatToInt":         42.42,
		"FloatToUint":        42.42,
		"FloatToBool":        42.42,
		"FloatToString":      42.42,
		"SliceUint8ToString": []uint8("foo"),
		"StringToSliceUint8": "foo",
		"ArrayUint8ToString": [3]uint8{'f', 'o', 'o'},
		"StringToInt":        "42",
		"StringToUint":       "42",
		"StringToBool":       "1",
		"StringToFloat":      "42.42",
		"StringToStrSlice":   "A",
		"StringToIntSlice":   "42",
		"StringToStrArray":   "A",
		"StringToIntArray":   "42",
		"SliceToMap":         []interface{}{},
		"MapToSlice":         map[string]interface{}{},
		"ArrayToMap":         []interface{}{},
		"MapToArray":         map[string]interface{}{},
	}

	expectedResultStrict := TypeConversionResult{
		IntToFloat:  42.0,
		IntToUint:   42,
		UintToInt:   42,
		UintToFloat: 42,
		BoolToInt:   0,
		BoolToUint:  0,
		BoolToFloat: 0,
		FloatToInt:  42,
		FloatToUint: 42,
	}

	expectedResultWeak := TypeConversionResult{
		IntToFloat:         42.0,
		IntToUint:          42,
		IntToBool:          true,
		IntToString:        "42",
		UintToInt:          42,
		UintToFloat:        42,
		UintToBool:         true,
		UintToString:       "42",
		BoolToInt:          1,
		BoolToUint:         1,
		BoolToFloat:        1,
		BoolToString:       "1",
		FloatToInt:         42,
		FloatToUint:        42,
		FloatToBool:        true,
		FloatToString:      "42.42",
		SliceUint8ToString: "foo",
		StringToSliceUint8: []byte("foo"),
		ArrayUint8ToString: "foo",
		StringToInt:        42,
		StringToUint:       42,
		StringToBool:       true,
		StringToFloat:      42.42,
		StringToStrSlice:   []string{"A"},
		StringToIntSlice:   []int{42},
		StringToStrArray:   [1]string{"A"},
		StringToIntArray:   [1]int{42},
		SliceToMap:         map[string]interface{}{},
		MapToSlice:         []interface{}{},
		ArrayToMap:         map[string]interface{}{},
		MapToArray:         [1]interface{}{},
	}

	// Test strict type conversion
	var resultStrict TypeConversionResult
	err := Decode(input, &resultStrict)
	if err == nil {
		t.Errorf("should return an error")
	}
	if !reflect.DeepEqual(resultStrict, expectedResultStrict) {
		t.Errorf("expected %v, got: %v", expectedResultStrict, resultStrict)
	}

	// Test weak type conversion
	var decoder *Decoder
	var resultWeak TypeConversionResult

	config := &DecoderConfig{
		WeaklyTypedInput: true,
		Result:           &resultWeak,
	}

	decoder, err = NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if !reflect.DeepEqual(resultWeak, expectedResultWeak) {
		t.Errorf("expected \n%#v, got: \n%#v", expectedResultWeak, resultWeak)
	}
}

func TestDecoder_ErrorUnused(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": "hello",
		"foo":     "bar",
	}

	var result Basic
	config := &DecoderConfig{
		ErrorUnused: true,
		Result:      &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestMap(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vother": map[interface{}]interface{}{
			"foo": "foo",
			"bar": "bar",
		},
	}

	var result Map
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an error: %s", err)
	}

	if result.Vfoo != "foo" {
		t.Errorf("vfoo value should be 'foo': %#v", result.Vfoo)
	}

	if result.Vother == nil {
		t.Fatal("vother should not be nil")
	}

	if len(result.Vother) != 2 {
		t.Error("vother should have two items")
	}

	if result.Vother["foo"] != "foo" {
		t.Errorf("'foo' key should be foo, got: %#v", result.Vother["foo"])
	}

	if result.Vother["bar"] != "bar" {
		t.Errorf("'bar' key should be bar, got: %#v", result.Vother["bar"])
	}
}

func TestMapMerge(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vother": map[interface{}]interface{}{
			"foo": "foo",
			"bar": "bar",
		},
	}

	var result Map
	result.Vother = map[string]string{"hello": "world"}
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an error: %s", err)
	}

	if result.Vfoo != "foo" {
		t.Errorf("vfoo value should be 'foo': %#v", result.Vfoo)
	}

	expected := map[string]string{
		"foo":   "foo",
		"bar":   "bar",
		"hello": "world",
	}
	if !reflect.DeepEqual(result.Vother, expected) {
		t.Errorf("bad: %#v", result.Vother)
	}
}

func TestMapOfStruct(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"value": map[string]interface{}{
			"foo": map[string]string{"vstring": "one"},
			"bar": map[string]string{"vstring": "two"},
		},
	}

	var result MapOfStruct
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got an err: %s", err)
	}

	if result.Value == nil {
		t.Fatal("value should not be nil")
	}

	if len(result.Value) != 2 {
		t.Error("value should have two items")
	}

	if result.Value["foo"].Vstring != "one" {
		t.Errorf("foo value should be 'one', got: %s", result.Value["foo"].Vstring)
	}

	if result.Value["bar"].Vstring != "two" {
		t.Errorf("bar value should be 'two', got: %s", result.Value["bar"].Vstring)
	}
}

func TestNestedType(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": map[string]interface{}{
			"vstring": "foo",
			"vint":    42,
			"vbool":   true,
		},
	}

	var result Nested
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
}

func TestNestedTypePointer(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": &map[string]interface{}{
			"vstring": "foo",
			"vint":    42,
			"vbool":   true,
		},
	}

	var result NestedPointer
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
}

// Test for issue #46.
func TestNestedTypeInterface(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": &map[string]interface{}{
			"vstring": "foo",
			"vint":    42,
			"vbool":   true,

			"vdata": map[string]interface{}{
				"vstring": "bar",
			},
		},
	}

	var result NestedPointer
	result.Vbar = new(Basic)
	result.Vbar.Vdata = new(Basic)
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

	if result.Vbar.Vdata.(*Basic).Vstring != "bar" {
		t.Errorf("vstring value should be 'bar': %#v", result.Vbar.Vdata.(*Basic).Vstring)
	}
}

func TestSlice(t *testing.T) {
	t.Parallel()

	inputStringSlice := map[string]interface{}{
		"vfoo": "foo",
		"vbar": []string{"foo", "bar", "baz"},
	}

	inputStringSlicePointer := map[string]interface{}{
		"vfoo": "foo",
		"vbar": &[]string{"foo", "bar", "baz"},
	}

	outputStringSlice := &Slice{
		"foo",
		[]string{"foo", "bar", "baz"},
	}

	testSliceInput(t, inputStringSlice, outputStringSlice)
	testSliceInput(t, inputStringSlicePointer, outputStringSlice)
}

func TestInvalidSlice(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": 42,
	}

	result := Slice{}
	err := Decode(input, &result)
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestSliceOfStruct(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"value": []map[string]interface{}{
			{"vstring": "one"},
			{"vstring": "two"},
		},
	}

	var result SliceOfStruct
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got unexpected error: %s", err)
	}

	if len(result.Value) != 2 {
		t.Fatalf("expected two values, got %d", len(result.Value))
	}

	if result.Value[0].Vstring != "one" {
		t.Errorf("first value should be 'one', got: %s", result.Value[0].Vstring)
	}

	if result.Value[1].Vstring != "two" {
		t.Errorf("second value should be 'two', got: %s", result.Value[1].Vstring)
	}
}

func TestSliceCornerCases(t *testing.T) {
	t.Parallel()

	// Input with a map with zero values
	input := map[string]interface{}{}
	var resultWeak []Basic

	err := WeakDecode(input, &resultWeak)
	if err != nil {
		t.Fatalf("got unexpected error: %s", err)
	}

	if len(resultWeak) != 0 {
		t.Errorf("length should be 0")
	}
	// Input with more values
	input = map[string]interface{}{
		"Vstring": "foo",
	}

	resultWeak = nil
	err = WeakDecode(input, &resultWeak)
	if err != nil {
		t.Fatalf("got unexpected error: %s", err)
	}

	if resultWeak[0].Vstring != "foo" {
		t.Errorf("value does not match")
	}
}

func TestSliceToMap(t *testing.T) {
	t.Parallel()

	input := []map[string]interface{}{
		{
			"foo": "bar",
		},
		{
			"bar": "baz",
		},
	}

	var result map[string]interface{}
	err := WeakDecode(input, &result)
	if err != nil {
		t.Fatalf("got an error: %s", err)
	}

	expected := map[string]interface{}{
		"foo": "bar",
		"bar": "baz",
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("bad: %#v", result)
	}
}

func TestArray(t *testing.T) {
	t.Parallel()

	inputStringArray := map[string]interface{}{
		"vfoo": "foo",
		"vbar": [2]string{"foo", "bar"},
	}

	inputStringArrayPointer := map[string]interface{}{
		"vfoo": "foo",
		"vbar": &[2]string{"foo", "bar"},
	}

	outputStringArray := &Array{
		"foo",
		[2]string{"foo", "bar"},
	}

	testArrayInput(t, inputStringArray, outputStringArray)
	testArrayInput(t, inputStringArrayPointer, outputStringArray)
}

func TestInvalidArray(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": 42,
	}

	result := Array{}
	err := Decode(input, &result)
	if err == nil {
		t.Errorf("expected failure")
	}
}

func TestArrayOfStruct(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"value": []map[string]interface{}{
			{"vstring": "one"},
			{"vstring": "two"},
		},
	}

	var result ArrayOfStruct
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got unexpected error: %s", err)
	}

	if len(result.Value) != 2 {
		t.Fatalf("expected two values, got %d", len(result.Value))
	}

	if result.Value[0].Vstring != "one" {
		t.Errorf("first value should be 'one', got: %s", result.Value[0].Vstring)
	}

	if result.Value[1].Vstring != "two" {
		t.Errorf("second value should be 'two', got: %s", result.Value[1].Vstring)
	}
}

func TestArrayToMap(t *testing.T) {
	t.Parallel()

	input := []map[string]interface{}{
		{
			"foo": "bar",
		},
		{
			"bar": "baz",
		},
	}

	var result map[string]interface{}
	err := WeakDecode(input, &result)
	if err != nil {
		t.Fatalf("got an error: %s", err)
	}

	expected := map[string]interface{}{
		"foo": "bar",
		"bar": "baz",
	}
	if !reflect.DeepEqual(result, expected) {
		t.Errorf("bad: %#v", result)
	}
}

func TestDecodeTable(t *testing.T) {
	t.Parallel()

	// We need to make new types so that we don't get the short-circuit
	// copy functionality. We want to test the deep copying functionality.
	type BasicCopy Basic
	type NestedPointerCopy NestedPointer
	type MapCopy Map

	tests := []struct {
		name    string
		in      interface{}
		target  interface{}
		out     interface{}
		wantErr bool
	}{
		{
			"basic struct input",
			&Basic{
				Vstring: "vstring",
				Vint:    2,
				Vuint:   3,
				Vbool:   true,
				Vfloat:  4.56,
				Vextra:  "vextra",
				vsilent: true,
				Vdata:   []byte("data"),
			},
			&map[string]interface{}{},
			&map[string]interface{}{
				"Vstring":     "vstring",
				"Vint":        2,
				"Vuint":       uint(3),
				"Vbool":       true,
				"Vfloat":      4.56,
				"Vextra":      "vextra",
				"Vdata":       []byte("data"),
				"VjsonInt":    0,
				"VjsonFloat":  0.0,
				"VjsonNumber": json.Number(""),
			},
			false,
		},
		{
			"embedded struct input",
			&Embedded{
				Vunique: "vunique",
				Basic: Basic{
					Vstring: "vstring",
					Vint:    2,
					Vuint:   3,
					Vbool:   true,
					Vfloat:  4.56,
					Vextra:  "vextra",
					vsilent: true,
					Vdata:   []byte("data"),
				},
			},
			&map[string]interface{}{},
			&map[string]interface{}{
				"Vunique": "vunique",
				"Basic": map[string]interface{}{
					"Vstring":     "vstring",
					"Vint":        2,
					"Vuint":       uint(3),
					"Vbool":       true,
					"Vfloat":      4.56,
					"Vextra":      "vextra",
					"Vdata":       []byte("data"),
					"VjsonInt":    0,
					"VjsonFloat":  0.0,
					"VjsonNumber": json.Number(""),
				},
			},
			false,
		},
		{
			"struct => struct",
			&Basic{
				Vstring: "vstring",
				Vint:    2,
				Vuint:   3,
				Vbool:   true,
				Vfloat:  4.56,
				Vextra:  "vextra",
				Vdata:   []byte("data"),
				vsilent: true,
			},
			&BasicCopy{},
			&BasicCopy{
				Vstring: "vstring",
				Vint:    2,
				Vuint:   3,
				Vbool:   true,
				Vfloat:  4.56,
				Vextra:  "vextra",
				Vdata:   []byte("data"),
			},
			false,
		},
		{
			"struct => struct with pointers",
			&NestedPointer{
				Vfoo: "hello",
				Vbar: nil,
			},
			&NestedPointerCopy{},
			&NestedPointerCopy{
				Vfoo: "hello",
			},
			false,
		},
		{
			"basic pointer to non-pointer",
			&BasicPointer{
				Vstring: stringPtr("vstring"),
				Vint:    intPtr(2),
				Vuint:   uintPtr(3),
				Vbool:   boolPtr(true),
				Vfloat:  floatPtr(4.56),
				Vdata:   interfacePtr([]byte("data")),
			},
			&Basic{},
			&Basic{
				Vstring: "vstring",
				Vint:    2,
				Vuint:   3,
				Vbool:   true,
				Vfloat:  4.56,
				Vdata:   []byte("data"),
			},
			false,
		},
		{
			"slice non-pointer to pointer",
			&Slice{},
			&SlicePointer{},
			&SlicePointer{},
			false,
		},
		{
			"slice non-pointer to pointer, zero field",
			&Slice{},
			&SlicePointer{
				Vbar: &[]string{"yo"},
			},
			&SlicePointer{},
			false,
		},
		{
			"slice to slice alias",
			&Slice{},
			&SliceOfAlias{},
			&SliceOfAlias{},
			false,
		},
		{
			"nil map to map",
			&Map{},
			&MapCopy{},
			&MapCopy{},
			false,
		},
		{
			"nil map to non-empty map",
			&Map{},
			&MapCopy{Vother: map[string]string{"foo": "bar"}},
			&MapCopy{},
			false,
		},

		{
			"slice input - should error",
			[]string{"foo", "bar"},
			&map[string]interface{}{},
			&map[string]interface{}{},
			true,
		},
		{
			"struct with slice property",
			&Slice{
				Vfoo: "vfoo",
				Vbar: []string{"foo", "bar"},
			},
			&map[string]interface{}{},
			&map[string]interface{}{
				"Vfoo": "vfoo",
				"Vbar": []string{"foo", "bar"},
			},
			false,
		},
		{
			"struct with slice of struct property",
			&SliceOfStruct{
				Value: []Basic{
					Basic{
						Vstring: "vstring",
						Vint:    2,
						Vuint:   3,
						Vbool:   true,
						Vfloat:  4.56,
						Vextra:  "vextra",
						vsilent: true,
						Vdata:   []byte("data"),
					},
				},
			},
			&map[string]interface{}{},
			&map[string]interface{}{
				"Value": []Basic{
					Basic{
						Vstring: "vstring",
						Vint:    2,
						Vuint:   3,
						Vbool:   true,
						Vfloat:  4.56,
						Vextra:  "vextra",
						vsilent: true,
						Vdata:   []byte("data"),
					},
				},
			},
			false,
		},
		{
			"struct with map property",
			&Map{
				Vfoo:   "vfoo",
				Vother: map[string]string{"vother": "vother"},
			},
			&map[string]interface{}{},
			&map[string]interface{}{
				"Vfoo": "vfoo",
				"Vother": map[string]string{
					"vother": "vother",
				}},
			false,
		},
		{
			"tagged struct",
			&Tagged{
				Extra: "extra",
				Value: "value",
			},
			&map[string]string{},
			&map[string]string{
				"bar": "extra",
				"foo": "value",
			},
			false,
		},
		{
			"omit tag struct",
			&struct {
				Value string `mapstructure:"value"`
				Omit  string `mapstructure:"-"`
			}{
				Value: "value",
				Omit:  "omit",
			},
			&map[string]string{},
			&map[string]string{
				"value": "value",
			},
			false,
		},
		{
			"decode to wrong map type",
			&struct {
				Value string
			}{
				Value: "string",
			},
			&map[string]int{},
			&map[string]int{},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := Decode(tt.in, tt.target); (err != nil) != tt.wantErr {
				t.Fatalf("%q: TestMapOutputForStructuredInputs() unexpected error: %s", tt.name, err)
			}

			if !reflect.DeepEqual(tt.out, tt.target) {
				t.Fatalf("%q: TestMapOutputForStructuredInputs() expected: %#v, got: %#v", tt.name, tt.out, tt.target)
			}
		})
	}
}

func TestInvalidType(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": 42,
	}

	var result Basic
	err := Decode(input, &result)
	if err == nil {
		t.Fatal("error should exist")
	}

	derr, ok := err.(*Error)
	if !ok {
		t.Fatalf("error should be kind of Error, instead: %#v", err)
	}

	if derr.Errors[0] != "'Vstring' expected type 'string', got unconvertible type 'int'" {
		t.Errorf("got unexpected error: %s", err)
	}

	inputNegIntUint := map[string]interface{}{
		"vuint": -42,
	}

	err = Decode(inputNegIntUint, &result)
	if err == nil {
		t.Fatal("error should exist")
	}

	derr, ok = err.(*Error)
	if !ok {
		t.Fatalf("error should be kind of Error, instead: %#v", err)
	}

	if derr.Errors[0] != "cannot parse 'Vuint', -42 overflows uint" {
		t.Errorf("got unexpected error: %s", err)
	}

	inputNegFloatUint := map[string]interface{}{
		"vuint": -42.0,
	}

	err = Decode(inputNegFloatUint, &result)
	if err == nil {
		t.Fatal("error should exist")
	}

	derr, ok = err.(*Error)
	if !ok {
		t.Fatalf("error should be kind of Error, instead: %#v", err)
	}

	if derr.Errors[0] != "cannot parse 'Vuint', -42.000000 overflows uint" {
		t.Errorf("got unexpected error: %s", err)
	}
}

func TestDecodeMetadata(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": map[string]interface{}{
			"vstring": "foo",
			"Vuint":   42,
			"foo":     "bar",
		},
		"bar": "nil",
	}

	var md Metadata
	var result Nested

	err := DecodeMetadata(input, &result, &md)
	if err != nil {
		t.Fatalf("err: %s", err.Error())
	}

	expectedKeys := []string{"Vbar", "Vbar.Vstring", "Vbar.Vuint", "Vfoo"}
	sort.Strings(md.Keys)
	if !reflect.DeepEqual(md.Keys, expectedKeys) {
		t.Fatalf("bad keys: %#v", md.Keys)
	}

	expectedUnused := []string{"Vbar.foo", "bar"}
	if !reflect.DeepEqual(md.Unused, expectedUnused) {
		t.Fatalf("bad unused: %#v", md.Unused)
	}
}

func TestMetadata(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vfoo": "foo",
		"vbar": map[string]interface{}{
			"vstring": "foo",
			"Vuint":   42,
			"foo":     "bar",
		},
		"bar": "nil",
	}

	var md Metadata
	var result Nested
	config := &DecoderConfig{
		Metadata: &md,
		Result:   &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("err: %s", err.Error())
	}

	expectedKeys := []string{"Vbar", "Vbar.Vstring", "Vbar.Vuint", "Vfoo"}
	sort.Strings(md.Keys)
	if !reflect.DeepEqual(md.Keys, expectedKeys) {
		t.Fatalf("bad keys: %#v", md.Keys)
	}

	expectedUnused := []string{"Vbar.foo", "bar"}
	if !reflect.DeepEqual(md.Unused, expectedUnused) {
		t.Fatalf("bad unused: %#v", md.Unused)
	}
}

func TestMetadata_Embedded(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"vstring": "foo",
		"vunique": "bar",
	}

	var md Metadata
	var result EmbeddedSquash
	config := &DecoderConfig{
		Metadata: &md,
		Result:   &result,
	}

	decoder, err := NewDecoder(config)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = decoder.Decode(input)
	if err != nil {
		t.Fatalf("err: %s", err.Error())
	}

	expectedKeys := []string{"Vstring", "Vunique"}

	sort.Strings(md.Keys)
	if !reflect.DeepEqual(md.Keys, expectedKeys) {
		t.Fatalf("bad keys: %#v", md.Keys)
	}

	expectedUnused := []string{}
	if !reflect.DeepEqual(md.Unused, expectedUnused) {
		t.Fatalf("bad unused: %#v", md.Unused)
	}
}

func TestNonPtrValue(t *testing.T) {
	t.Parallel()

	err := Decode(map[string]interface{}{}, Basic{})
	if err == nil {
		t.Fatal("error should exist")
	}

	if err.Error() != "result must be a pointer" {
		t.Errorf("got unexpected error: %s", err)
	}
}

func TestTagged(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"foo": "bar",
		"bar": "value",
	}

	var result Tagged
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}

	if result.Value != "bar" {
		t.Errorf("value should be 'bar', got: %#v", result.Value)
	}

	if result.Extra != "value" {
		t.Errorf("extra should be 'value', got: %#v", result.Extra)
	}
}

func TestWeakDecode(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"foo": "4",
		"bar": "value",
	}

	var result struct {
		Foo int
		Bar string
	}

	if err := WeakDecode(input, &result); err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.Foo != 4 {
		t.Fatalf("bad: %#v", result)
	}
	if result.Bar != "value" {
		t.Fatalf("bad: %#v", result)
	}
}

func TestWeakDecodeMetadata(t *testing.T) {
	t.Parallel()

	input := map[string]interface{}{
		"foo":    "4",
		"bar":    "value",
		"unused": "value",
	}

	var md Metadata
	var result struct {
		Foo int
		Bar string
	}

	if err := WeakDecodeMetadata(input, &result, &md); err != nil {
		t.Fatalf("err: %s", err)
	}
	if result.Foo != 4 {
		t.Fatalf("bad: %#v", result)
	}
	if result.Bar != "value" {
		t.Fatalf("bad: %#v", result)
	}

	expectedKeys := []string{"Bar", "Foo"}
	sort.Strings(md.Keys)
	if !reflect.DeepEqual(md.Keys, expectedKeys) {
		t.Fatalf("bad keys: %#v", md.Keys)
	}

	expectedUnused := []string{"unused"}
	if !reflect.DeepEqual(md.Unused, expectedUnused) {
		t.Fatalf("bad unused: %#v", md.Unused)
	}
}

func testSliceInput(t *testing.T, input map[string]interface{}, expected *Slice) {
	var result Slice
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got error: %s", err)
	}

	if result.Vfoo != expected.Vfoo {
		t.Errorf("Vfoo expected '%s', got '%s'", expected.Vfoo, result.Vfoo)
	}

	if result.Vbar == nil {
		t.Fatalf("Vbar a slice, got '%#v'", result.Vbar)
	}

	if len(result.Vbar) != len(expected.Vbar) {
		t.Errorf("Vbar length should be %d, got %d", len(expected.Vbar), len(result.Vbar))
	}

	for i, v := range result.Vbar {
		if v != expected.Vbar[i] {
			t.Errorf(
				"Vbar[%d] should be '%#v', got '%#v'",
				i, expected.Vbar[i], v)
		}
	}
}

func testArrayInput(t *testing.T, input map[string]interface{}, expected *Array) {
	var result Array
	err := Decode(input, &result)
	if err != nil {
		t.Fatalf("got error: %s", err)
	}

	if result.Vfoo != expected.Vfoo {
		t.Errorf("Vfoo expected '%s', got '%s'", expected.Vfoo, result.Vfoo)
	}

	if result.Vbar == [2]string{} {
		t.Fatalf("Vbar a slice, got '%#v'", result.Vbar)
	}

	if len(result.Vbar) != len(expected.Vbar) {
		t.Errorf("Vbar length should be %d, got %d", len(expected.Vbar), len(result.Vbar))
	}

	for i, v := range result.Vbar {
		if v != expected.Vbar[i] {
			t.Errorf(
				"Vbar[%d] should be '%#v', got '%#v'",
				i, expected.Vbar[i], v)
		}
	}
}

func stringPtr(v string) *string              { return &v }
func intPtr(v int) *int                       { return &v }
func uintPtr(v uint) *uint                    { return &v }
func boolPtr(v bool) *bool                    { return &v }
func floatPtr(v float64) *float64             { return &v }
func interfacePtr(v interface{}) *interface{} { return &v }
