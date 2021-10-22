package dynamodbattribute

import (
	"math"
	"reflect"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

type simpleMarshalStruct struct {
	Byte    []byte
	String  string
	Int     int
	Uint    uint
	Float32 float32
	Float64 float64
	Bool    bool
	Null    *interface{}
}

type complexMarshalStruct struct {
	Simple []simpleMarshalStruct
}

type myByteStruct struct {
	Byte []byte
}

type myByteSetStruct struct {
	ByteSet [][]byte
}

type marshallerTestInput struct {
	input    interface{}
	expected interface{}
	err      awserr.Error
}

var marshalerScalarInputs = []marshallerTestInput{
	{
		input:    nil,
		expected: &dynamodb.AttributeValue{NULL: &trueValue},
	},
	{
		input:    "some string",
		expected: &dynamodb.AttributeValue{S: aws.String("some string")},
	},
	{
		input:    true,
		expected: &dynamodb.AttributeValue{BOOL: &trueValue},
	},
	{
		input:    false,
		expected: &dynamodb.AttributeValue{BOOL: &falseValue},
	},
	{
		input:    3.14,
		expected: &dynamodb.AttributeValue{N: aws.String("3.14")},
	},
	{
		input:    math.MaxFloat32,
		expected: &dynamodb.AttributeValue{N: aws.String("340282346638528860000000000000000000000")},
	},
	{
		input:    math.MaxFloat64,
		expected: &dynamodb.AttributeValue{N: aws.String("179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")},
	},
	{
		input:    12,
		expected: &dynamodb.AttributeValue{N: aws.String("12")},
	},
	{
		input:    Number("12"),
		expected: &dynamodb.AttributeValue{N: aws.String("12")},
	},
	{
		input: simpleMarshalStruct{},
		expected: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Byte":    {NULL: &trueValue},
				"Bool":    {BOOL: &falseValue},
				"Float32": {N: aws.String("0")},
				"Float64": {N: aws.String("0")},
				"Int":     {N: aws.String("0")},
				"Null":    {NULL: &trueValue},
				"String":  {NULL: &trueValue},
				"Uint":    {N: aws.String("0")},
			},
		},
	},
}

var marshallerMapTestInputs = []marshallerTestInput{
	// Scalar tests
	{
		input:    nil,
		expected: map[string]*dynamodb.AttributeValue{},
	},
	{
		input:    map[string]interface{}{"string": "some string"},
		expected: map[string]*dynamodb.AttributeValue{"string": {S: aws.String("some string")}},
	},
	{
		input:    map[string]interface{}{"bool": true},
		expected: map[string]*dynamodb.AttributeValue{"bool": {BOOL: &trueValue}},
	},
	{
		input:    map[string]interface{}{"bool": false},
		expected: map[string]*dynamodb.AttributeValue{"bool": {BOOL: &falseValue}},
	},
	{
		input:    map[string]interface{}{"null": nil},
		expected: map[string]*dynamodb.AttributeValue{"null": {NULL: &trueValue}},
	},
	{
		input:    map[string]interface{}{"float": 3.14},
		expected: map[string]*dynamodb.AttributeValue{"float": {N: aws.String("3.14")}},
	},
	{
		input:    map[string]interface{}{"float": math.MaxFloat32},
		expected: map[string]*dynamodb.AttributeValue{"float": {N: aws.String("340282346638528860000000000000000000000")}},
	},
	{
		input:    map[string]interface{}{"float": math.MaxFloat64},
		expected: map[string]*dynamodb.AttributeValue{"float": {N: aws.String("179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")}},
	},
	{
		input:    map[string]interface{}{"num": 12.},
		expected: map[string]*dynamodb.AttributeValue{"num": {N: aws.String("12")}},
	},
	{
		input:    map[string]interface{}{"byte": []byte{48, 49}},
		expected: map[string]*dynamodb.AttributeValue{"byte": {B: []byte{48, 49}}},
	},
	{
		input:    struct{ Byte []byte }{Byte: []byte{48, 49}},
		expected: map[string]*dynamodb.AttributeValue{"Byte": {B: []byte{48, 49}}},
	},
	{
		input:    map[string]interface{}{"byte_set": [][]byte{{48, 49}, {50, 51}}},
		expected: map[string]*dynamodb.AttributeValue{"byte_set": {BS: [][]byte{{48, 49}, {50, 51}}}},
	},
	{
		input:    struct{ ByteSet [][]byte }{ByteSet: [][]byte{{48, 49}, {50, 51}}},
		expected: map[string]*dynamodb.AttributeValue{"ByteSet": {BS: [][]byte{{48, 49}, {50, 51}}}},
	},
	// List
	{
		input: map[string]interface{}{"list": []interface{}{"a string", 12., 3.14, true, nil, false}},
		expected: map[string]*dynamodb.AttributeValue{
			"list": {
				L: []*dynamodb.AttributeValue{
					{S: aws.String("a string")},
					{N: aws.String("12")},
					{N: aws.String("3.14")},
					{BOOL: &trueValue},
					{NULL: &trueValue},
					{BOOL: &falseValue},
				},
			},
		},
	},
	// Map
	{
		input: map[string]interface{}{"map": map[string]interface{}{"nestednum": 12.}},
		expected: map[string]*dynamodb.AttributeValue{
			"map": {
				M: map[string]*dynamodb.AttributeValue{
					"nestednum": {
						N: aws.String("12"),
					},
				},
			},
		},
	},
	// Structs
	{
		input: simpleMarshalStruct{},
		expected: map[string]*dynamodb.AttributeValue{
			"Byte":    {NULL: &trueValue},
			"Bool":    {BOOL: &falseValue},
			"Float32": {N: aws.String("0")},
			"Float64": {N: aws.String("0")},
			"Int":     {N: aws.String("0")},
			"Null":    {NULL: &trueValue},
			"String":  {NULL: &trueValue},
			"Uint":    {N: aws.String("0")},
		},
	},
	{
		input: complexMarshalStruct{},
		expected: map[string]*dynamodb.AttributeValue{
			"Simple": {NULL: &trueValue},
		},
	},
	{
		input: struct {
			Simple []string `json:"simple"`
		}{},
		expected: map[string]*dynamodb.AttributeValue{
			"simple": {NULL: &trueValue},
		},
	},
	{
		input: struct {
			Simple []string `json:"simple,omitempty"`
		}{},
		expected: map[string]*dynamodb.AttributeValue{},
	},
	{
		input: struct {
			Simple []string `json:"-"`
		}{},
		expected: map[string]*dynamodb.AttributeValue{},
	},
	{
		input: complexMarshalStruct{Simple: []simpleMarshalStruct{{Int: -2}, {Uint: 5}}},
		expected: map[string]*dynamodb.AttributeValue{
			"Simple": {
				L: []*dynamodb.AttributeValue{
					{
						M: map[string]*dynamodb.AttributeValue{
							"Byte":    {NULL: &trueValue},
							"Bool":    {BOOL: &falseValue},
							"Float32": {N: aws.String("0")},
							"Float64": {N: aws.String("0")},
							"Int":     {N: aws.String("-2")},
							"Null":    {NULL: &trueValue},
							"String":  {NULL: &trueValue},
							"Uint":    {N: aws.String("0")},
						},
					},
					{
						M: map[string]*dynamodb.AttributeValue{
							"Byte":    {NULL: &trueValue},
							"Bool":    {BOOL: &falseValue},
							"Float32": {N: aws.String("0")},
							"Float64": {N: aws.String("0")},
							"Int":     {N: aws.String("0")},
							"Null":    {NULL: &trueValue},
							"String":  {NULL: &trueValue},
							"Uint":    {N: aws.String("5")},
						},
					},
				},
			},
		},
	},
}

var marshallerListTestInputs = []marshallerTestInput{
	{
		input:    nil,
		expected: []*dynamodb.AttributeValue{},
	},
	{
		input:    []interface{}{},
		expected: []*dynamodb.AttributeValue{},
	},
	{
		input:    []simpleMarshalStruct{},
		expected: []*dynamodb.AttributeValue{},
	},
	{
		input: []interface{}{"a string", 12., 3.14, true, nil, false},
		expected: []*dynamodb.AttributeValue{
			{S: aws.String("a string")},
			{N: aws.String("12")},
			{N: aws.String("3.14")},
			{BOOL: &trueValue},
			{NULL: &trueValue},
			{BOOL: &falseValue},
		},
	},
	{
		input: []simpleMarshalStruct{{}},
		expected: []*dynamodb.AttributeValue{
			{
				M: map[string]*dynamodb.AttributeValue{
					"Byte":    {NULL: &trueValue},
					"Bool":    {BOOL: &falseValue},
					"Float32": {N: aws.String("0")},
					"Float64": {N: aws.String("0")},
					"Int":     {N: aws.String("0")},
					"Null":    {NULL: &trueValue},
					"String":  {NULL: &trueValue},
					"Uint":    {N: aws.String("0")},
				},
			},
		},
	},
}

func Test_New_Marshal(t *testing.T) {
	for _, test := range marshalerScalarInputs {
		testMarshal(t, test)
	}
}

func testMarshal(t *testing.T, test marshallerTestInput) {
	actual, err := Marshal(test.input)
	if test.err != nil {
		if err == nil {
			t.Errorf("Marshal with input %#v retured %#v, expected error `%s`", test.input, actual, test.err)
		} else if err.Error() != test.err.Error() {
			t.Errorf("Marshal with input %#v retured error `%s`, expected error `%s`", test.input, err, test.err)
		}
	} else {
		if err != nil {
			t.Errorf("Marshal with input %#v retured error `%s`", test.input, err)
		}
		compareObjects(t, test.expected, actual)
	}
}

func Test_New_Unmarshal(t *testing.T) {
	// Using the same inputs from Marshal, test the reverse mapping.
	for i, test := range marshalerScalarInputs {
		if test.input == nil {
			continue
		}
		actual := reflect.New(reflect.TypeOf(test.input)).Interface()
		if err := Unmarshal(test.expected.(*dynamodb.AttributeValue), actual); err != nil {
			t.Errorf("Unmarshal %d, with input %#v retured error `%s`", i+1, test.expected, err)
		}
		compareObjects(t, test.input, reflect.ValueOf(actual).Elem().Interface())
	}
}

func Test_New_UnmarshalError(t *testing.T) {
	// Test that we get an error using Unmarshal to convert to a nil value.
	expected := &InvalidUnmarshalError{Type: reflect.TypeOf(nil)}
	if err := Unmarshal(nil, nil); err == nil {
		t.Errorf("Unmarshal with input %T returned no error, expected error `%v`", nil, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("Unmarshal with input %T returned error `%v`, expected error `%v`", nil, err, expected)
	}

	// Test that we get an error using Unmarshal to convert to a non-pointer value.
	var actual map[string]interface{}
	expected = &InvalidUnmarshalError{Type: reflect.TypeOf(actual)}
	if err := Unmarshal(nil, actual); err == nil {
		t.Errorf("Unmarshal with input %T returned no error, expected error `%v`", actual, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("Unmarshal with input %T returned error `%v`, expected error `%v`", actual, err, expected)
	}

	// Test that we get an error using Unmarshal to convert to nil struct.
	var actual2 *struct{ A int }
	expected = &InvalidUnmarshalError{Type: reflect.TypeOf(actual2)}
	if err := Unmarshal(nil, actual2); err == nil {
		t.Errorf("Unmarshal with input %T returned no error, expected error `%v`", actual2, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("Unmarshal with input %T returned error `%v`, expected error `%v`", actual2, err, expected)
	}
}

func Test_New_MarshalMap(t *testing.T) {
	for _, test := range marshallerMapTestInputs {
		testMarshalMap(t, test)
	}
}

func testMarshalMap(t *testing.T, test marshallerTestInput) {
	actual, err := MarshalMap(test.input)
	if test.err != nil {
		if err == nil {
			t.Errorf("MarshalMap with input %#v retured %#v, expected error `%s`", test.input, actual, test.err)
		} else if err.Error() != test.err.Error() {
			t.Errorf("MarshalMap with input %#v retured error `%s`, expected error `%s`", test.input, err, test.err)
		}
	} else {
		if err != nil {
			t.Errorf("MarshalMap with input %#v retured error `%s`", test.input, err)
		}
		compareObjects(t, test.expected, actual)
	}
}

func Test_New_UnmarshalMap(t *testing.T) {
	// Using the same inputs from MarshalMap, test the reverse mapping.
	for i, test := range marshallerMapTestInputs {
		if test.input == nil {
			continue
		}
		actual := reflect.New(reflect.TypeOf(test.input)).Interface()
		if err := UnmarshalMap(test.expected.(map[string]*dynamodb.AttributeValue), actual); err != nil {
			t.Errorf("Unmarshal %d, with input %#v retured error `%s`", i+1, test.expected, err)
		}
		compareObjects(t, test.input, reflect.ValueOf(actual).Elem().Interface())
	}
}

func Test_New_UnmarshalMapError(t *testing.T) {
	// Test that we get an error using UnmarshalMap to convert to a nil value.
	expected := &InvalidUnmarshalError{Type: reflect.TypeOf(nil)}
	if err := UnmarshalMap(nil, nil); err == nil {
		t.Errorf("UnmarshalMap with input %T returned no error, expected error `%v`", nil, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("UnmarshalMap with input %T returned error `%v`, expected error `%v`", nil, err, expected)
	}

	// Test that we get an error using UnmarshalMap to convert to a non-pointer value.
	var actual map[string]interface{}
	expected = &InvalidUnmarshalError{Type: reflect.TypeOf(actual)}
	if err := UnmarshalMap(nil, actual); err == nil {
		t.Errorf("UnmarshalMap with input %T returned no error, expected error `%v`", actual, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("UnmarshalMap with input %T returned error `%v`, expected error `%v`", actual, err, expected)
	}

	// Test that we get an error using UnmarshalMap to convert to nil struct.
	var actual2 *struct{ A int }
	expected = &InvalidUnmarshalError{Type: reflect.TypeOf(actual2)}
	if err := UnmarshalMap(nil, actual2); err == nil {
		t.Errorf("UnmarshalMap with input %T returned no error, expected error `%v`", actual2, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("UnmarshalMap with input %T returned error `%v`, expected error `%v`", actual2, err, expected)
	}
}

func Test_New_MarshalList(t *testing.T) {
	for _, test := range marshallerListTestInputs {
		testMarshalList(t, test)
	}
}

func testMarshalList(t *testing.T, test marshallerTestInput) {
	actual, err := MarshalList(test.input)
	if test.err != nil {
		if err == nil {
			t.Errorf("MarshalList with input %#v retured %#v, expected error `%s`", test.input, actual, test.err)
		} else if err.Error() != test.err.Error() {
			t.Errorf("MarshalList with input %#v retured error `%s`, expected error `%s`", test.input, err, test.err)
		}
	} else {
		if err != nil {
			t.Errorf("MarshalList with input %#v retured error `%s`", test.input, err)
		}
		compareObjects(t, test.expected, actual)
	}
}

func Test_New_UnmarshalList(t *testing.T) {
	// Using the same inputs from MarshalList, test the reverse mapping.
	for i, test := range marshallerListTestInputs {
		if test.input == nil {
			continue
		}
		iv := reflect.ValueOf(test.input)

		actual := reflect.New(iv.Type())
		if iv.Kind() == reflect.Slice {
			actual.Elem().Set(reflect.MakeSlice(iv.Type(), iv.Len(), iv.Cap()))
		}

		if err := UnmarshalList(test.expected.([]*dynamodb.AttributeValue), actual.Interface()); err != nil {
			t.Errorf("Unmarshal %d, with input %#v retured error `%s`", i+1, test.expected, err)
		}
		compareObjects(t, test.input, actual.Elem().Interface())
	}
}

func Test_New_UnmarshalListError(t *testing.T) {
	// Test that we get an error using UnmarshalList to convert to a nil value.
	expected := &InvalidUnmarshalError{Type: reflect.TypeOf(nil)}
	if err := UnmarshalList(nil, nil); err == nil {
		t.Errorf("UnmarshalList with input %T returned no error, expected error `%v`", nil, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("UnmarshalList with input %T returned error `%v`, expected error `%v`", nil, err, expected)
	}

	// Test that we get an error using UnmarshalList to convert to a non-pointer value.
	var actual map[string]interface{}
	expected = &InvalidUnmarshalError{Type: reflect.TypeOf(actual)}
	if err := UnmarshalList(nil, actual); err == nil {
		t.Errorf("UnmarshalList with input %T returned no error, expected error `%v`", actual, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("UnmarshalList with input %T returned error `%v`, expected error `%v`", actual, err, expected)
	}

	// Test that we get an error using UnmarshalList to convert to nil struct.
	var actual2 *struct{ A int }
	expected = &InvalidUnmarshalError{Type: reflect.TypeOf(actual2)}
	if err := UnmarshalList(nil, actual2); err == nil {
		t.Errorf("UnmarshalList with input %T returned no error, expected error `%v`", actual2, expected)
	} else if err.Error() != expected.Error() {
		t.Errorf("UnmarshalList with input %T returned error `%v`, expected error `%v`", actual2, err, expected)
	}
}

// see github issue #1594
func TestDecodeArrayType(t *testing.T) {
	cases := []struct {
		to, from interface{}
	}{
		{
			&[2]int{1, 2},
			&[2]int{},
		},
		{
			&[2]int64{1, 2},
			&[2]int64{},
		},
		{
			&[2]byte{1, 2},
			&[2]byte{},
		},
		{
			&[2]bool{true, false},
			&[2]bool{},
		},
		{
			&[2]string{"1", "2"},
			&[2]string{},
		},
		{
			&[2][]string{{"1", "2"}},
			&[2][]string{},
		},
	}

	for _, c := range cases {
		marshaled, err := Marshal(c.to)
		if err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		if err = Unmarshal(marshaled, c.from); err != nil {
			t.Errorf("expected no error, but received %v", err)
		}

		if !reflect.DeepEqual(c.to, c.from) {
			t.Errorf("expected %v, but received %v", c.to, c.from)
		}
	}
}

func compareObjects(t *testing.T, expected interface{}, actual interface{}) {
	if !reflect.DeepEqual(expected, actual) {
		ev := reflect.ValueOf(expected)
		av := reflect.ValueOf(actual)
		t.Errorf("\nExpected kind(%s,%T):\n%s\nActual kind(%s,%T):\n%s\n",
			ev.Kind(),
			ev.Interface(),
			awsutil.Prettify(expected),
			av.Kind(),
			ev.Interface(),
			awsutil.Prettify(actual))
	}
}

func BenchmarkMarshalOneMember(b *testing.B) {
	fieldCache = fieldCacher{}

	simple := simpleMarshalStruct{
		String:  "abc",
		Int:     123,
		Uint:    123,
		Float32: 123.321,
		Float64: 123.321,
		Bool:    true,
		Null:    nil,
	}

	type MyCompositeStruct struct {
		A simpleMarshalStruct `dynamodbav:"a"`
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := Marshal(MyCompositeStruct{
				A: simple,
			}); err != nil {
				b.Error("unexpected error:", err)
			}
		}
	})
}

func BenchmarkMarshalTwoMembers(b *testing.B) {
	fieldCache = fieldCacher{}

	simple := simpleMarshalStruct{
		String:  "abc",
		Int:     123,
		Uint:    123,
		Float32: 123.321,
		Float64: 123.321,
		Bool:    true,
		Null:    nil,
	}

	type MyCompositeStruct struct {
		A simpleMarshalStruct `dynamodbav:"a"`
		B simpleMarshalStruct `dynamodbav:"b"`
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := Marshal(MyCompositeStruct{
				A: simple,
				B: simple,
			}); err != nil {
				b.Error("unexpected error:", err)
			}
		}
	})
}

func BenchmarkUnmarshalOneMember(b *testing.B) {
	fieldCache = fieldCacher{}

	myStructAVMap, _ := Marshal(simpleMarshalStruct{
		String:  "abc",
		Int:     123,
		Uint:    123,
		Float32: 123.321,
		Float64: 123.321,
		Bool:    true,
		Null:    nil,
	})

	type MyCompositeStructOne struct {
		A simpleMarshalStruct `dynamodbav:"a"`
	}
	var out MyCompositeStructOne
	avMap := map[string]*dynamodb.AttributeValue{
		"a": myStructAVMap,
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := Unmarshal(&dynamodb.AttributeValue{M: avMap}, &out); err != nil {
				b.Error("unexpected error:", err)
			}
		}
	})
}

func BenchmarkUnmarshalTwoMembers(b *testing.B) {
	fieldCache = fieldCacher{}

	myStructAVMap, _ := Marshal(simpleMarshalStruct{
		String:  "abc",
		Int:     123,
		Uint:    123,
		Float32: 123.321,
		Float64: 123.321,
		Bool:    true,
		Null:    nil,
	})

	type MyCompositeStructTwo struct {
		A simpleMarshalStruct `dynamodbav:"a"`
		B simpleMarshalStruct `dynamodbav:"b"`
	}
	var out MyCompositeStructTwo
	avMap := map[string]*dynamodb.AttributeValue{
		"a": myStructAVMap,
		"b": myStructAVMap,
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := Unmarshal(&dynamodb.AttributeValue{M: avMap}, &out); err != nil {
				b.Error("unexpected error:", err)
			}
		}
	})
}

func Test_Encode_YAML_TagKey(t *testing.T) {
	input := struct {
		String      string         `yaml:"string"`
		EmptyString string         `yaml:"empty"`
		OmitString  string         `yaml:"omitted,omitempty"`
		Ignored     string         `yaml:"-"`
		Byte        []byte         `yaml:"byte"`
		Float32     float32        `yaml:"float32"`
		Float64     float64        `yaml:"float64"`
		Int         int            `yaml:"int"`
		Uint        uint           `yaml:"uint"`
		Slice       []string       `yaml:"slice"`
		Map         map[string]int `yaml:"map"`
		NoTag       string
	}{
		String:  "String",
		Ignored: "Ignored",
		Slice:   []string{"one", "two"},
		Map: map[string]int{
			"one": 1,
			"two": 2,
		},
		NoTag: "NoTag",
	}

	expected := &dynamodb.AttributeValue{
		M: map[string]*dynamodb.AttributeValue{
			"string":  {S: aws.String("String")},
			"empty":   {NULL: &trueValue},
			"byte":    {NULL: &trueValue},
			"float32": {N: aws.String("0")},
			"float64": {N: aws.String("0")},
			"int":     {N: aws.String("0")},
			"uint":    {N: aws.String("0")},
			"slice": {
				L: []*dynamodb.AttributeValue{
					{S: aws.String("one")},
					{S: aws.String("two")},
				},
			},
			"map": {
				M: map[string]*dynamodb.AttributeValue{
					"one": {N: aws.String("1")},
					"two": {N: aws.String("2")},
				},
			},
			"NoTag": {S: aws.String("NoTag")},
		},
	}

	enc := NewEncoder(func(e *Encoder) {
		e.TagKey = "yaml"
	})

	actual, err := enc.Encode(input)
	if err != nil {
		t.Errorf("Encode with input %#v retured error `%s`, expected nil", input, err)
	}

	compareObjects(t, expected, actual)
}
