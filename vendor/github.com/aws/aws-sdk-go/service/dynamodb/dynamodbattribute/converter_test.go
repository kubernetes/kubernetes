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

type mySimpleStruct struct {
	String  string
	Int     int
	Uint    uint
	Float32 float32
	Float64 float64
	Bool    bool
	Null    *interface{}
}

type myComplexStruct struct {
	Simple []mySimpleStruct
}

type converterTestInput struct {
	input     interface{}
	expected  interface{}
	err       awserr.Error
	inputType string // "enum" of types
}

var trueValue = true
var falseValue = false

var converterScalarInputs = []converterTestInput{
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
		input: mySimpleStruct{},
		expected: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Bool":    {BOOL: &falseValue},
				"Float32": {N: aws.String("0")},
				"Float64": {N: aws.String("0")},
				"Int":     {N: aws.String("0")},
				"Null":    {NULL: &trueValue},
				"String":  {S: aws.String("")},
				"Uint":    {N: aws.String("0")},
			},
		},
		inputType: "mySimpleStruct",
	},
}

var converterMapTestInputs = []converterTestInput{
	// Scalar tests
	{
		input: nil,
		err:   awserr.New("SerializationError", "in must be a map[string]interface{} or struct, got <nil>", nil),
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
		input:    map[string]interface{}{"int": int(12)},
		expected: map[string]*dynamodb.AttributeValue{"int": {N: aws.String("12")}},
	},
	// List
	{
		input: map[string]interface{}{"list": []interface{}{"a string", 12, 3.14, true, nil, false}},
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
		input: map[string]interface{}{"map": map[string]interface{}{"nestedint": 12}},
		expected: map[string]*dynamodb.AttributeValue{
			"map": {
				M: map[string]*dynamodb.AttributeValue{
					"nestedint": {
						N: aws.String("12"),
					},
				},
			},
		},
	},
	// Structs
	{
		input: mySimpleStruct{},
		expected: map[string]*dynamodb.AttributeValue{
			"Bool":    {BOOL: &falseValue},
			"Float32": {N: aws.String("0")},
			"Float64": {N: aws.String("0")},
			"Int":     {N: aws.String("0")},
			"Null":    {NULL: &trueValue},
			"String":  {S: aws.String("")},
			"Uint":    {N: aws.String("0")},
		},
		inputType: "mySimpleStruct",
	},
	{
		input: myComplexStruct{},
		expected: map[string]*dynamodb.AttributeValue{
			"Simple": {NULL: &trueValue},
		},
		inputType: "myComplexStruct",
	},
	{
		input: myComplexStruct{Simple: []mySimpleStruct{{Int: -2}, {Uint: 5}}},
		expected: map[string]*dynamodb.AttributeValue{
			"Simple": {
				L: []*dynamodb.AttributeValue{
					{
						M: map[string]*dynamodb.AttributeValue{
							"Bool":    {BOOL: &falseValue},
							"Float32": {N: aws.String("0")},
							"Float64": {N: aws.String("0")},
							"Int":     {N: aws.String("-2")},
							"Null":    {NULL: &trueValue},
							"String":  {S: aws.String("")},
							"Uint":    {N: aws.String("0")},
						},
					},
					{
						M: map[string]*dynamodb.AttributeValue{
							"Bool":    {BOOL: &falseValue},
							"Float32": {N: aws.String("0")},
							"Float64": {N: aws.String("0")},
							"Int":     {N: aws.String("0")},
							"Null":    {NULL: &trueValue},
							"String":  {S: aws.String("")},
							"Uint":    {N: aws.String("5")},
						},
					},
				},
			},
		},
		inputType: "myComplexStruct",
	},
}

var converterListTestInputs = []converterTestInput{
	{
		input: nil,
		err:   awserr.New("SerializationError", "in must be an array or slice, got <nil>", nil),
	},
	{
		input:    []interface{}{},
		expected: []*dynamodb.AttributeValue{},
	},
	{
		input: []interface{}{"a string", 12, 3.14, true, nil, false},
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
		input: []mySimpleStruct{{}},
		expected: []*dynamodb.AttributeValue{
			{
				M: map[string]*dynamodb.AttributeValue{
					"Bool":    {BOOL: &falseValue},
					"Float32": {N: aws.String("0")},
					"Float64": {N: aws.String("0")},
					"Int":     {N: aws.String("0")},
					"Null":    {NULL: &trueValue},
					"String":  {S: aws.String("")},
					"Uint":    {N: aws.String("0")},
				},
			},
		},
		inputType: "mySimpleStruct",
	},
}

func TestConvertTo(t *testing.T) {
	for _, test := range converterScalarInputs {
		testConvertTo(t, test)
	}
}

func testConvertTo(t *testing.T, test converterTestInput) {
	actual, err := ConvertTo(test.input)
	if test.err != nil {
		if err == nil {
			t.Errorf("ConvertTo with input %#v retured %#v, expected error `%s`", test.input, actual, test.err)
		} else if err.Error() != test.err.Error() {
			t.Errorf("ConvertTo with input %#v retured error `%s`, expected error `%s`", test.input, err, test.err)
		}
	} else {
		if err != nil {
			t.Errorf("ConvertTo with input %#v retured error `%s`", test.input, err)
		}
		compareObjects(t, test.expected, actual)
	}
}

func TestConvertFrom(t *testing.T) {
	// Using the same inputs from TestConvertTo, test the reverse mapping.
	for _, test := range converterScalarInputs {
		if test.expected != nil {
			testConvertFrom(t, test)
		}
	}
}

func testConvertFrom(t *testing.T, test converterTestInput) {
	switch test.inputType {
	case "mySimpleStruct":
		var actual mySimpleStruct
		if err := ConvertFrom(test.expected.(*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFrom with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	case "myComplexStruct":
		var actual myComplexStruct
		if err := ConvertFrom(test.expected.(*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFrom with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	default:
		var actual interface{}
		if err := ConvertFrom(test.expected.(*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFrom with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	}
}

func TestConvertFromError(t *testing.T) {
	// Test that we get an error using ConvertFrom to convert to a map.
	var actual map[string]interface{}
	expected := awserr.New("SerializationError", `v must be a non-nil pointer to an interface{} or struct, got *map[string]interface {}`, nil).Error()
	if err := ConvertFrom(nil, &actual); err == nil {
		t.Errorf("ConvertFrom with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFrom with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}

	// Test that we get an error using ConvertFrom to convert to a list.
	var actual2 []interface{}
	expected = awserr.New("SerializationError", `v must be a non-nil pointer to an interface{} or struct, got *[]interface {}`, nil).Error()
	if err := ConvertFrom(nil, &actual2); err == nil {
		t.Errorf("ConvertFrom with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFrom with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}
}

func TestConvertToMap(t *testing.T) {
	for _, test := range converterMapTestInputs {
		testConvertToMap(t, test)
	}
}

func testConvertToMap(t *testing.T, test converterTestInput) {
	actual, err := ConvertToMap(test.input)
	if test.err != nil {
		if err == nil {
			t.Errorf("ConvertToMap with input %#v retured %#v, expected error `%s`", test.input, actual, test.err)
		} else if err.Error() != test.err.Error() {
			t.Errorf("ConvertToMap with input %#v retured error `%s`, expected error `%s`", test.input, err, test.err)
		}
	} else {
		if err != nil {
			t.Errorf("ConvertToMap with input %#v retured error `%s`", test.input, err)
		}
		compareObjects(t, test.expected, actual)
	}
}

func TestConvertFromMap(t *testing.T) {
	// Using the same inputs from TestConvertToMap, test the reverse mapping.
	for _, test := range converterMapTestInputs {
		if test.expected != nil {
			testConvertFromMap(t, test)
		}
	}
}

func testConvertFromMap(t *testing.T, test converterTestInput) {
	switch test.inputType {
	case "mySimpleStruct":
		var actual mySimpleStruct
		if err := ConvertFromMap(test.expected.(map[string]*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFromMap with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	case "myComplexStruct":
		var actual myComplexStruct
		if err := ConvertFromMap(test.expected.(map[string]*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFromMap with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	default:
		var actual map[string]interface{}
		if err := ConvertFromMap(test.expected.(map[string]*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFromMap with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	}
}

func TestConvertFromMapError(t *testing.T) {
	// Test that we get an error using ConvertFromMap to convert to an interface{}.
	var actual interface{}
	expected := awserr.New("SerializationError", `v must be a non-nil pointer to a map[string]interface{} or struct, got *interface {}`, nil).Error()
	if err := ConvertFromMap(nil, &actual); err == nil {
		t.Errorf("ConvertFromMap with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFromMap with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}

	// Test that we get an error using ConvertFromMap to convert to a slice.
	var actual2 []interface{}
	expected = awserr.New("SerializationError", `v must be a non-nil pointer to a map[string]interface{} or struct, got *[]interface {}`, nil).Error()
	if err := ConvertFromMap(nil, &actual2); err == nil {
		t.Errorf("ConvertFromMap with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFromMap with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}
}

func TestConvertToList(t *testing.T) {
	for _, test := range converterListTestInputs {
		testConvertToList(t, test)
	}
}

func testConvertToList(t *testing.T, test converterTestInput) {
	actual, err := ConvertToList(test.input)
	if test.err != nil {
		if err == nil {
			t.Errorf("ConvertToList with input %#v retured %#v, expected error `%s`", test.input, actual, test.err)
		} else if err.Error() != test.err.Error() {
			t.Errorf("ConvertToList with input %#v retured error `%s`, expected error `%s`", test.input, err, test.err)
		}
	} else {
		if err != nil {
			t.Errorf("ConvertToList with input %#v retured error `%s`", test.input, err)
		}
		compareObjects(t, test.expected, actual)
	}
}

func TestConvertFromList(t *testing.T) {
	// Using the same inputs from TestConvertToList, test the reverse mapping.
	for _, test := range converterListTestInputs {
		if test.expected != nil {
			testConvertFromList(t, test)
		}
	}
}

func testConvertFromList(t *testing.T, test converterTestInput) {
	switch test.inputType {
	case "mySimpleStruct":
		var actual []mySimpleStruct
		if err := ConvertFromList(test.expected.([]*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFromList with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	case "myComplexStruct":
		var actual []myComplexStruct
		if err := ConvertFromList(test.expected.([]*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFromList with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	default:
		var actual []interface{}
		if err := ConvertFromList(test.expected.([]*dynamodb.AttributeValue), &actual); err != nil {
			t.Errorf("ConvertFromList with input %#v retured error `%s`", test.expected, err)
		}
		compareObjects(t, test.input, actual)
	}
}

func TestConvertFromListError(t *testing.T) {
	// Test that we get an error using ConvertFromList to convert to a map.
	var actual map[string]interface{}
	expected := awserr.New("SerializationError", `v must be a non-nil pointer to an array or slice, got *map[string]interface {}`, nil).Error()
	if err := ConvertFromList(nil, &actual); err == nil {
		t.Errorf("ConvertFromList with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFromList with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}

	// Test that we get an error using ConvertFromList to convert to a struct.
	var actual2 myComplexStruct
	expected = awserr.New("SerializationError", `v must be a non-nil pointer to an array or slice, got *dynamodbattribute.myComplexStruct`, nil).Error()
	if err := ConvertFromList(nil, &actual2); err == nil {
		t.Errorf("ConvertFromList with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFromList with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}

	// Test that we get an error using ConvertFromList to convert to an interface{}.
	var actual3 interface{}
	expected = awserr.New("SerializationError", `v must be a non-nil pointer to an array or slice, got *interface {}`, nil).Error()
	if err := ConvertFromList(nil, &actual3); err == nil {
		t.Errorf("ConvertFromList with input %#v returned no error, expected error `%s`", nil, expected)
	} else if err.Error() != expected {
		t.Errorf("ConvertFromList with input %#v returned error `%s`, expected error `%s`", nil, err, expected)
	}
}

func compareObjects(t *testing.T, expected interface{}, actual interface{}) {
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("\nExpected %s:\n%s\nActual %s:\n%s\n",
			reflect.ValueOf(expected).Kind(),
			awsutil.Prettify(expected),
			reflect.ValueOf(actual).Kind(),
			awsutil.Prettify(actual))
	}
}
