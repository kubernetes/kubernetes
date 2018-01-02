package dynamodbattribute

import (
	"reflect"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/dynamodb"
)

type testBinarySetStruct struct {
	Binarys [][]byte `dynamodbav:",binaryset"`
}
type testNumberSetStruct struct {
	Numbers []int `dynamodbav:",numberset"`
}
type testStringSetStruct struct {
	Strings []string `dynamodbav:",stringset"`
}

type testIntAsStringStruct struct {
	Value int `dynamodbav:",string"`
}

type testOmitEmptyStruct struct {
	Value  string  `dynamodbav:",omitempty"`
	Value2 *string `dynamodbav:",omitempty"`
	Value3 int
}

type testAliasedString string
type testAliasedStringSlice []string
type testAliasedInt int
type testAliasedIntSlice []int
type testAliasedMap map[string]int
type testAliasedSlice []string
type testAliasedByteSlice []byte
type testAliasedBool bool
type testAliasedBoolSlice []bool

type testAliasedStruct struct {
	Value  testAliasedString
	Value2 testAliasedInt
	Value3 testAliasedMap
	Value4 testAliasedSlice

	Value5 testAliasedByteSlice
	Value6 []testAliasedInt
	Value7 []testAliasedString

	Value8  []testAliasedByteSlice `dynamodbav:",binaryset"`
	Value9  []testAliasedInt       `dynamodbav:",numberset"`
	Value10 []testAliasedString    `dynamodbav:",stringset"`

	Value11 testAliasedIntSlice
	Value12 testAliasedStringSlice

	Value13 testAliasedBool
	Value14 testAliasedBoolSlice
}

type testNamedPointer *int

var testDate, _ = time.Parse(time.RFC3339, "2016-05-03T17:06:26.209072Z")

var sharedTestCases = []struct {
	in               *dynamodb.AttributeValue
	actual, expected interface{}
	err              error
}{
	{ // Binary slice
		in:       &dynamodb.AttributeValue{B: []byte{48, 49}},
		actual:   &[]byte{},
		expected: []byte{48, 49},
	},
	{ // Binary slice
		in:       &dynamodb.AttributeValue{B: []byte{48, 49}},
		actual:   &[]byte{},
		expected: []byte{48, 49},
	},
	{ // Binary slice oversized
		in: &dynamodb.AttributeValue{B: []byte{48, 49}},
		actual: func() *[]byte {
			v := make([]byte, 0, 10)
			return &v
		}(),
		expected: []byte{48, 49},
	},
	{ // Binary slice pointer
		in: &dynamodb.AttributeValue{B: []byte{48, 49}},
		actual: func() **[]byte {
			v := make([]byte, 0, 10)
			v2 := &v
			return &v2
		}(),
		expected: []byte{48, 49},
	},
	{ // Bool
		in:       &dynamodb.AttributeValue{BOOL: aws.Bool(true)},
		actual:   new(bool),
		expected: true,
	},
	{ // List
		in: &dynamodb.AttributeValue{L: []*dynamodb.AttributeValue{
			{N: aws.String("123")},
		}},
		actual:   &[]int{},
		expected: []int{123},
	},
	{ // Map, interface
		in: &dynamodb.AttributeValue{M: map[string]*dynamodb.AttributeValue{
			"abc": {N: aws.String("123")},
		}},
		actual:   &map[string]int{},
		expected: map[string]int{"abc": 123},
	},
	{ // Map, struct
		in: &dynamodb.AttributeValue{M: map[string]*dynamodb.AttributeValue{
			"Abc": {N: aws.String("123")},
		}},
		actual:   &struct{ Abc int }{},
		expected: struct{ Abc int }{Abc: 123},
	},
	{ // Map, struct
		in: &dynamodb.AttributeValue{M: map[string]*dynamodb.AttributeValue{
			"abc": {N: aws.String("123")},
		}},
		actual: &struct {
			Abc int `json:"abc" dynamodbav:"abc"`
		}{},
		expected: struct {
			Abc int `json:"abc" dynamodbav:"abc"`
		}{Abc: 123},
	},
	{ // Number, int
		in:       &dynamodb.AttributeValue{N: aws.String("123")},
		actual:   new(int),
		expected: 123,
	},
	{ // Number, Float
		in:       &dynamodb.AttributeValue{N: aws.String("123.1")},
		actual:   new(float64),
		expected: float64(123.1),
	},
	{ // Null
		in:       &dynamodb.AttributeValue{NULL: aws.Bool(true)},
		actual:   new(string),
		expected: "",
	},
	{ // Null ptr
		in:       &dynamodb.AttributeValue{NULL: aws.Bool(true)},
		actual:   new(*string),
		expected: nil,
	},
	{ // String
		in:       &dynamodb.AttributeValue{S: aws.String("abc")},
		actual:   new(string),
		expected: "abc",
	},
	{ // Binary Set
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Binarys": {BS: [][]byte{{48, 49}, {50, 51}}},
			},
		},
		actual:   &testBinarySetStruct{},
		expected: testBinarySetStruct{Binarys: [][]byte{{48, 49}, {50, 51}}},
	},
	{ // Number Set
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Numbers": {NS: []*string{aws.String("123"), aws.String("321")}},
			},
		},
		actual:   &testNumberSetStruct{},
		expected: testNumberSetStruct{Numbers: []int{123, 321}},
	},
	{ // String Set
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Strings": {SS: []*string{aws.String("abc"), aws.String("efg")}},
			},
		},
		actual:   &testStringSetStruct{},
		expected: testStringSetStruct{Strings: []string{"abc", "efg"}},
	},
	{ // Int value as string
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Value": {S: aws.String("123")},
			},
		},
		actual:   &testIntAsStringStruct{},
		expected: testIntAsStringStruct{Value: 123},
	},
	{ // Omitempty
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Value3": {N: aws.String("0")},
			},
		},
		actual:   &testOmitEmptyStruct{},
		expected: testOmitEmptyStruct{Value: "", Value2: nil, Value3: 0},
	},
	{ // aliased type
		in: &dynamodb.AttributeValue{
			M: map[string]*dynamodb.AttributeValue{
				"Value":  {S: aws.String("123")},
				"Value2": {N: aws.String("123")},
				"Value3": {M: map[string]*dynamodb.AttributeValue{
					"Key": {N: aws.String("321")},
				}},
				"Value4": {L: []*dynamodb.AttributeValue{
					{S: aws.String("1")},
					{S: aws.String("2")},
					{S: aws.String("3")},
				}},
				"Value5": {B: []byte{0, 1, 2}},
				"Value6": {L: []*dynamodb.AttributeValue{
					{N: aws.String("1")},
					{N: aws.String("2")},
					{N: aws.String("3")},
				}},
				"Value7": {L: []*dynamodb.AttributeValue{
					{S: aws.String("1")},
					{S: aws.String("2")},
					{S: aws.String("3")},
				}},
				"Value8": {BS: [][]byte{
					{0, 1, 2}, {3, 4, 5},
				}},
				"Value9": {NS: []*string{
					aws.String("1"),
					aws.String("2"),
					aws.String("3"),
				}},
				"Value10": {SS: []*string{
					aws.String("1"),
					aws.String("2"),
					aws.String("3"),
				}},
				"Value11": {L: []*dynamodb.AttributeValue{
					{N: aws.String("1")},
					{N: aws.String("2")},
					{N: aws.String("3")},
				}},
				"Value12": {L: []*dynamodb.AttributeValue{
					{S: aws.String("1")},
					{S: aws.String("2")},
					{S: aws.String("3")},
				}},
				"Value13": {BOOL: aws.Bool(true)},
				"Value14": {L: []*dynamodb.AttributeValue{
					{BOOL: aws.Bool(true)},
					{BOOL: aws.Bool(false)},
					{BOOL: aws.Bool(true)},
				}},
			},
		},
		actual: &testAliasedStruct{},
		expected: testAliasedStruct{
			Value: "123", Value2: 123,
			Value3: testAliasedMap{
				"Key": 321,
			},
			Value4: testAliasedSlice{"1", "2", "3"},
			Value5: testAliasedByteSlice{0, 1, 2},
			Value6: []testAliasedInt{1, 2, 3},
			Value7: []testAliasedString{"1", "2", "3"},
			Value8: []testAliasedByteSlice{
				{0, 1, 2},
				{3, 4, 5},
			},
			Value9:  []testAliasedInt{1, 2, 3},
			Value10: []testAliasedString{"1", "2", "3"},
			Value11: testAliasedIntSlice{1, 2, 3},
			Value12: testAliasedStringSlice{"1", "2", "3"},
			Value13: true,
			Value14: testAliasedBoolSlice{true, false, true},
		},
	},
	{
		in:       &dynamodb.AttributeValue{N: aws.String("123")},
		actual:   new(testNamedPointer),
		expected: testNamedPointer(aws.Int(123)),
	},
	{ // time.Time
		in:       &dynamodb.AttributeValue{S: aws.String("2016-05-03T17:06:26.209072Z")},
		actual:   new(time.Time),
		expected: testDate,
	},
	{ // time.Time List
		in: &dynamodb.AttributeValue{L: []*dynamodb.AttributeValue{
			{S: aws.String("2016-05-03T17:06:26.209072Z")},
			{S: aws.String("2016-05-04T17:06:26.209072Z")},
		}},
		actual:   new([]time.Time),
		expected: []time.Time{testDate, testDate.Add(24 * time.Hour)},
	},
	{ // time.Time struct
		in: &dynamodb.AttributeValue{M: map[string]*dynamodb.AttributeValue{
			"abc": {S: aws.String("2016-05-03T17:06:26.209072Z")},
		}},
		actual: &struct {
			Abc time.Time `json:"abc" dynamodbav:"abc"`
		}{},
		expected: struct {
			Abc time.Time `json:"abc" dynamodbav:"abc"`
		}{Abc: testDate},
	},
	{ // time.Time ptr struct
		in: &dynamodb.AttributeValue{M: map[string]*dynamodb.AttributeValue{
			"abc": {S: aws.String("2016-05-03T17:06:26.209072Z")},
		}},
		actual: &struct {
			Abc *time.Time `json:"abc" dynamodbav:"abc"`
		}{},
		expected: struct {
			Abc *time.Time `json:"abc" dynamodbav:"abc"`
		}{Abc: &testDate},
	},
}

var sharedListTestCases = []struct {
	in               []*dynamodb.AttributeValue
	actual, expected interface{}
	err              error
}{
	{
		in: []*dynamodb.AttributeValue{
			{B: []byte{48, 49}},
			{BOOL: aws.Bool(true)},
			{N: aws.String("123")},
			{S: aws.String("123")},
		},
		actual: func() *[]interface{} {
			v := []interface{}{}
			return &v
		}(),
		expected: []interface{}{[]byte{48, 49}, true, 123., "123"},
	},
	{
		in: []*dynamodb.AttributeValue{
			{N: aws.String("1")},
			{N: aws.String("2")},
			{N: aws.String("3")},
		},
		actual:   &[]interface{}{},
		expected: []interface{}{1., 2., 3.},
	},
}

var sharedMapTestCases = []struct {
	in               map[string]*dynamodb.AttributeValue
	actual, expected interface{}
	err              error
}{
	{
		in: map[string]*dynamodb.AttributeValue{
			"B":    {B: []byte{48, 49}},
			"BOOL": {BOOL: aws.Bool(true)},
			"N":    {N: aws.String("123")},
			"S":    {S: aws.String("123")},
		},
		actual: &map[string]interface{}{},
		expected: map[string]interface{}{
			"B": []byte{48, 49}, "BOOL": true,
			"N": 123., "S": "123",
		},
	},
}

func assertConvertTest(t *testing.T, i int, actual, expected interface{}, err, expectedErr error) {
	i++
	if expectedErr != nil {
		if err != nil {
			if e, a := expectedErr, err; !reflect.DeepEqual(e, a) {
				t.Errorf("case %d expect %v, got %v", i, e, a)
			}
		} else {
			t.Fatalf("case %d, expected error, %v", i, expectedErr)
		}
	} else if err != nil {
		t.Fatalf("case %d, expect no error, got %v", i, err)
	} else {
		if e, a := ptrToValue(expected), ptrToValue(actual); !reflect.DeepEqual(e, a) {
			t.Errorf("case %d, expect %v, got %v", i, e, a)
		}
	}
}

func ptrToValue(in interface{}) interface{} {
	v := reflect.ValueOf(in)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if !v.IsValid() {
		return nil
	}
	if v.Kind() == reflect.Ptr {
		return ptrToValue(v.Interface())
	}
	return v.Interface()
}
