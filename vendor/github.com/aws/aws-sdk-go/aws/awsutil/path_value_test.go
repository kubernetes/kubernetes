package awsutil_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws/awsutil"
	"github.com/stretchr/testify/assert"
)

type Struct struct {
	A []Struct
	z []Struct
	B *Struct
	D *Struct
	C string
	E map[string]string
}

var data = Struct{
	A: []Struct{{C: "value1"}, {C: "value2"}, {C: "value3"}},
	z: []Struct{{C: "value1"}, {C: "value2"}, {C: "value3"}},
	B: &Struct{B: &Struct{C: "terminal"}, D: &Struct{C: "terminal2"}},
	C: "initial",
}
var data2 = Struct{A: []Struct{
	{A: []Struct{{C: "1"}, {C: "1"}, {C: "1"}, {C: "1"}, {C: "1"}}},
	{A: []Struct{{C: "2"}, {C: "2"}, {C: "2"}, {C: "2"}, {C: "2"}}},
}}

func TestValueAtPathSuccess(t *testing.T) {
	var testCases = []struct {
		expect []interface{}
		data   interface{}
		path   string
	}{
		{[]interface{}{"initial"}, data, "C"},
		{[]interface{}{"value1"}, data, "A[0].C"},
		{[]interface{}{"value2"}, data, "A[1].C"},
		{[]interface{}{"value3"}, data, "A[2].C"},
		{[]interface{}{"value3"}, data, "a[2].c"},
		{[]interface{}{"value3"}, data, "A[-1].C"},
		{[]interface{}{"value1", "value2", "value3"}, data, "A[].C"},
		{[]interface{}{"terminal"}, data, "B . B . C"},
		{[]interface{}{"initial"}, data, "A.D.X || C"},
		{[]interface{}{"initial"}, data, "A[0].B || C"},
		{[]interface{}{
			Struct{A: []Struct{{C: "1"}, {C: "1"}, {C: "1"}, {C: "1"}, {C: "1"}}},
			Struct{A: []Struct{{C: "2"}, {C: "2"}, {C: "2"}, {C: "2"}, {C: "2"}}},
		}, data2, "A"},
	}
	for i, c := range testCases {
		v, err := awsutil.ValuesAtPath(c.data, c.path)
		assert.NoError(t, err, "case %d, expected no error, %s", i, c.path)
		assert.Equal(t, c.expect, v, "case %d, %s", i, c.path)
	}
}

func TestValueAtPathFailure(t *testing.T) {
	var testCases = []struct {
		expect      []interface{}
		errContains string
		data        interface{}
		path        string
	}{
		{nil, "", data, "C.x"},
		{nil, "SyntaxError: Invalid token: tDot", data, ".x"},
		{nil, "", data, "X.Y.Z"},
		{nil, "", data, "A[100].C"},
		{nil, "", data, "A[3].C"},
		{nil, "", data, "B.B.C.Z"},
		{nil, "", data, "z[-1].C"},
		{nil, "", nil, "A.B.C"},
		{[]interface{}{}, "", Struct{}, "A"},
		{nil, "", data, "A[0].B.C"},
		{nil, "", data, "D"},
	}

	for i, c := range testCases {
		v, err := awsutil.ValuesAtPath(c.data, c.path)
		if c.errContains != "" {
			assert.Contains(t, err.Error(), c.errContains, "case %d, expected error, %s", i, c.path)
			continue
		} else {
			assert.NoError(t, err, "case %d, expected no error, %s", i, c.path)
		}
		assert.Equal(t, c.expect, v, "case %d, %s", i, c.path)
	}
}

func TestSetValueAtPathSuccess(t *testing.T) {
	var s Struct
	awsutil.SetValueAtPath(&s, "C", "test1")
	awsutil.SetValueAtPath(&s, "B.B.C", "test2")
	awsutil.SetValueAtPath(&s, "B.D.C", "test3")
	assert.Equal(t, "test1", s.C)
	assert.Equal(t, "test2", s.B.B.C)
	assert.Equal(t, "test3", s.B.D.C)

	awsutil.SetValueAtPath(&s, "B.*.C", "test0")
	assert.Equal(t, "test0", s.B.B.C)
	assert.Equal(t, "test0", s.B.D.C)

	var s2 Struct
	awsutil.SetValueAtPath(&s2, "b.b.c", "test0")
	assert.Equal(t, "test0", s2.B.B.C)
	awsutil.SetValueAtPath(&s2, "A", []Struct{{}})
	assert.Equal(t, []Struct{{}}, s2.A)

	str := "foo"

	s3 := Struct{}
	awsutil.SetValueAtPath(&s3, "b.b.c", str)
	assert.Equal(t, "foo", s3.B.B.C)

	s3 = Struct{B: &Struct{B: &Struct{C: str}}}
	awsutil.SetValueAtPath(&s3, "b.b.c", nil)
	assert.Equal(t, "", s3.B.B.C)

	s3 = Struct{}
	awsutil.SetValueAtPath(&s3, "b.b.c", nil)
	assert.Equal(t, "", s3.B.B.C)

	s3 = Struct{}
	awsutil.SetValueAtPath(&s3, "b.b.c", &str)
	assert.Equal(t, "foo", s3.B.B.C)

	var s4 struct{ Name *string }
	awsutil.SetValueAtPath(&s4, "Name", str)
	assert.Equal(t, str, *s4.Name)

	s4 = struct{ Name *string }{}
	awsutil.SetValueAtPath(&s4, "Name", nil)
	assert.Equal(t, (*string)(nil), s4.Name)

	s4 = struct{ Name *string }{Name: &str}
	awsutil.SetValueAtPath(&s4, "Name", nil)
	assert.Equal(t, (*string)(nil), s4.Name)

	s4 = struct{ Name *string }{}
	awsutil.SetValueAtPath(&s4, "Name", &str)
	assert.Equal(t, str, *s4.Name)
}
