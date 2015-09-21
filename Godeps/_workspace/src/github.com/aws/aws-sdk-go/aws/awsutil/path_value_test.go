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
}

var data = Struct{
	A: []Struct{Struct{C: "value1"}, Struct{C: "value2"}, Struct{C: "value3"}},
	z: []Struct{Struct{C: "value1"}, Struct{C: "value2"}, Struct{C: "value3"}},
	B: &Struct{B: &Struct{C: "terminal"}, D: &Struct{C: "terminal2"}},
	C: "initial",
}

func TestValueAtPathSuccess(t *testing.T) {
	assert.Equal(t, []interface{}{"initial"}, awsutil.ValuesAtPath(data, "C"))
	assert.Equal(t, []interface{}{"value1"}, awsutil.ValuesAtPath(data, "A[0].C"))
	assert.Equal(t, []interface{}{"value2"}, awsutil.ValuesAtPath(data, "A[1].C"))
	assert.Equal(t, []interface{}{"value3"}, awsutil.ValuesAtPath(data, "A[2].C"))
	assert.Equal(t, []interface{}{"value3"}, awsutil.ValuesAtAnyPath(data, "a[2].c"))
	assert.Equal(t, []interface{}{"value3"}, awsutil.ValuesAtPath(data, "A[-1].C"))
	assert.Equal(t, []interface{}{"value1", "value2", "value3"}, awsutil.ValuesAtPath(data, "A[].C"))
	assert.Equal(t, []interface{}{"terminal"}, awsutil.ValuesAtPath(data, "B . B . C"))
	assert.Equal(t, []interface{}{"terminal", "terminal2"}, awsutil.ValuesAtPath(data, "B.*.C"))
	assert.Equal(t, []interface{}{"initial"}, awsutil.ValuesAtPath(data, "A.D.X || C"))
}

func TestValueAtPathFailure(t *testing.T) {
	assert.Equal(t, []interface{}(nil), awsutil.ValuesAtPath(data, "C.x"))
	assert.Equal(t, []interface{}(nil), awsutil.ValuesAtPath(data, ".x"))
	assert.Equal(t, []interface{}{}, awsutil.ValuesAtPath(data, "X.Y.Z"))
	assert.Equal(t, []interface{}{}, awsutil.ValuesAtPath(data, "A[100].C"))
	assert.Equal(t, []interface{}{}, awsutil.ValuesAtPath(data, "A[3].C"))
	assert.Equal(t, []interface{}{}, awsutil.ValuesAtPath(data, "B.B.C.Z"))
	assert.Equal(t, []interface{}(nil), awsutil.ValuesAtPath(data, "z[-1].C"))
	assert.Equal(t, []interface{}{}, awsutil.ValuesAtPath(nil, "A.B.C"))
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
	awsutil.SetValueAtAnyPath(&s2, "b.b.c", "test0")
	assert.Equal(t, "test0", s2.B.B.C)
}
