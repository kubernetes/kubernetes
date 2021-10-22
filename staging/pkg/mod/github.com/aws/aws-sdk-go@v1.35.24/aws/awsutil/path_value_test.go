package awsutil_test

import (
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awsutil"
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
		if err != nil {
			t.Errorf("case %v, expected no error, %v", i, c.path)
		}
		if e, a := c.expect, v; !awsutil.DeepEqual(e, a) {
			t.Errorf("case %v, %v", i, c.path)
		}
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
			if !strings.Contains(err.Error(), c.errContains) {
				t.Errorf("case %v, expected error, %v", i, c.path)
			}
			continue
		} else {
			if err != nil {
				t.Errorf("case %v, expected no error, %v", i, c.path)
			}
		}
		if e, a := c.expect, v; !awsutil.DeepEqual(e, a) {
			t.Errorf("case %v, %v", i, c.path)
		}
	}
}

func TestSetValueAtPathSuccess(t *testing.T) {
	var s Struct
	awsutil.SetValueAtPath(&s, "C", "test1")
	awsutil.SetValueAtPath(&s, "B.B.C", "test2")
	awsutil.SetValueAtPath(&s, "B.D.C", "test3")
	if e, a := "test1", s.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}
	if e, a := "test2", s.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}
	if e, a := "test3", s.B.D.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	awsutil.SetValueAtPath(&s, "B.*.C", "test0")
	if e, a := "test0", s.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}
	if e, a := "test0", s.B.D.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	var s2 Struct
	awsutil.SetValueAtPath(&s2, "b.b.c", "test0")
	if e, a := "test0", s2.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}
	awsutil.SetValueAtPath(&s2, "A", []Struct{{}})
	if e, a := []Struct{{}}, s2.A; !awsutil.DeepEqual(e, a) {
		t.Errorf("expected %v, but received %v", e, a)
	}

	str := "foo"

	s3 := Struct{}
	awsutil.SetValueAtPath(&s3, "b.b.c", str)
	if e, a := "foo", s3.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	s3 = Struct{B: &Struct{B: &Struct{C: str}}}
	awsutil.SetValueAtPath(&s3, "b.b.c", nil)
	if e, a := "", s3.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	s3 = Struct{}
	awsutil.SetValueAtPath(&s3, "b.b.c", nil)
	if e, a := "", s3.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	s3 = Struct{}
	awsutil.SetValueAtPath(&s3, "b.b.c", &str)
	if e, a := "foo", s3.B.B.C; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	var s4 struct{ Name *string }
	awsutil.SetValueAtPath(&s4, "Name", str)
	if e, a := str, *s4.Name; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	s4 = struct{ Name *string }{}
	awsutil.SetValueAtPath(&s4, "Name", nil)
	if e, a := (*string)(nil), s4.Name; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	s4 = struct{ Name *string }{Name: &str}
	awsutil.SetValueAtPath(&s4, "Name", nil)
	if e, a := (*string)(nil), s4.Name; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}

	s4 = struct{ Name *string }{}
	awsutil.SetValueAtPath(&s4, "Name", &str)
	if e, a := str, *s4.Name; e != a {
		t.Errorf("expected %v, but received %v", e, a)
	}
}
