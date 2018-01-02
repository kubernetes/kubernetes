package dynamodbattribute

import (
	"reflect"
	"testing"
)

type testUnionValues struct {
	Name  string
	Value interface{}
}

type unionSimple struct {
	A int
	B string
	C []string
}

type unionComplex struct {
	unionSimple
	A int
}

type unionTagged struct {
	A int `json:"A"`
}

type unionTaggedComplex struct {
	unionSimple
	unionTagged
	B string
}

func TestUnionStructFields(t *testing.T) {
	var cases = []struct {
		in     interface{}
		expect []testUnionValues
	}{
		{
			in: unionSimple{1, "2", []string{"abc"}},
			expect: []testUnionValues{
				{"A", 1},
				{"B", "2"},
				{"C", []string{"abc"}},
			},
		},
		{
			in: unionComplex{
				unionSimple: unionSimple{1, "2", []string{"abc"}},
				A:           2,
			},
			expect: []testUnionValues{
				{"B", "2"},
				{"C", []string{"abc"}},
				{"A", 2},
			},
		},
		{
			in: unionTaggedComplex{
				unionSimple: unionSimple{1, "2", []string{"abc"}},
				unionTagged: unionTagged{3},
				B:           "3",
			},
			expect: []testUnionValues{
				{"C", []string{"abc"}},
				{"A", 3},
				{"B", "3"},
			},
		},
	}

	for i, c := range cases {
		v := reflect.ValueOf(c.in)

		fields := unionStructFields(v.Type(), MarshalOptions{SupportJSONTags: true})
		for j, f := range fields {
			expected := c.expect[j]
			if e, a := expected.Name, f.Name; e != a {
				t.Errorf("%d:%d expect %v, got %v", i, j, e, f)
			}
			actual := v.FieldByIndex(f.Index).Interface()
			if e, a := expected.Value, actual; !reflect.DeepEqual(e, a) {
				t.Errorf("%d:%d expect %v, got %v", i, j, e, f)
			}
		}
	}
}

func TestFieldByName(t *testing.T) {
	fields := []field{
		{Name: "Abc"}, {Name: "mixCase"}, {Name: "UPPERCASE"},
	}

	cases := []struct {
		Name, FieldName string
		Found           bool
	}{
		{"abc", "Abc", true}, {"ABC", "Abc", true}, {"Abc", "Abc", true},
		{"123", "", false},
		{"ab", "", false},
		{"MixCase", "mixCase", true},
		{"uppercase", "UPPERCASE", true}, {"UPPERCASE", "UPPERCASE", true},
	}

	for _, c := range cases {
		f, ok := fieldByName(fields, c.Name)
		if e, a := c.Found, ok; e != a {
			t.Errorf("expect %v, got %v", e, a)
		}
		if ok {
			if e, a := c.FieldName, f.Name; e != a {
				t.Errorf("expect %v, got %v", e, a)
			}
		}
	}
}
