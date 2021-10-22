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
		for j, f := range fields.All() {
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

func TestCachedFields(t *testing.T) {
	type myStruct struct {
		Dog  int
		CAT  string
		bird bool
	}

	fields := unionStructFields(reflect.TypeOf(myStruct{}), MarshalOptions{})

	const expectedNumFields = 2
	if numFields := len(fields.All()); numFields != expectedNumFields {
		t.Errorf("expected number of fields to be %d but got %d", expectedNumFields, numFields)
	}

	cases := []struct {
		Name      string
		FieldName string
		Found     bool
	}{
		{"Dog", "Dog", true},
		{"dog", "Dog", true},
		{"DOG", "Dog", true},
		{"Yorkie", "", false},
		{"Cat", "CAT", true},
		{"cat", "CAT", true},
		{"CAT", "CAT", true},
		{"tiger", "", false},
		{"bird", "", false},
	}

	for _, c := range cases {
		f, found := fields.FieldByName(c.Name)
		if found != c.Found {
			t.Errorf("expected found to be %v but got %v", c.Found, found)
		}
		if found && f.Name != c.FieldName {
			t.Errorf("expected field name to be %s but got %s", c.FieldName, f.Name)
		}
	}
}
