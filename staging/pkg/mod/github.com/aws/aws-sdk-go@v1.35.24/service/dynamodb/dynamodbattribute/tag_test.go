package dynamodbattribute

import (
	"reflect"
	"testing"
)

func TestTagParse(t *testing.T) {
	cases := []struct {
		in       reflect.StructTag
		json, av bool
		expect   tag
	}{
		{`json:""`, true, false, tag{}},
		{`json:"name"`, true, false, tag{Name: "name"}},
		{`json:"name,omitempty"`, true, false, tag{Name: "name", OmitEmpty: true}},
		{`json:"-"`, true, false, tag{Ignore: true}},
		{`json:",omitempty"`, true, false, tag{OmitEmpty: true}},
		{`json:",string"`, true, false, tag{AsString: true}},
		{`dynamodbav:""`, false, true, tag{}},
		{`dynamodbav:","`, false, true, tag{}},
		{`dynamodbav:"name"`, false, true, tag{Name: "name"}},
		{`dynamodbav:"name"`, false, true, tag{Name: "name"}},
		{`dynamodbav:"-"`, false, true, tag{Ignore: true}},
		{`dynamodbav:",omitempty"`, false, true, tag{OmitEmpty: true}},
		{`dynamodbav:",omitemptyelem"`, false, true, tag{OmitEmptyElem: true}},
		{`dynamodbav:",string"`, false, true, tag{AsString: true}},
		{`dynamodbav:",binaryset"`, false, true, tag{AsBinSet: true}},
		{`dynamodbav:",numberset"`, false, true, tag{AsNumSet: true}},
		{`dynamodbav:",stringset"`, false, true, tag{AsStrSet: true}},
		{`dynamodbav:",stringset,omitemptyelem"`, false, true, tag{AsStrSet: true, OmitEmptyElem: true}},
		{`dynamodbav:"name,stringset,omitemptyelem"`, false, true, tag{Name: "name", AsStrSet: true, OmitEmptyElem: true}},
	}

	for i, c := range cases {
		actual := tag{}
		if c.json {
			actual.parseStructTag("json", c.in)
		}
		if c.av {
			actual.parseAVTag(c.in)
		}
		if e, a := c.expect, actual; !reflect.DeepEqual(e, a) {
			t.Errorf("case %d, expect %v, got %v", i, e, a)
		}
	}
}
