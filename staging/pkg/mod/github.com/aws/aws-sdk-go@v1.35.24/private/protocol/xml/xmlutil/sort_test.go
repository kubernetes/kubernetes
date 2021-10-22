package xmlutil

import (
	"encoding/xml"
	"reflect"
	"sort"
	"testing"
)

func TestXmlAttrSlice(t *testing.T) {
	tests := []struct {
		input    []xml.Attr
		expected []xml.Attr
	}{
		{
			input:    []xml.Attr{},
			expected: []xml.Attr{},
		},
		{
			input: []xml.Attr{
				{
					Name: xml.Name{
						Space: "foo",
						Local: "bar",
					},
					Value: "baz",
				},
				{
					Name: xml.Name{
						Space: "foo",
						Local: "baz",
					},
					Value: "bar",
				},
				{
					Name: xml.Name{
						Space: "foo",
						Local: "bar",
					},
					Value: "bar",
				},
				{
					Name: xml.Name{
						Space: "baz",
						Local: "bar",
					},
					Value: "foo",
				},
			},
			expected: []xml.Attr{
				{
					Name: xml.Name{
						Space: "baz",
						Local: "bar",
					},
					Value: "foo",
				},
				{
					Name: xml.Name{
						Space: "foo",
						Local: "bar",
					},
					Value: "bar",
				},
				{
					Name: xml.Name{
						Space: "foo",
						Local: "bar",
					},
					Value: "baz",
				},
				{
					Name: xml.Name{
						Space: "foo",
						Local: "baz",
					},
					Value: "bar",
				},
			},
		},
	}
	for i, tt := range tests {
		sort.Sort(xmlAttrSlice(tt.input))
		if e, a := tt.expected, tt.input; !reflect.DeepEqual(e, a) {
			t.Errorf("case %d expected %v, got %v", i, e, a)
		}
	}
}
