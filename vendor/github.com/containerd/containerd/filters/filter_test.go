package filters

import (
	"reflect"
	"strings"
	"testing"
)

func TestFilters(t *testing.T) {
	type cEntry struct {
		Name   string
		Other  string
		Labels map[string]string
	}

	corpusS := []cEntry{
		{
			Name: "foo",
			Labels: map[string]string{
				"foo": "true",
			},
		},
		{
			Name: "bar",
		},
		{
			Name: "foo",
			Labels: map[string]string{
				"foo":                "present",
				"more complex label": "present",
			},
		},
		{
			Name: "bar",
			Labels: map[string]string{
				"bar": "true",
			},
		},
		{
			Name: "fooer",
			Labels: map[string]string{
				"more complex label with \\ and \"": "present",
			},
		},
		{
			Name: "fooer",
			Labels: map[string]string{
				"more complex label with \\ and \".post": "present",
			},
		},
		{
			Name:  "baz",
			Other: "too complex, yo",
		},
		{
			Name:  "bazo",
			Other: "abc",
		},
		{
			Name: "compound",
			Labels: map[string]string{
				"foo": "omg_asdf.asdf-qwer",
			},
		},
	}

	var corpus []interface{}
	for _, entry := range corpusS {
		corpus = append(corpus, entry)
	}

	// adapt shows an example of how to build an adaptor function for a type.
	adapt := func(o interface{}) Adaptor {
		obj := o.(cEntry)
		return AdapterFunc(func(fieldpath []string) (string, bool) {
			switch fieldpath[0] {
			case "name":
				return obj.Name, len(obj.Name) > 0
			case "other":
				return obj.Other, len(obj.Other) > 0
			case "labels":
				value, ok := obj.Labels[strings.Join(fieldpath[1:], ".")]
				return value, ok
			}

			return "", false
		})
	}

	for _, testcase := range []struct {
		name      string
		input     string
		expected  []interface{}
		errString string
	}{
		{
			name:     "Empty",
			input:    "",
			expected: corpus,
		},
		{
			name:     "Present",
			input:    "name",
			expected: corpus,
		},
		{
			name:  "LabelPresent",
			input: "labels.foo",
			expected: []interface{}{
				corpus[0],
				corpus[2],
				corpus[8],
			},
		},
		{
			name:  "NameAndLabelPresent",
			input: "labels.foo,name",
			expected: []interface{}{
				corpus[0],
				corpus[2],
				corpus[8],
			},
		},
		{
			name:  "LabelValue",
			input: "labels.foo==true",
			expected: []interface{}{
				corpus[0],
			},
		},
		{
			name:  "LabelValuePunctuated",
			input: "labels.foo==omg_asdf.asdf-qwer",
			expected: []interface{}{
				corpus[8],
			},
		},
		{
			name:  "Name",
			input: "name==bar",
			expected: []interface{}{
				corpus[1],
				corpus[3],
			},
		},
		{
			name:  "NameNotEqual",
			input: "name!=bar",
			expected: []interface{}{
				corpus[0],
				corpus[2],
				corpus[4],
				corpus[5],
				corpus[6],
				corpus[7],
				corpus[8],
			},
		},
		{
			name:  "NameAndLabelPresent",
			input: "name==bar,labels.bar",
			expected: []interface{}{
				corpus[3],
			},
		},
		{
			name:  "QuotedValue",
			input: "other==\"too complex, yo\"",
			expected: []interface{}{
				corpus[6],
			},
		},
		{
			name:  "RegexpValue",
			input: "other~=[abc]+,name!=foo",
			expected: []interface{}{
				corpus[6],
				corpus[7],
			},
		},
		{
			name:  "NameAndLabelValue",
			input: "name==bar,labels.bar==true",
			expected: []interface{}{
				corpus[3],
			},
		},
		{
			name:  "NameAndLabelValueNoMatch",
			input: "name==bar,labels.bar==wrong",
		},
		{
			name:  "LabelQuotedFieldPathPresent",
			input: `name==foo,labels."more complex label"`,
			expected: []interface{}{
				corpus[2],
			},
		},
		{
			name:  "LabelQuotedFieldPathPresentWithQuoted",
			input: `labels."more complex label with \\ and \""==present`,
			expected: []interface{}{
				corpus[4],
			},
		},
		{
			name:  "LabelQuotedFieldPathPresentWithQuotedEmbed",
			input: `labels."more complex label with \\ and \"".post==present`,
			expected: []interface{}{
				corpus[5],
			},
		},
		{
			name:      "LabelQuotedFieldPathPresentWithQuotedEmbedInvalid",
			input:     `labels.?"more complex label with \\ and \"".post==present`,
			errString: `filters: parse error: [labels. >|?|< "more complex label with \\ and \"".post==present]: expected field or quoted`,
		},
		{
			name:      "TrailingComma",
			input:     "name==foo,",
			errString: `filters: parse error: [name==foo,]: expected field or quoted`,
		},
		{
			name:      "TrailingFieldSeparator",
			input:     "labels.",
			errString: `filters: parse error: [labels.]: expected field or quoted`,
		},
		{
			name:      "MissingValue",
			input:     "image~=,id?=?fbaq",
			errString: `filters: parse error: [image~= >|,|< id?=?fbaq]: expected value or quoted`,
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			t.Logf("testcase: %q", testcase.input)
			filter, err := Parse(testcase.input)
			if testcase.errString != "" {
				if err == nil {
					t.Fatalf("expected an error, but received nil")
				}
				if err.Error() != testcase.errString {
					t.Fatalf("error %v != %v", err, testcase.errString)
				}

				return
			}
			if err != nil {
				t.Fatal(err)
			}

			if filter == nil {
				t.Fatal("filter should not be nil")
			}

			t.Log("filter", filter)
			var results []interface{}
			for _, item := range corpus {
				adaptor := adapt(item)
				if filter.Match(adaptor) {
					results = append(results, item)
				}
			}

			if !reflect.DeepEqual(results, testcase.expected) {
				t.Fatalf("%q: %#v != %#v", testcase.input, results, testcase.expected)
			}
		})
	}
}
