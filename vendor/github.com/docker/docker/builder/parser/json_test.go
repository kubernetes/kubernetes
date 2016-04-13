package parser

import (
	"testing"
)

var invalidJSONArraysOfStrings = []string{
	`["a",42,"b"]`,
	`["a",123.456,"b"]`,
	`["a",{},"b"]`,
	`["a",{"c": "d"},"b"]`,
	`["a",["c"],"b"]`,
	`["a",true,"b"]`,
	`["a",false,"b"]`,
	`["a",null,"b"]`,
}

var validJSONArraysOfStrings = map[string][]string{
	`[]`:           {},
	`[""]`:         {""},
	`["a"]`:        {"a"},
	`["a","b"]`:    {"a", "b"},
	`[ "a", "b" ]`: {"a", "b"},
	`[	"a",	"b"	]`: {"a", "b"},
	`	[	"a",	"b"	]	`: {"a", "b"},
	`["abc 123", "♥", "☃", "\" \\ \/ \b \f \n \r \t \u0000"]`: {"abc 123", "♥", "☃", "\" \\ / \b \f \n \r \t \u0000"},
}

func TestJSONArraysOfStrings(t *testing.T) {
	for json, expected := range validJSONArraysOfStrings {
		if node, _, err := parseJSON(json); err != nil {
			t.Fatalf("%q should be a valid JSON array of strings, but wasn't! (err: %q)", json, err)
		} else {
			i := 0
			for node != nil {
				if i >= len(expected) {
					t.Fatalf("expected result is shorter than parsed result (%d vs %d+) in %q", len(expected), i+1, json)
				}
				if node.Value != expected[i] {
					t.Fatalf("expected %q (not %q) in %q at pos %d", expected[i], node.Value, json, i)
				}
				node = node.Next
				i++
			}
			if i != len(expected) {
				t.Fatalf("expected result is longer than parsed result (%d vs %d) in %q", len(expected), i+1, json)
			}
		}
	}
	for _, json := range invalidJSONArraysOfStrings {
		if _, _, err := parseJSON(json); err != errDockerfileNotStringArray {
			t.Fatalf("%q should be an invalid JSON array of strings, but wasn't!", json)
		}
	}
}
