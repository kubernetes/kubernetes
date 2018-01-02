package jsonpath

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
)

func TestTokensAndPaths(t *testing.T) {
	j := []byte(`
	{
		"foo": 1,
		"bar": 2,
		"test": "Hello, world!",
		"baz": 123.1,
		"array": [
			{"foo": 1},
			{"bar": 2},
			{"baz": 3}
		],
		"subobj": {
			"foo": 1,
			"subarray": [1,2,3],
			"subsubobj": {
				"bar": 2,
				"baz": 3,
				"array": ["hello", "world"]
			}
		},
		"bool": true,
		"a": [[0],[[1]]]
	}`)

	expPaths := []JsonPath{
		{""},
		{"foo"}, {"foo"},
		{"bar"}, {"bar"},
		{"test"}, {"test"},
		{"baz"}, {"baz"},
		{"array"}, {"array", -1},
		{"array", 0, ""}, {"array", 0, "foo"}, {"array", 0, "foo"}, {"array", 0},
		{"array", 1, ""}, {"array", 1, "bar"}, {"array", 1, "bar"}, {"array", 1},
		{"array", 2, ""}, {"array", 2, "baz"}, {"array", 2, "baz"}, {"array", 2},
		{"array"},
		{"subobj"},
		{"subobj", ""},
		{"subobj", "foo"}, {"subobj", "foo"},
		{"subobj", "subarray"}, {"subobj", "subarray", -1},
		{"subobj", "subarray", 0}, {"subobj", "subarray", 1}, {"subobj", "subarray", 2},
		{"subobj", "subarray"},
		{"subobj", "subsubobj"}, {"subobj", "subsubobj", ""},
		{"subobj", "subsubobj", "bar"}, {"subobj", "subsubobj", "bar"},
		{"subobj", "subsubobj", "baz"}, {"subobj", "subsubobj", "baz"},
		{"subobj", "subsubobj", "array"}, {"subobj", "subsubobj", "array", -1},
		{"subobj", "subsubobj", "array", 0}, {"subobj", "subsubobj", "array", 1},
		{"subobj", "subsubobj", "array"},
		{"subobj", "subsubobj"}, {"subobj"},
		{"bool"}, {"bool"},
		{"a"}, {"a", -1}, {"a", 0, -1}, {"a", 0, 0}, {"a", 0}, {"a", 1, -1},
		{"a", 1, 0, -1}, {"a", 1, 0, 0}, {"a", 1, 0}, {"a", 1}, {"a"},
		{},
	}

	expTokens := []json.Token{
		json.Delim('{'),
		KeyString("foo"), float64(1),
		KeyString("bar"), float64(2),
		KeyString("test"), "Hello, world!",
		KeyString("baz"), float64(123.1),
		KeyString("array"), json.Delim('['),
		json.Delim('{'), KeyString("foo"), float64(1), json.Delim('}'),
		json.Delim('{'), KeyString("bar"), float64(2), json.Delim('}'),
		json.Delim('{'), KeyString("baz"), float64(3), json.Delim('}'),
		json.Delim(']'),
		KeyString("subobj"),
		json.Delim('{'),
		KeyString("foo"), float64(1),
		KeyString("subarray"), json.Delim('['),
		float64(1), float64(2), float64(3),
		json.Delim(']'),
		KeyString("subsubobj"), json.Delim('{'),
		KeyString("bar"), float64(2),
		KeyString("baz"), float64(3),
		KeyString("array"), json.Delim('['),
		"hello", "world",
		json.Delim(']'),
		json.Delim('}'), json.Delim('}'),
		KeyString("bool"), true,
		KeyString("a"),
		json.Delim('['), json.Delim('['), float64(0), json.Delim(']'),
		json.Delim('['), json.Delim('['), float64(1), json.Delim(']'), json.Delim(']'),
		json.Delim(']'),
		json.Delim('}'),
	}
	outTokens := []json.Token{}
	outPaths := []JsonPath{}

	d := NewDecoder(bytes.NewBuffer(j))

	for {
		st, err := d.Token()
		if err == io.EOF {
			break
		} else if err != nil {
			t.Error(err)
			break
		}
		outTokens = append(outTokens, st)
		outPaths = append(outPaths, d.Path())
	}

	// Check tokens
	if len(outTokens) != len(expTokens) {
		t.Errorf("Out has %v elements, expected %v", len(outTokens), len(expTokens))
	}
	for i, v := range expTokens {
		if v != outTokens[i] {
			t.Errorf("@%v exp: %T:%v but was: %T:%v", i, v, v, outTokens[i], outTokens[i])
		}
	}

	// Check paths
	if len(outPaths) != len(expPaths) {
		t.Errorf("Outpaths has %v elements, expected %v", len(outPaths), len(expPaths))
	}
	for i, v := range expPaths {
		if !v.Equal(outPaths[i]) {
			t.Errorf("@%v exp: %T:%v but was: %T:%v", i, v, v, outPaths[i], outPaths[i])
		}
	}
}
