package jsonpath

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"
)

var decoderSkipTests = []struct {
	in    string
	path  []interface{}
	match bool
	err   error
	out   interface{}
}{
	{in: `{}`, path: []interface{}{"a", "b2"}, match: false, err: nil},
	{in: `{"a":{"b":{"c":14}}}`, path: []interface{}{"a"}, match: true, out: map[string]interface{}{"b": map[string]interface{}{"c": float64(14)}}, err: nil},
	{in: `{"a":{"b":{"c":14}}}`, path: []interface{}{"a", "b"}, match: true, out: map[string]interface{}{"c": float64(14)}, err: nil},
	{in: `{"a":{"b":{"c":14}}}`, path: []interface{}{"a", "b", "c"}, match: true, out: float64(14), err: nil},
	{in: `{"a":{"d":{"b":{"c":3}}, "b":{"c":14}}}`, path: []interface{}{"a", "b", "c"}, match: true, out: float64(14), err: nil},
	{in: `{"a":{"b":{"c":14},"b2":3}}`, path: []interface{}{"a", "b2"}, match: true, out: float64(3), err: nil},

	{in: `[]`, path: []interface{}{2}, match: false, err: nil},
	{in: `[0,1]`, path: []interface{}{0}, match: true, out: float64(0), err: nil},
	{in: `[0,1,2]`, path: []interface{}{2}, match: true, out: float64(2), err: nil},
	{in: `[0,1 , 2]`, path: []interface{}{2}, match: true, out: float64(2), err: nil},
	{in: `[0,{"b":1},2]`, path: []interface{}{1}, match: true, out: map[string]interface{}{"b": float64(1)}, err: nil},
	{in: `[1,{"b":1},2]`, path: []interface{}{2}, match: true, out: float64(2), err: nil},
	{in: `[1,{"b":[1]},2]`, path: []interface{}{2}, match: true, out: float64(2), err: nil},
	{in: `[1,[{"b":[1]},3],2]`, path: []interface{}{2}, match: true, out: float64(2), err: nil},

	{in: `[1,[{"b":[1]},3],4]`, path: []interface{}{1, 1}, match: true, out: float64(3), err: nil},
	{in: `[1,[{"b":[1]},3],2]`, path: []interface{}{1, 0, "b", 0}, match: true, out: float64(1), err: nil},
	{in: `[1,[{"b":[1]},3],5]`, path: []interface{}{2}, match: true, out: float64(5), err: nil},
	{in: `{"b":[{"a":0},{"a":1}]}`, path: []interface{}{"b", 0, "a"}, match: true, out: float64(0), err: nil},
	{in: `{"b":[{"a":0},{"a":1}]}`, path: []interface{}{"b", 1, "a"}, match: true, out: float64(1), err: nil},
	{in: `{"a":"b","b":"z","z":"s"}`, path: []interface{}{"b"}, match: true, out: "z", err: nil},
	{in: `{"a":"b","b":"z","l":0,"z":"s"}`, path: []interface{}{"z"}, match: true, out: "s", err: nil},
}

func TestDecoderSeekTo(t *testing.T) {
	var testDesc string
	var v interface{}
	var match bool
	var err error

	for ti, tst := range decoderSkipTests {

		w := NewDecoder(bytes.NewBuffer([]byte(tst.in)))
		testDesc = fmt.Sprintf("#%v '%s'", ti, tst.in)

		match, err = w.SeekTo(tst.path...)
		if match != tst.match {
			t.Errorf("#%v expected match=%v was match=%v : %v", ti, tst.match, match, testDesc)
		}
		if !reflect.DeepEqual(err, tst.err) {
			t.Errorf("#%v unexpected error: '%v' expecting '%v' : %v", ti, err, tst.err, testDesc)
		}

		if match {
			if err = w.Decode(&v); err != nil {
				t.Errorf("#%v decode: error: '%v' : %v", ti, err, testDesc)
			}
			if !reflect.DeepEqual(v, tst.out) {
				t.Errorf("#%v decode: expected %#v, was %#v : %v", ti, tst.out, v, testDesc)
			}
		}
	}
}

func TestDecoderMoveMultiple(t *testing.T) {

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
		"bool": true
	}`)

	tests := [][]struct {
		path  []interface{}
		out   interface{}
		match bool
	}{
		{ // Every Element
			{path: []interface{}{"foo"}, match: true, out: float64(1)},
			{path: []interface{}{"bar"}, match: true, out: float64(2)},
			{path: []interface{}{"test"}, match: true, out: "Hello, world!"},
			{path: []interface{}{"baz"}, match: true, out: float64(123.1)},
			{path: []interface{}{"array", 0, "foo"}, match: true, out: float64(1)},
			{path: []interface{}{"array", 1, "bar"}, match: true, out: float64(2)},
			{path: []interface{}{"array", 2, "baz"}, match: true, out: float64(3)},
			{path: []interface{}{"subobj", "foo"}, match: true, out: float64(1)},
			{path: []interface{}{"subobj", "subarray", 0}, match: true, out: float64(1)},
			{path: []interface{}{"subobj", "subarray", 1}, match: true, out: float64(2)},
			{path: []interface{}{"subobj", "subarray", 2}, match: true, out: float64(3)},
			{path: []interface{}{"subobj", "subsubobj", "bar"}, match: true, out: float64(2)},
			{path: []interface{}{"subobj", "subsubobj", "baz"}, match: true, out: float64(3)},
			{path: []interface{}{"subobj", "subsubobj", "array", 0}, match: true, out: "hello"},
			{path: []interface{}{"subobj", "subsubobj", "array", 1}, match: true, out: "world"},
			{path: []interface{}{"bool"}, match: true, out: true},
		},
		{ // Deep, then shallow
			{path: []interface{}{"subobj", "subsubobj", "array", 0}, match: true, out: "hello"},
			{path: []interface{}{"bool"}, match: true, out: true},
		},
		{ // Complex, then shallow
			{path: []interface{}{"array"}, match: true, out: []interface{}{map[string]interface{}{"foo": float64(1)}, map[string]interface{}{"bar": float64(2)}, map[string]interface{}{"baz": float64(3)}}},
			{path: []interface{}{"subobj", "subsubobj"}, match: true, out: map[string]interface{}{"bar": float64(2), "baz": float64(3), "array": []interface{}{"hello", "world"}}},
			{path: []interface{}{"bool"}, match: true, out: true},
		},
		{
			{path: []interface{}{"foo"}, match: true, out: float64(1)},
			{path: []interface{}{"test"}, match: true, out: "Hello, world!"},
			{path: []interface{}{"array", 1, "bar"}, match: true, out: float64(2)},
			{path: []interface{}{"subobj", "subarray", 1}, match: true, out: float64(2)},
			{path: []interface{}{"subobj", "subarray", 2}, match: true, out: float64(3)},
			{path: []interface{}{"subobj", "subsubobj", "array", 1}, match: true, out: "world"},
		},
	}

	for _, tst := range tests {

		w := NewDecoder(bytes.NewBuffer(j))

		var err error
		var v interface{}
		var m bool

		for i, step := range tst {

			m, err = w.SeekTo(step.path...)
			if m != step.match {
				t.Errorf("@%v expected match=%v, but was match=%v", i, step.match, m)
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			err = w.Decode(&v)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(v, step.out) {
				t.Errorf("expected %v but was %v", step.out, v)
			}
		}
	}

}
