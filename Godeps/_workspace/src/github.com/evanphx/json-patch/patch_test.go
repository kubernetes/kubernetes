package jsonpatch

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
)

func reformatJSON(j string) string {
	buf := new(bytes.Buffer)

	json.Indent(buf, []byte(j), "", "  ")

	return buf.String()
}

func compareJSON(a, b string) bool {
	// return Equal([]byte(a), []byte(b))

	var obj_a, obj_b map[string]interface{}
	json.Unmarshal([]byte(a), &obj_a)
	json.Unmarshal([]byte(b), &obj_b)

	// fmt.Printf("Comparing %#v\nagainst %#v\n", obj_a, obj_b)
	return reflect.DeepEqual(obj_a, obj_b)
}

func applyPatch(doc, patch string) (string, error) {
	obj, err := DecodePatch([]byte(patch))

	if err != nil {
		panic(err)
	}

	out, err := obj.Apply([]byte(doc))

	if err != nil {
		return "", err
	}

	return string(out), nil
}

type Case struct {
	doc, patch, result string
}

var Cases = []Case{
	{
		`{ "foo": "bar"}`,
		`[
         { "op": "add", "path": "/baz", "value": "qux" }
     ]`,
		`{
       "baz": "qux",
       "foo": "bar"
     }`,
	},
	{
		`{ "foo": [ "bar", "baz" ] }`,
		`[
     { "op": "add", "path": "/foo/1", "value": "qux" }
    ]`,
		`{ "foo": [ "bar", "qux", "baz" ] }`,
	},
	{
		`{ "baz": "qux", "foo": "bar" }`,
		`[ { "op": "remove", "path": "/baz" } ]`,
		`{ "foo": "bar" }`,
	},
	{
		`{ "foo": [ "bar", "qux", "baz" ] }`,
		`[ { "op": "remove", "path": "/foo/1" } ]`,
		`{ "foo": [ "bar", "baz" ] }`,
	},
	{
		`{ "baz": "qux", "foo": "bar" }`,
		`[ { "op": "replace", "path": "/baz", "value": "boo" } ]`,
		`{ "baz": "boo", "foo": "bar" }`,
	},
	{
		`{
     "foo": {
       "bar": "baz",
       "waldo": "fred"
     },
     "qux": {
       "corge": "grault"
     }
   }`,
		`[ { "op": "move", "from": "/foo/waldo", "path": "/qux/thud" } ]`,
		`{
     "foo": {
       "bar": "baz"
     },
     "qux": {
       "corge": "grault",
       "thud": "fred"
     }
   }`,
	},
	{
		`{ "foo": [ "all", "grass", "cows", "eat" ] }`,
		`[ { "op": "move", "from": "/foo/1", "path": "/foo/3" } ]`,
		`{ "foo": [ "all", "cows", "eat", "grass" ] }`,
	},
	{
		`{ "foo": "bar" }`,
		`[ { "op": "add", "path": "/child", "value": { "grandchild": { } } } ]`,
		`{ "foo": "bar", "child": { "grandchild": { } } }`,
	},
	{
		`{ "foo": ["bar"] }`,
		`[ { "op": "add", "path": "/foo/-", "value": ["abc", "def"] } ]`,
		`{ "foo": ["bar", ["abc", "def"]] }`,
	},
	{
		`{ "foo": "bar", "qux": { "baz": 1, "bar": null } }`,
		`[ { "op": "remove", "path": "/qux/bar" } ]`,
		`{ "foo": "bar", "qux": { "baz": 1 } }`,
	},
}

type BadCase struct {
	doc, patch string
}

var MutationTestCases = []BadCase{
	{
		`{ "foo": "bar", "qux": { "baz": 1, "bar": null } }`,
		`[ { "op": "remove", "path": "/qux/bar" } ]`,
	},
}

var BadCases = []BadCase{
	{
		`{ "foo": "bar" }`,
		`[ { "op": "add", "path": "/baz/bat", "value": "qux" } ]`,
	},
}

func TestAllCases(t *testing.T) {
	for _, c := range Cases {
		out, err := applyPatch(c.doc, c.patch)

		if err != nil {
			t.Errorf("Unable to apply patch: %s", err)
		}

		if !compareJSON(out, c.result) {
			t.Errorf("Patch did not apply. Expected:\n%s\n\nActual:\n%s",
				reformatJSON(c.result), reformatJSON(out))
		}
	}

	for _, c := range MutationTestCases {
		out, err := applyPatch(c.doc, c.patch)

		if err != nil {
			t.Errorf("Unable to apply patch: %s", err)
		}

		if compareJSON(out, c.doc) {
			t.Errorf("Patch did not apply. Original:\n%s\n\nPatched:\n%s",
				reformatJSON(c.doc), reformatJSON(out))
		}
	}

	for _, c := range BadCases {
		_, err := applyPatch(c.doc, c.patch)

		if err == nil {
			t.Errorf("Patch should have failed to apply but it did not")
		}
	}
}

type TestCase struct {
	doc, patch string
	result     bool
	failedPath string
}

var TestCases = []TestCase{
	{
		`{
			"baz": "qux",
			"foo": [ "a", 2, "c" ]
		}`,
		`[
			{ "op": "test", "path": "/baz", "value": "qux" },
			{ "op": "test", "path": "/foo/1", "value": 2 }
		]`,
		true,
		"",
	},
	{
		`{ "baz": "qux" }`,
		`[ { "op": "test", "path": "/baz", "value": "bar" } ]`,
		false,
		"/baz",
	},
	{
		`{
			"baz": "qux",
			"foo": ["a", 2, "c"]
		}`,
		`[
			{ "op": "test", "path": "/baz", "value": "qux" },
			{ "op": "test", "path": "/foo/1", "value": "c" }
		]`,
		false,
		"/foo/1",
	},
}

func TestAllTest(t *testing.T) {
	for _, c := range TestCases {
		_, err := applyPatch(c.doc, c.patch)

		if c.result && err != nil {
			t.Errorf("Testing failed when it should have passed: %s", err)
		} else if !c.result && err == nil {
			t.Errorf("Testing passed when it should have faild: %s", err)
		} else if !c.result {
			expected := fmt.Sprintf("Testing value %s failed", c.failedPath)
			if err.Error() != expected {
				t.Errorf("Testing failed as expected but invalid message: expected [%s], got [%s]", expected, err)
			}
		}
	}
}
