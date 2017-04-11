package jsonpatch

import (
	"strings"
	"testing"
)

func mergePatch(doc, patch string) string {
	out, err := MergePatch([]byte(doc), []byte(patch))

	if err != nil {
		panic(err)
	}

	return string(out)
}

func TestMergePatchReplaceKey(t *testing.T) {
	doc := `{ "title": "hello" }`
	pat := `{ "title": "goodbye" }`

	res := mergePatch(doc, pat)

	if !compareJSON(pat, res) {
		t.Fatalf("Key was not replaced")
	}
}

func TestMergePatchIgnoresOtherValues(t *testing.T) {
	doc := `{ "title": "hello", "age": 18 }`
	pat := `{ "title": "goodbye" }`

	res := mergePatch(doc, pat)

	exp := `{ "title": "goodbye", "age": 18 }`

	if !compareJSON(exp, res) {
		t.Fatalf("Key was not replaced")
	}
}

func TestMergePatchNilDoc(t *testing.T) {
	doc := `{ "title": null }`
	pat := `{ "title": {"foo": "bar"} }`

	res := mergePatch(doc, pat)

	exp := `{ "title": {"foo": "bar"} }`

	if !compareJSON(exp, res) {
		t.Fatalf("Key was not replaced")
	}
}

func TestMergePatchRecursesIntoObjects(t *testing.T) {
	doc := `{ "person": { "title": "hello", "age": 18 } }`
	pat := `{ "person": { "title": "goodbye" } }`

	res := mergePatch(doc, pat)

	exp := `{ "person": { "title": "goodbye", "age": 18 } }`

	if !compareJSON(exp, res) {
		t.Fatalf("Key was not replaced")
	}
}

type nonObjectCases struct {
	doc, pat, res string
}

func TestMergePatchReplacesNonObjectsWholesale(t *testing.T) {
	a1 := `[1]`
	a2 := `[2]`
	o1 := `{ "a": 1 }`
	o2 := `{ "a": 2 }`
	o3 := `{ "a": 1, "b": 1 }`
	o4 := `{ "a": 2, "b": 1 }`

	cases := []nonObjectCases{
		{a1, a2, a2},
		{o1, a2, a2},
		{a1, o1, o1},
		{o3, o2, o4},
	}

	for _, c := range cases {
		act := mergePatch(c.doc, c.pat)

		if !compareJSON(c.res, act) {
			t.Errorf("whole object replacement failed")
		}
	}
}

func TestMergePatchReturnsErrorOnBadJSON(t *testing.T) {
	_, err := MergePatch([]byte(`[[[[`), []byte(`1`))

	if err == nil {
		t.Errorf("Did not return an error for bad json: %s", err)
	}

	_, err = MergePatch([]byte(`1`), []byte(`[[[[`))

	if err == nil {
		t.Errorf("Did not return an error for bad json: %s", err)
	}
}

func TestMergePatchReturnsEmptyArrayOnEmptyArray(t *testing.T) {
	doc := `{ "array": ["one", "two"] }`
	pat := `{ "array": [] }`

	exp := `{ "array": [] }`

	res, err := MergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if !compareJSON(exp, string(res)) {
		t.Fatalf("Emtpy array did not return not return as empty array")
	}
}

var rfcTests = []struct {
	target   string
	patch    string
	expected string
}{
	// test cases from https://tools.ietf.org/html/rfc7386#appendix-A
	{target: `{"a":"b"}`, patch: `{"a":"c"}`, expected: `{"a":"c"}`},
	{target: `{"a":"b"}`, patch: `{"b":"c"}`, expected: `{"a":"b","b":"c"}`},
	{target: `{"a":"b"}`, patch: `{"a":null}`, expected: `{}`},
	{target: `{"a":"b","b":"c"}`, patch: `{"a":null}`, expected: `{"b":"c"}`},
	{target: `{"a":["b"]}`, patch: `{"a":"c"}`, expected: `{"a":"c"}`},
	{target: `{"a":"c"}`, patch: `{"a":["b"]}`, expected: `{"a":["b"]}`},
	{target: `{"a":{"b": "c"}}`, patch: `{"a": {"b": "d","c": null}}`, expected: `{"a":{"b":"d"}}`},
	{target: `{"a":[{"b":"c"}]}`, patch: `{"a":[1]}`, expected: `{"a":[1]}`},
	{target: `["a","b"]`, patch: `["c","d"]`, expected: `["c","d"]`},
	{target: `{"a":"b"}`, patch: `["c"]`, expected: `["c"]`},
	// {target: `{"a":"foo"}`, patch: `null`, expected: `null`},
	// {target: `{"a":"foo"}`, patch: `"bar"`, expected: `"bar"`},
	{target: `{"e":null}`, patch: `{"a":1}`, expected: `{"a":1,"e":null}`},
	{target: `[1,2]`, patch: `{"a":"b","c":null}`, expected: `{"a":"b"}`},
	{target: `{}`, patch: `{"a":{"bb":{"ccc":null}}}`, expected: `{"a":{"bb":{}}}`},
}

func TestMergePatchRFCCases(t *testing.T) {
	for i, c := range rfcTests {
		out := mergePatch(c.target, c.patch)

		if !compareJSON(out, c.expected) {
			t.Errorf("case[%d], patch '%s' did not apply properly to '%s'. expected:\n'%s'\ngot:\n'%s'", i, c.patch, c.target, c.expected, out)
		}
	}
}

var rfcFailTests = `
     {"a":"foo"}  |   null
     {"a":"foo"}  |   "bar"
`

func TestMergePatchFailRFCCases(t *testing.T) {
	tests := strings.Split(rfcFailTests, "\n")

	for _, c := range tests {
		if strings.TrimSpace(c) == "" {
			continue
		}

		parts := strings.SplitN(c, "|", 2)

		doc := strings.TrimSpace(parts[0])
		pat := strings.TrimSpace(parts[1])

		out, err := MergePatch([]byte(doc), []byte(pat))

		if err != errBadJSONPatch {
			t.Errorf("error not returned properly: %s, %s", err, string(out))
		}
	}

}

func TestMergeReplaceKey(t *testing.T) {
	doc := `{ "title": "hello", "nested": {"one": 1, "two": 2} }`
	pat := `{ "title": "goodbye", "nested": {"one": 2, "two": 2}  }`

	exp := `{ "title": "goodbye", "nested": {"one": 2}  }`

	res, err := CreateMergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if !compareJSON(exp, string(res)) {
		t.Fatalf("Key was not replaced")
	}
}

func TestMergeGetArray(t *testing.T) {
	doc := `{ "title": "hello", "array": ["one", "two"], "notmatch": [1, 2, 3] }`
	pat := `{ "title": "hello", "array": ["one", "two", "three"], "notmatch": [1, 2, 3]  }`

	exp := `{ "array": ["one", "two", "three"] }`

	res, err := CreateMergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if !compareJSON(exp, string(res)) {
		t.Fatalf("Array was not added")
	}
}

func TestMergeGetObjArray(t *testing.T) {
	doc := `{ "title": "hello", "array": [{"banana": true}, {"evil": false}], "notmatch": [{"one":1}, {"two":2}, {"three":3}] }`
	pat := `{ "title": "hello", "array": [{"banana": false}, {"evil": true}], "notmatch": [{"one":1}, {"two":2}, {"three":3}] }`

	exp := `{  "array": [{"banana": false}, {"evil": true}] }`

	res, err := CreateMergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if !compareJSON(exp, string(res)) {
		t.Fatalf("Object array was not added")
	}
}

func TestMergeDeleteKey(t *testing.T) {
	doc := `{ "title": "hello", "nested": {"one": 1, "two": 2} }`
	pat := `{ "title": "hello", "nested": {"one": 1}  }`

	exp := `{"nested":{"two":null}}`

	res, err := CreateMergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	// We cannot use "compareJSON", since Equals does not report a difference if the value is null
	if exp != string(res) {
		t.Fatalf("Key was not removed")
	}
}

func TestMergeEmptyArray(t *testing.T) {
	doc := `{ "array": null }`
	pat := `{ "array": [] }`

	exp := `{"array":[]}`

	res, err := CreateMergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	// We cannot use "compareJSON", since Equals does not report a difference if the value is null
	if exp != string(res) {
		t.Fatalf("Key was not removed")
	}
}

func TestMergeObjArray(t *testing.T) {
	doc := `{ "array": [ {"a": {"b": 2}}, {"a": {"b": 3}} ]}`
	exp := `{}`

	res, err := CreateMergePatch([]byte(doc), []byte(doc))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	// We cannot use "compareJSON", since Equals does not report a difference if the value is null
	if exp != string(res) {
		t.Fatalf("Array was not empty, was " + string(res))
	}
}

func TestMergeComplexMatch(t *testing.T) {
	doc := `{"hello": "world","t": true ,"f": false, "n": null,"i": 123,"pi": 3.1416,"a": [1, 2, 3, 4], "nested": {"hello": "world","t": true ,"f": false, "n": null,"i": 123,"pi": 3.1416,"a": [1, 2, 3, 4]} }`
	empty := `{}`
	res, err := CreateMergePatch([]byte(doc), []byte(doc))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	// We cannot use "compareJSON", since Equals does not report a difference if the value is null
	if empty != string(res) {
		t.Fatalf("Did not get empty result, was:%s", string(res))
	}
}

func TestMergeComplexAddAll(t *testing.T) {
	doc := `{"hello": "world","t": true ,"f": false, "n": null,"i": 123,"pi": 3.1416,"a": [1, 2, 3, 4], "nested": {"hello": "world","t": true ,"f": false, "n": null,"i": 123,"pi": 3.1416,"a": [1, 2, 3, 4]} }`
	empty := `{}`
	res, err := CreateMergePatch([]byte(empty), []byte(doc))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if !compareJSON(doc, string(res)) {
		t.Fatalf("Did not get everything as, it was:\n%s", string(res))
	}
}

func TestMergeComplexRemoveAll(t *testing.T) {
	doc := `{"hello": "world","t": true ,"f": false, "n": null,"i": 123,"pi": 3.1416,"a": [1, 2, 3, 4], "nested": {"hello": "world","t": true ,"f": false, "n": null,"i": 123,"pi": 3.1416,"a": [1, 2, 3, 4]} }`
	exp := `{"a":null,"f":null,"hello":null,"i":null,"n":null,"nested":null,"pi":null,"t":null}`
	empty := `{}`
	res, err := CreateMergePatch([]byte(doc), []byte(empty))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if exp != string(res) {
		t.Fatalf("Did not get result, was:%s", string(res))
	}

	// FIXME: Crashes if using compareJSON like this:
	/*
		if !compareJSON(doc, string(res)) {
			t.Fatalf("Did not get everything as, it was:\n%s", string(res))
		}
	*/
}

func TestMergeObjectWithInnerArray(t *testing.T) {
	stateString := `{
	  "OuterArray": [
	    {
		  "InnerArray": [
	        {
	          "StringAttr": "abc123"
	        }
	      ],
	      "StringAttr": "def456"
	    }
	  ]
	}`

	patch, err := CreateMergePatch([]byte(stateString), []byte(stateString))
	if err != nil {
		t.Fatal(err)
	}

	if string(patch) != "{}" {
		t.Fatalf("Patch should have been {} but was: %v", string(patch))
	}
}

func TestMergeReplaceKeyRequiringEscape(t *testing.T) {
	doc := `{ "title": "hello", "nested": {"title/escaped": 1, "two": 2} }`
	pat := `{ "title": "goodbye", "nested": {"title/escaped": 2, "two": 2}  }`

	exp := `{ "title": "goodbye", "nested": {"title~1escaped": 2}  }`

	res, err := CreateMergePatch([]byte(doc), []byte(pat))

	if err != nil {
		t.Errorf("Unexpected error: %s, %s", err, string(res))
	}

	if !compareJSON(exp, string(res)) {
		t.Log(string(res))
		t.Fatalf("Key was not replaced")
	}
}

func TestMergePatchReplaceKeyRequiringEscaping(t *testing.T) {
	doc := `{ "obj": { "title/escaped": "hello" } }`
	pat := `{ "obj": { "title~1escaped": "goodbye" } }`
	exp := `{ "obj": { "title/escaped": "goodbye" } }`

	res := mergePatch(doc, pat)

	if !compareJSON(exp, res) {
		t.Fatalf("Key was not replaced")
	}
}

func TestMergeMergePatches(t *testing.T) {
	cases := []struct {
		demonstrates string
		p1           string
		p2           string
		exp          string
	}{
		{
			demonstrates: "simple patches are merged normally",
			p1:           `{"add1": 1}`,
			p2:           `{"add2": 2}`,
			exp:          `{"add1": 1, "add2": 2}`,
		},
		{
			demonstrates: "nulls are kept",
			p1:           `{"del1": null}`,
			p2:           `{"del2": null}`,
			exp:          `{"del1": null, "del2": null}`,
		},
		{
			demonstrates: "a key added then deleted is kept deleted",
			p1:           `{"add_then_delete": "atd"}`,
			p2:           `{"add_then_delete": null}`,
			exp:          `{"add_then_delete": null}`,
		},
		{
			demonstrates: "a key deleted then added is kept added",
			p1:           `{"delete_then_add": null}`,
			p2:           `{"delete_then_add": "dta"}`,
			exp:          `{"delete_then_add": "dta"}`,
		},
		{
			demonstrates: "object overrides array",
			p1:           `[]`,
			p2:           `{"del": null, "add": "a"}`,
			exp:          `{"del": null, "add": "a"}`,
		},
		{
			demonstrates: "array overrides object",
			p1:           `{"del": null, "add": "a"}`,
			p2:           `[]`,
			exp:          `[]`,
		},
	}

	for _, c := range cases {
		out, err := MergeMergePatches([]byte(c.p1), []byte(c.p2))

		if err != nil {
			panic(err)
		}

		if !compareJSON(c.exp, string(out)) {
			t.Logf("Error while trying to demonstrate: %v", c.demonstrates)
			t.Logf("Got %v", string(out))
			t.Logf("Expected %v", c.exp)
			t.Fatalf("Merged merge patch is incorrect")
		}
	}
}
