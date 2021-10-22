package yaml

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"testing"

	"github.com/davecgh/go-spew/spew"
	yaml "gopkg.in/yaml.v2"
)

type MarshalTest struct {
	A string
	B int64
	// Would like to test float64, but it's not supported in go-yaml.
	// (See https://github.com/go-yaml/yaml/issues/83.)
	C float32
}

func TestMarshal(t *testing.T) {
	f32String := strconv.FormatFloat(math.MaxFloat32, 'g', -1, 32)
	s := MarshalTest{"a", math.MaxInt64, math.MaxFloat32}
	e := []byte(fmt.Sprintf("A: a\nB: %d\nC: %s\n", math.MaxInt64, f32String))

	y, err := Marshal(s)
	if err != nil {
		t.Errorf("error marshaling YAML: %v", err)
	}

	if !reflect.DeepEqual(y, e) {
		t.Errorf("marshal YAML was unsuccessful, expected: %#v, got: %#v",
			string(e), string(y))
	}
}

type UnmarshalString struct {
	A    string
	True string
}

type UnmarshalStringMap struct {
	A map[string]string
}

type UnmarshalNestedString struct {
	A NestedString
}

type NestedString struct {
	A string
}

type UnmarshalSlice struct {
	A []NestedSlice
}

type NestedSlice struct {
	B string
	C *string
}

func TestUnmarshal(t *testing.T) {
	y := []byte("a: 1")
	s1 := UnmarshalString{}
	e1 := UnmarshalString{A: "1"}
	unmarshal(t, y, &s1, &e1)

	y = []byte("a: true")
	s1 = UnmarshalString{}
	e1 = UnmarshalString{A: "true"}
	unmarshal(t, y, &s1, &e1)

	y = []byte("true: 1")
	s1 = UnmarshalString{}
	e1 = UnmarshalString{True: "1"}
	unmarshal(t, y, &s1, &e1)

	y = []byte("a:\n  a: 1")
	s2 := UnmarshalNestedString{}
	e2 := UnmarshalNestedString{NestedString{"1"}}
	unmarshal(t, y, &s2, &e2)

	y = []byte("a:\n  - b: abc\n    c: def\n  - b: 123\n    c: 456\n")
	s3 := UnmarshalSlice{}
	e3 := UnmarshalSlice{[]NestedSlice{NestedSlice{"abc", strPtr("def")}, NestedSlice{"123", strPtr("456")}}}
	unmarshal(t, y, &s3, &e3)

	y = []byte("a:\n  b: 1")
	s4 := UnmarshalStringMap{}
	e4 := UnmarshalStringMap{map[string]string{"b": "1"}}
	unmarshal(t, y, &s4, &e4)

	y = []byte(`
a:
  name: TestA
b:
  name: TestB
`)
	type NamedThing struct {
		Name string `json:"name"`
	}
	s5 := map[string]*NamedThing{}
	e5 := map[string]*NamedThing{
		"a": &NamedThing{Name: "TestA"},
		"b": &NamedThing{Name: "TestB"},
	}
	unmarshal(t, y, &s5, &e5)
}

func unmarshal(t *testing.T, y []byte, s, e interface{}, opts ...JSONOpt) {
	err := Unmarshal(y, s, opts...)
	if err != nil {
		t.Errorf("error unmarshaling YAML: %v", err)
	}

	if !reflect.DeepEqual(s, e) {
		t.Errorf("unmarshal YAML was unsuccessful, expected: %+#v, got: %+#v",
			e, s)
	}
}

func TestUnmarshalStrict(t *testing.T) {
	y := []byte("a: 1")
	s1 := UnmarshalString{}
	e1 := UnmarshalString{A: "1"}
	unmarshalStrict(t, y, &s1, &e1)

	y = []byte("a: true")
	s1 = UnmarshalString{}
	e1 = UnmarshalString{A: "true"}
	unmarshalStrict(t, y, &s1, &e1)

	y = []byte("true: 1")
	s1 = UnmarshalString{}
	e1 = UnmarshalString{True: "1"}
	unmarshalStrict(t, y, &s1, &e1)

	y = []byte("a:\n  a: 1")
	s2 := UnmarshalNestedString{}
	e2 := UnmarshalNestedString{NestedString{"1"}}
	unmarshalStrict(t, y, &s2, &e2)

	y = []byte("a:\n  - b: abc\n    c: def\n  - b: 123\n    c: 456\n")
	s3 := UnmarshalSlice{}
	e3 := UnmarshalSlice{[]NestedSlice{NestedSlice{"abc", strPtr("def")}, NestedSlice{"123", strPtr("456")}}}
	unmarshalStrict(t, y, &s3, &e3)

	y = []byte("a:\n  b: 1")
	s4 := UnmarshalStringMap{}
	e4 := UnmarshalStringMap{map[string]string{"b": "1"}}
	unmarshalStrict(t, y, &s4, &e4)

	y = []byte(`
a:
  name: TestA
b:
  name: TestB
`)
	type NamedThing struct {
		Name string `json:"name"`
	}
	s5 := map[string]*NamedThing{}
	e5 := map[string]*NamedThing{
		"a": &NamedThing{Name: "TestA"},
		"b": &NamedThing{Name: "TestB"},
	}
	unmarshal(t, y, &s5, &e5)

	// When using not-so-strict unmarshal, we should
	// be picking up the ID-1 as the value in the "id" field
	y = []byte(`
a:
  name: TestA
  id: ID-A
  id: ID-1
`)
	type NamedThing2 struct {
		Name string `json:"name"`
		ID   string `json:"id"`
	}
	s6 := map[string]*NamedThing2{}
	e6 := map[string]*NamedThing2{
		"a": {Name: "TestA", ID: "ID-1"},
	}
	unmarshal(t, y, &s6, &e6)
}

func TestUnmarshalStrictFails(t *testing.T) {
	y := []byte("a: true\na: false")
	s1 := UnmarshalString{}
	unmarshalStrictFail(t, y, &s1)

	y = []byte("a:\n  - b: abc\n    c: 32\n      b: 123")
	s2 := UnmarshalSlice{}
	unmarshalStrictFail(t, y, &s2)

	y = []byte("a:\n  b: 1\n    c: 3")
	s3 := UnmarshalStringMap{}
	unmarshalStrictFail(t, y, &s3)

	type NamedThing struct {
		Name string `json:"name"`
		ID   string `json:"id"`
	}
	// When using strict unmarshal, we should see
	// the unmarshal fail if there are multiple keys
	y = []byte(`
a:
  name: TestA
  id: ID-A
  id: ID-1
`)
	s4 := NamedThing{}
	unmarshalStrictFail(t, y, &s4)

	// Strict unmarshal should fail for unknown fields
	y = []byte(`
name: TestB
id: ID-B
unknown: Some-Value
`)
	s5 := NamedThing{}
	unmarshalStrictFail(t, y, &s5)
}

func unmarshalStrict(t *testing.T, y []byte, s, e interface{}, opts ...JSONOpt) {
	err := UnmarshalStrict(y, s, opts...)
	if err != nil {
		t.Errorf("error unmarshaling YAML: %v", err)
	}

	if !reflect.DeepEqual(s, e) {
		t.Errorf("unmarshal YAML was unsuccessful, expected: %+#v, got: %+#v",
			e, s)
	}
}

func unmarshalStrictFail(t *testing.T, y []byte, s interface{}, opts ...JSONOpt) {
	err := UnmarshalStrict(y, s, opts...)
	if err == nil {
		t.Errorf("error unmarshaling YAML: %v", err)
	}
}

type Case struct {
	input  string
	output string
	// By default we test that reversing the output == input. But if there is a
	// difference in the reversed output, you can optionally specify it here.
	reverse *string
}

type RunType int

const (
	RunTypeJSONToYAML RunType = iota
	RunTypeYAMLToJSON
)

func TestJSONToYAML(t *testing.T) {
	cases := []Case{
		{
			`{"t":"a"}`,
			"t: a\n",
			nil,
		}, {
			`{"t":null}`,
			"t: null\n",
			nil,
		},
	}

	runCases(t, RunTypeJSONToYAML, cases)
}

func TestYAMLToJSON(t *testing.T) {
	cases := []Case{
		{
			"t: a\n",
			`{"t":"a"}`,
			nil,
		}, {
			"t: \n",
			`{"t":null}`,
			strPtr("t: null\n"),
		}, {
			"t: null\n",
			`{"t":null}`,
			nil,
		}, {
			"1: a\n",
			`{"1":"a"}`,
			strPtr("\"1\": a\n"),
		}, {
			"1000000000000000000000000000000000000: a\n",
			`{"1e+36":"a"}`,
			strPtr("\"1e+36\": a\n"),
		}, {
			"1e+36: a\n",
			`{"1e+36":"a"}`,
			strPtr("\"1e+36\": a\n"),
		}, {
			"\"1e+36\": a\n",
			`{"1e+36":"a"}`,
			nil,
		}, {
			"\"1.2\": a\n",
			`{"1.2":"a"}`,
			nil,
		}, {
			"- t: a\n",
			`[{"t":"a"}]`,
			nil,
		}, {
			"- t: a\n" +
				"- t:\n" +
				"    b: 1\n" +
				"    c: 2\n",
			`[{"t":"a"},{"t":{"b":1,"c":2}}]`,
			nil,
		}, {
			`[{t: a}, {t: {b: 1, c: 2}}]`,
			`[{"t":"a"},{"t":{"b":1,"c":2}}]`,
			strPtr("- t: a\n" +
				"- t:\n" +
				"    b: 1\n" +
				"    c: 2\n"),
		}, {
			"- t: \n",
			`[{"t":null}]`,
			strPtr("- t: null\n"),
		}, {
			"- t: null\n",
			`[{"t":null}]`,
			nil,
		},
	}

	// Cases that should produce errors.
	_ = []Case{
		{
			"~: a",
			`{"null":"a"}`,
			nil,
		}, {
			"a: !!binary gIGC\n",
			"{\"a\":\"\x80\x81\x82\"}",
			nil,
		},
	}

	runCases(t, RunTypeYAMLToJSON, cases)
}

func runCases(t *testing.T, runType RunType, cases []Case) {
	var f func([]byte) ([]byte, error)
	var invF func([]byte) ([]byte, error)
	var msg string
	var invMsg string
	if runType == RunTypeJSONToYAML {
		f = JSONToYAML
		invF = YAMLToJSON
		msg = "JSON to YAML"
		invMsg = "YAML back to JSON"
	} else {
		f = YAMLToJSON
		invF = JSONToYAML
		msg = "YAML to JSON"
		invMsg = "JSON back to YAML"
	}

	for _, c := range cases {
		// Convert the string.
		t.Logf("converting %s\n", c.input)
		output, err := f([]byte(c.input))
		if err != nil {
			t.Errorf("Failed to convert %s, input: `%s`, err: %v", msg, c.input, err)
		}

		// Check it against the expected output.
		if string(output) != c.output {
			t.Errorf("Failed to convert %s, input: `%s`, expected `%s`, got `%s`",
				msg, c.input, c.output, string(output))
		}

		// Set the string that we will compare the reversed output to.
		reverse := c.input
		// If a special reverse string was specified, use that instead.
		if c.reverse != nil {
			reverse = *c.reverse
		}

		// Reverse the output.
		input, err := invF(output)
		if err != nil {
			t.Errorf("Failed to convert %s, input: `%s`, err: %v", invMsg, string(output), err)
		}

		// Check the reverse is equal to the input (or to *c.reverse).
		if string(input) != reverse {
			t.Errorf("Failed to convert %s, input: `%s`, expected `%s`, got `%s`",
				invMsg, string(output), reverse, string(input))
		}
	}

}

// To be able to easily fill in the *Case.reverse string above.
func strPtr(s string) *string {
	return &s
}

func TestYAMLToJSONStrict(t *testing.T) {
	const data = `
foo: bar
foo: baz
`
	if _, err := YAMLToJSON([]byte(data)); err != nil {
		t.Error("expected YAMLtoJSON to pass on duplicate field names")
	}
	if _, err := YAMLToJSONStrict([]byte(data)); err == nil {
		t.Error("expected YAMLtoJSONStrict to fail on duplicate field names")
	}
}

func TestJSONObjectToYAMLObject(t *testing.T) {
	intOrInt64 := func(i64 int64) interface{} {
		if i := int(i64); i64 == int64(i) {
			return i
		}
		return i64
	}

	tests := []struct {
		name     string
		input    map[string]interface{}
		expected yaml.MapSlice
	}{
		{name: "nil", expected: yaml.MapSlice(nil)},
		{name: "empty", input: map[string]interface{}{}, expected: yaml.MapSlice(nil)},
		{
			name: "values",
			input: map[string]interface{}{
				"nil slice":          []interface{}(nil),
				"nil map":            map[string]interface{}(nil),
				"empty slice":        []interface{}{},
				"empty map":          map[string]interface{}{},
				"bool":               true,
				"float64":            float64(42.1),
				"fractionless":       float64(42),
				"int":                int(42),
				"int64":              int64(42),
				"int64 big":          float64(math.Pow(2, 62)),
				"negative int64 big": -float64(math.Pow(2, 62)),
				"map":                map[string]interface{}{"foo": "bar"},
				"slice":              []interface{}{"foo", "bar"},
				"string":             string("foo"),
				"uint64 big":         float64(math.Pow(2, 63)),
			},
			expected: yaml.MapSlice{
				{Key: "nil slice"},
				{Key: "nil map"},
				{Key: "empty slice", Value: []interface{}{}},
				{Key: "empty map", Value: yaml.MapSlice(nil)},
				{Key: "bool", Value: true},
				{Key: "float64", Value: float64(42.1)},
				{Key: "fractionless", Value: int(42)},
				{Key: "int", Value: int(42)},
				{Key: "int64", Value: int(42)},
				{Key: "int64 big", Value: intOrInt64(int64(1) << 62)},
				{Key: "negative int64 big", Value: intOrInt64(-(1 << 62))},
				{Key: "map", Value: yaml.MapSlice{{Key: "foo", Value: "bar"}}},
				{Key: "slice", Value: []interface{}{"foo", "bar"}},
				{Key: "string", Value: string("foo")},
				{Key: "uint64 big", Value: uint64(1) << 63},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := JSONObjectToYAMLObject(tt.input)
			sortMapSlicesInPlace(tt.expected)
			sortMapSlicesInPlace(got)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("jsonToYAML() = %v, want %v", spew.Sdump(got), spew.Sdump(tt.expected))
			}

			jsonBytes, err := json.Marshal(tt.input)
			if err != nil {
				t.Fatalf("unexpected json.Marshal error: %v", err)
			}
			var gotByRoundtrip yaml.MapSlice
			if err := yaml.Unmarshal(jsonBytes, &gotByRoundtrip); err != nil {
				t.Fatalf("unexpected yaml.Unmarshal error: %v", err)
			}

			// yaml.Unmarshal loses precision, it's rounding to the 4th last digit.
			// Replicate this here in the test, but don't change the type.
			for i := range got {
				switch got[i].Key {
				case "int64 big", "uint64 big", "negative int64 big":
					switch v := got[i].Value.(type) {
					case int64:
						d := int64(500)
						if v < 0 {
							d = -500
						}
						got[i].Value = int64((v+d)/1000) * 1000
					case uint64:
						got[i].Value = uint64((v+500)/1000) * 1000
					case int:
						d := int(500)
						if v < 0 {
							d = -500
						}
						got[i].Value = int((v+d)/1000) * 1000
					default:
						t.Fatalf("unexpected type for key %s: %v:%T", got[i].Key, v, v)
					}
				}
			}

			if !reflect.DeepEqual(got, gotByRoundtrip) {
				t.Errorf("yaml.Unmarshal(json.Marshal(tt.input)) = %v, want %v\njson: %s", spew.Sdump(gotByRoundtrip), spew.Sdump(got), string(jsonBytes))
			}
		})
	}
}

func sortMapSlicesInPlace(x interface{}) {
	switch x := x.(type) {
	case []interface{}:
		for i := range x {
			sortMapSlicesInPlace(x[i])
		}
	case yaml.MapSlice:
		sort.Slice(x, func(a, b int) bool {
			return x[a].Key.(string) < x[b].Key.(string)
		})
	}
}
