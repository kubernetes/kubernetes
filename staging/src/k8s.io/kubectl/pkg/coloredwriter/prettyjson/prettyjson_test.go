/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package prettyjson

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/fatih/color"
)

func Example() {
	defer func(noColor bool) { color.NoColor = noColor }(color.NoColor)
	color.NoColor = false

	v := map[string]interface{}{
		"str":   "foo",
		"num":   100,
		"bool":  false,
		"null":  nil,
		"array": []string{"foo", "bar", "baz"},
		"map": map[string]interface{}{
			"foo": "bar",
		},
	}
	s, _ := Marshal(v)
	fmt.Println(string(s))
	// Output:
	// {
	//   [34;1m"array"[0m: [
	//     [32;1m"foo"[0m,
	//     [32;1m"bar"[0m,
	//     [32;1m"baz"[0m
	//   ],
	//   [34;1m"bool"[0m: [33;1mfalse[0m,
	//   [34;1m"map"[0m: {
	//     [34;1m"foo"[0m: [32;1m"bar"[0m
	//   },
	//   [34;1m"null"[0m: [30;1mnull[0m,
	//   [34;1m"num"[0m: [36;1m100[0m,
	//   [34;1m"str"[0m: [32;1m"foo"[0m
	// }
}

func TestMarshal(t *testing.T) {
	prettyJSON := func(s string) string {
		var v interface{}

		decoder := json.NewDecoder(strings.NewReader(s))
		decoder.UseNumber()
		err := decoder.Decode(&v)

		if err != nil {
			t.Error(err)
		}

		prettyJSONByte, err := Marshal(v)

		if err != nil {
			t.Error(err)
		}

		return string(prettyJSONByte)
	}

	test := func(expected, actual string) {
		if expected != actual {
			t.Errorf("\nexpected:\n%s\n\nactual:\n%s", expected, actual)
		}
	}

	blueBold := color.New(color.FgBlue, color.Bold).SprintFunc()
	greenBold := color.New(color.FgGreen, color.Bold).SprintFunc()
	cyanBold := color.New(color.FgCyan, color.Bold).SprintFunc()
	blackBold := color.New(color.FgBlack, color.Bold).SprintFunc()
	yelloBold := color.New(color.FgYellow, color.Bold).SprintFunc()

	actual := prettyJSON(`{
  "key": {
    "a": "str",
    "b": 100,
    "c": null,
    "d": true,
    "e": false,
    "f": { "key": "str" },
	"g": {},
	"h": []
  }
}`)

	expectedFormat := `{
  %s: {
    %s: %s,
    %s: %s,
    %s: %s,
    %s: %s,
    %s: %s,
    %s: {
      %s: %s
    },
    %s: {},
    %s: []
  }
}`

	expected := fmt.Sprintf(expectedFormat,
		blueBold(`"key"`),
		blueBold(`"a"`), greenBold(`"str"`),
		blueBold(`"b"`), cyanBold("100"),
		blueBold(`"c"`), blackBold("null"),
		blueBold(`"d"`), yelloBold("true"),
		blueBold(`"e"`), yelloBold("false"),
		blueBold(`"f"`), blueBold(`"key"`), greenBold(`"str"`),
		blueBold(`"g"`),
		blueBold(`"h"`),
	)

	test(expected, actual)
	test("{}", prettyJSON("{}"))
	test("[]", prettyJSON("[]"))

	test(
		fmt.Sprintf("{\n  %s: %s\n}", blueBold(`"x"`), cyanBold("123456789123456789123456789")),
		prettyJSON(`{"x":123456789123456789123456789}`),
	)

	test(
		fmt.Sprintf("{\n  %s: %s\n}", blueBold(`"foo\"bar\n\r\t<>★"`), cyanBold("1")),
		prettyJSON(`{"foo\"bar\n\r\t<>★":1}`),
	)
}

func TestStringEscape(t *testing.T) {
	f := NewFormatter()
	f.DisabledColor = true
	s := `{"foo":"foo\"\nbar"}`
	r, err := f.Format([]byte(s))

	if err != nil {
		t.Error(err)
	}

	expected := `{
  "foo": "foo\"\nbar"
}`

	if string(r) != expected {
		t.Errorf("actual: %s\nexpected: %s", string(r), expected)
	}
}

func TestStringPercentEscape(t *testing.T) {
	f := NewFormatter()
	s := `{"foo":"foo%2Fbar"}`
	r, err := f.Format([]byte(s))

	if err != nil {
		t.Error(err)
	}

	expectedFormat := `{
  %s: %s
}`

	blueBold := color.New(color.FgBlue, color.Bold).SprintFunc()
	greenBold := color.New(color.FgGreen, color.Bold).SprintFunc()

	expected := fmt.Sprintf(expectedFormat,
		blueBold(`"foo"`), greenBold(`"foo%2Fbar"`),
	)

	if string(r) != expected {
		t.Errorf("actual: %s\nexpected: %s", string(r), expected)
	}
}

func TestStringPercentEscape_DisabledColor(t *testing.T) {
	f := NewFormatter()
	f.DisabledColor = true
	s := `{"foo":"foo%2Fbar"}`
	r, err := f.Format([]byte(s))

	if err != nil {
		t.Error(err)
	}

	expected := `{
  "foo": "foo%2Fbar"
}`

	if string(r) != expected {
		t.Errorf("actual: %s\nexpected: %s", string(r), expected)
	}
}

func BenchmarkFromat(b *testing.B) {
	s := []byte(`{"string": "a", "number": 3, "array": [1, 2, 3], "map": {"map": "value"}, "emptyArray": [], "emptyMap": {}}`)
	f := NewFormatter()

	if _, err := f.Format(s); err != nil {
		b.Fatal(err)
	}
	for i := 0; i < b.N; i++ {
		f.Format(s)
	}
}
