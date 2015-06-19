/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package jsonpath

import (
	"bytes"
	"testing"
)

type jsonpathTest struct {
	name     string
	template string
	input    interface{}
	expect   string
}

var jsonpathTests = []jsonpathTest{
	{"plain", "hello jsonpath", nil, "hello jsonpath"},
	{"variable", "hello ${.v}", A{v: "world"}, "hello world"},
	{"dict", "${.key}", map[string]string{"key": "value"}, "value"},
	{"nest", "hello ${.a.v}", B{a: A{v: "world"}}, "hello world"},
	{"quote", `${"${"}`, nil, "${"},
	{"array", "${[0:2]}", []string{"Monday", "Tudesday"}, "[Monday, Tudesday]"},
	{"filter", "${[?(@<5)]}", []int{2, 6, 3, 7}, "[2, 3]"},
}

// only for test use
type A struct {
	v string
}

// only for test use
type B struct {
	v string
	a A
}

func TestJSONPath(t *testing.T) {
	for _, test := range jsonpathTests {
		j := NewJSONPath(test.name)
		err := j.Parse(test.template)
		if err != nil {
			t.Errorf("in %s, parse %s error %v", test.name, test.template, err)
		}
		buf := new(bytes.Buffer)
		err = j.Execute(buf, test.input)
		if err != nil {
			t.Errorf("in %s, execute error %v", test.name, err)
		}
		out := buf.String()
		if out != test.expect {
			t.Errorf("in %s, expect to get %s, got %s", test.name, test.expect, out)
		}
	}
}
