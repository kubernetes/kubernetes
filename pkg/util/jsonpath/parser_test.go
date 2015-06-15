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
	"testing"
)

type parserTest struct {
	name  string
	text  string
	nodes []Node
}

var parserTests = []parserTest{
	{"plain", `hello jsonpath`,
		[]Node{newText("hello jsonpath")}},
	{"variable", `hello ${.jsonpath}`,
		[]Node{newText("hello "), newList(), newField("jsonpath")}},
	{"quote", `hello ${"${"}`,
		[]Node{newText("hello "), newList(), newText("${")}},
	{"array", `hello ${[1:3]}`,
		[]Node{newText("hello "), newList(), newArray([3]int{1, 3, 0}, [3]bool{true, true, false})}},
	{"filter", `${[?(@.price<3)]}`,
		[]Node{newList(), newFilter("@.price", "<", "3")}},
}

func collectNode(nodes []Node, cur Node) []Node {
	nodes = append(nodes, cur)
	if cur.Type() == NodeList {
		for _, node := range cur.(*ListNode).Nodes {
			nodes = collectNode(nodes, node)
		}
	}
	return nodes
}

func TestParser(t *testing.T) {
	for _, test := range parserTests {
		parser, err := Parse(test.name, test.text)
		if err != nil {
			t.Errorf("parse %s error %v", test.name, err)
		}
		result := collectNode([]Node{}, parser.Root)[1:]
		if len(result) != len(test.nodes) {
			t.Errorf("in %s, expect to get %d nodes, got %d nodes", test.name, len(test.nodes), len(result))
		}
		for i, expect := range test.nodes {
			if result[i].String() != expect.String() {
				t.Errorf("in %s, expect %v, got %v", test.name, expect, result[i])
			}
		}
	}
}
