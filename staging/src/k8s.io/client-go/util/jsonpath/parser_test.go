/*
Copyright 2015 The Kubernetes Authors.

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
	name        string
	text        string
	nodes       []Node
	shouldError bool
}

var parserTests = []parserTest{
	{"plain", `hello jsonpath`, []Node{newText("hello jsonpath")}, false},
	{"variable", `hello {.jsonpath}`,
		[]Node{newText("hello "), newList(), newField("jsonpath")}, false},
	{"arrayfiled", `hello {['jsonpath']}`,
		[]Node{newText("hello "), newList(), newField("jsonpath")}, false},
	{"quote", `{"{"}`, []Node{newList(), newText("{")}, false},
	{"array", `{[1:3]}`, []Node{newList(),
		newArray([3]ParamsEntry{{1, true, false}, {3, true, false}, {0, false, false}})}, false},
	{"allarray", `{.book[*].author}`,
		[]Node{newList(), newField("book"),
			newArray([3]ParamsEntry{{0, false, false}, {0, false, false}, {0, false, false}}), newField("author")}, false},
	{"wildcard", `{.bicycle.*}`,
		[]Node{newList(), newField("bicycle"), newWildcard()}, false},
	{"filter", `{[?(@.price<3)]}`,
		[]Node{newList(), newFilter(newList(), newList(), "<"),
			newList(), newField("price"), newList(), newInt(3)}, false},
	{"recursive", `{..}`, []Node{newList(), newRecursive()}, false},
	{"recurField", `{..price}`,
		[]Node{newList(), newRecursive(), newField("price")}, false},
	{"arraydict", `{['book.price']}`, []Node{newList(),
		newField("book"), newField("price"),
	}, false},
	{"union", `{['bicycle.price', 3, 'book.price']}`, []Node{newList(), newUnion([]*ListNode{}),
		newList(), newField("bicycle"), newField("price"),
		newList(), newArray([3]ParamsEntry{{3, true, false}, {4, true, true}, {0, false, false}}),
		newList(), newField("book"), newField("price"),
	}, false},
	{"range", `{range .items}{.name},{end}`, []Node{
		newList(), newIdentifier("range"), newField("items"),
		newList(), newField("name"), newText(","),
		newList(), newIdentifier("end"),
	}, false},
	{"malformat input", `{\\\}`, []Node{}, true},
	{"paired parentheses in quotes", `{[?(@.status.nodeInfo.osImage == "()")]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("()")}, false},
	{"paired parentheses in double quotes and with double quotes escape", `{[?(@.status.nodeInfo.osImage == "(\"\")")]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("(\"\")")}, false},
	{"unregular parentheses in double quotes", `{[?(@.test == "())(")]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("test"), newList(), newText("())(")}, false},
	{"plain text in single quotes", `{[?(@.status.nodeInfo.osImage == 'Linux')]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("Linux")}, false},
	{"test filter suffix", `{[?(@.status.nodeInfo.osImage == "{[()]}")]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("{[()]}")}, false},
	{"double inside single", `{[?(@.status.nodeInfo.osImage == "''")]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("''")}, false},
	{"single inside double", `{[?(@.status.nodeInfo.osImage == '""')]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("\"\"")}, false},
	{"single containing escaped single", `{[?(@.status.nodeInfo.osImage == '\\\'')]}`,
		[]Node{newList(), newFilter(newList(), newList(), "=="), newList(), newField("status"), newField("nodeInfo"), newField("osImage"), newList(), newText("\\'")}, false},
	{"negative index slice, equals a[len-5] to a[len-1]", `{[-5:]}`, []Node{newList(),
		newArray([3]ParamsEntry{{-5, true, false}, {0, false, false}, {0, false, false}})}, false},
	{"negative index slice, equals a[len-1]", `{[-1]}`, []Node{newList(),
		newArray([3]ParamsEntry{{-1, true, false}, {0, true, true}, {0, false, false}})}, false},
	{"negative index slice, equals a[1] to a[len-1]", `{[1:-1]}`, []Node{newList(),
		newArray([3]ParamsEntry{{1, true, false}, {-1, true, false}, {0, false, false}})}, false},
}

func collectNode(nodes []Node, cur Node) []Node {
	nodes = append(nodes, cur)
	switch cur.Type() {
	case NodeList:
		for _, node := range cur.(*ListNode).Nodes {
			nodes = collectNode(nodes, node)
		}
	case NodeFilter:
		nodes = collectNode(nodes, cur.(*FilterNode).Left)
		nodes = collectNode(nodes, cur.(*FilterNode).Right)
	case NodeUnion:
		for _, node := range cur.(*UnionNode).Nodes {
			nodes = collectNode(nodes, node)
		}
	}
	return nodes
}

func TestParser(t *testing.T) {
	for _, test := range parserTests {
		parser, err := Parse(test.name, test.text)
		if test.shouldError {
			if err == nil {
				t.Errorf("unexpected non-error when parsing %s", test.name)
			}
			continue
		}
		if err != nil {
			t.Errorf("parse %s error %v", test.name, err)
		}
		result := collectNode([]Node{}, parser.Root)[1:]
		if len(result) != len(test.nodes) {
			t.Errorf("in %s, expect to get %d nodes, got %d nodes", test.name, len(test.nodes), len(result))
			t.Error(result)
		}
		for i, expect := range test.nodes {
			if result[i].String() != expect.String() {
				t.Errorf("in %s, %dth node, expect %v, got %v", test.name, i, expect, result[i])
			}
		}
	}
}

type failParserTest struct {
	name string
	text string
	err  string
}

func TestFailParser(t *testing.T) {
	failParserTests := []failParserTest{
		{"unclosed action", "{.hello", "unclosed action"},
		{"unrecognized character", "{*}", "unrecognized character in action: U+002A '*'"},
		{"invalid number", "{+12.3.0}", "cannot parse number +12.3.0"},
		{"unterminated array", "{[1}", "unterminated array"},
		{"unterminated filter", "{[?(.price]}", "unterminated filter"},
		{"invalid multiple recursive descent", "{........}", "invalid multiple recursive descent"},
	}
	for _, test := range failParserTests {
		_, err := Parse(test.name, test.text)
		var out string
		if err == nil {
			out = "nil"
		} else {
			out = err.Error()
		}
		if out != test.err {
			t.Errorf("in %s, expect to get error %v, got %v", test.name, test.err, out)
		}
	}
}
