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

import "testing"

func TestParsePlainText(t *testing.T) {
	parser, err := Parse("plain", "hello jsonpath")
	if err != nil {
		t.Errorf("parse plain text error %v", err)
	}
	nodes := parser.Root.Nodes
	if len(nodes) != 1 {
		t.Errorf("expect one nodes, got %v", len(nodes))
	}
	if nodes[0].Type() != NodeText {
		t.Errorf("expect %v, got %v")
	}
}

func TestParseVariable(t *testing.T) {
	parser, err := Parse("variable", "hello ${.jsonpath}")
	if err != nil {
		t.Errorf("parse variable error %v", err)
	}
	nodes := parser.Root.Nodes
	if len(nodes) != 2 {
		t.Errorf("expect two nodes, got %v", len(nodes))
	}
	if nodes[0].Type() != NodeText {
		t.Errorf("expect NodeText, got %v", nodes[0])
	}
	if nodes[1].Type() != NodeList {
		t.Errorf("expect NodeList, got %v", nodes[1])
	}
	nodes = nodes[1].(*ListNode).Nodes
	node := nodes[0].(*FieldNode)
	if node.Value != "jsonpath" {
		t.Errorf("expect NodeVariable jsonpath, got %s", node.Value)
	}
}

func TestParseQuote(t *testing.T) {
	parser, err := Parse("variable", `hello ${"${"}`)
	if err != nil {
		t.Errorf("parse quote error %v", err)
	}
	nodes := parser.Root.Nodes
	if len(nodes) != 2 {
		t.Errorf("expect two nodes, got %v", len(nodes))
	}
	if nodes[0].Type() != NodeText {
		t.Errorf("expect NodeText, got %v", nodes[0])
	}
	if nodes[1].Type() != NodeList {
		t.Errorf("expect NodeList, got %v", nodes[1])
	}
	nodes = nodes[1].(*ListNode).Nodes
	node := nodes[0].(*TextNode)
	if string(node.Text[:]) != "${" {
		t.Errorf("expect ${, got %s", node.Text)
	}
}

func TestParseArray(t *testing.T) {
	parser, err := Parse("array", "hello ${[1..3]}")
	if err != nil {
		t.Errorf("parse quote error %v", err)
	}
	nodes := parser.Root.Nodes
	if len(nodes) != 2 {
		t.Errorf("expect two nodes, got %v", len(nodes))
	}
	if nodes[0].Type() != NodeText {
		t.Errorf("expect NodeText, got %v", nodes[0])
	}
	if nodes[1].Type() != NodeList {
		t.Errorf("expect NodeList, got %v", nodes[1])
	}
	nodes = nodes[1].(*ListNode).Nodes
	node := nodes[0].(*ArrayNode)
	if string(node.Value) != "1..3" {
		t.Errorf("expect ${, got %s", node.Value)
	}
}
