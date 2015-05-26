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

package jq

import "testing"

func TestParsePlainText(t *testing.T) {
	tree, err := Parse("plain", "hello jq")
	if err != nil {
		t.Errorf("parse plain text error %v", err)
	}
	nodes := tree.Root.Nodes
	if len(nodes) != 1 {
		t.Errorf("expect one nodes, got %v", len(nodes))
	}
	if nodes[0].Type() != NodeText {
		t.Errorf("expect %v, got %v")
	}
}

func TestParseVariable(t *testing.T) {
	tree, err := Parse("variable", "hello '.jq'")
	if err != nil {
		t.Errorf("parse plain text error %v", err)
	}
	nodes := tree.Root.Nodes
	if len(nodes) != 2 {
		t.Errorf("expect two nodes, got %v", len(nodes))
	}
	if nodes[0].Type() != NodeText {
		t.Errorf("expect NodeText, got %v", nodes[0])
	}
	if nodes[1].Type() != NodeVariable {
		t.Errorf("expect NodeVariable, got %v", nodes[1])
	}
	node := nodes[1].(*VariableNode)
	if node.Name != "jq" {
		t.Errorf("expect NodeVariable jq, got %s", node.Name)
	}
}
