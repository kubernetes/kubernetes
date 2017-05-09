package parse

import "testing"
func TestParsePlainText(t *testing.T) {
	tree, err := Parse("plain", "hello jsonpath")
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
	tree, err := Parse("variable", "hello ${.jsonpath}")
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
	if nodes[1].Type() != NodeList {
		t.Errorf("expect NodeList, got %v", nodes[1])
	}
	nodes = nodes[1].(*ListNode).Nodes
	node := nodes[0].(*VariableNode)
	if node.Name != "jsonpath" {
		t.Errorf("expect NodeVariable jsonpath, got %s", node.Name)
	}
}

func TestParseQuote(t *testing.T) {
	tree, err := Parse("variable", `hello ${"${"}`)
	if err != nil {
		t.Errorf("parse quote error %v", err)
	}
	nodes := tree.Root.Nodes
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
