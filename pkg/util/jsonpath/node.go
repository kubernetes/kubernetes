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

// NodeType identifies the type of a parse tree node.
type NodeType int

type Node interface {
	Type() NodeType
}

// Type returns itself and provides an easy default implementation
// for embedding in a Node. Embedded in all non-trivial Nodes.
func (t NodeType) Type() NodeType {
	return t
}

const (
	NodeText NodeType = iota
	NodeArray
	NodeList
	NodeField
)

// ListNode holds a sequence of nodes.
type ListNode struct {
	NodeType
	Nodes []Node // The element nodes in lexical order.
}

func newList() *ListNode {
	return &ListNode{NodeType: NodeList}
}

func (l *ListNode) Type() NodeType {
	return l.NodeType
}

func (l *ListNode) append(n Node) {
	l.Nodes = append(l.Nodes, n)
}

// TextNode holds plain text.
type TextNode struct {
	NodeType
	Text []byte // The text; may span newlines.
}

func newText(text string) *TextNode {
	return &TextNode{NodeType: NodeText, Text: []byte(text)}
}

// FieldNode holds filed of struct
type FieldNode struct {
	NodeType
	Value string
}

func newField(value string) *FieldNode {
	return &FieldNode{NodeType: NodeField, Value: value}
}

type ArrayNode struct {
	NodeType
	Value string
}

func newArray(value string) *ArrayNode {
	return &ArrayNode{NodeType: NodeArray, Value: value}
}
