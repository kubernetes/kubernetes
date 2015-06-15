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

import "fmt"

// NodeType identifies the type of a parse tree node.
type NodeType int

// Type returns itself and provides an easy default implementation
func (t NodeType) Type() NodeType {
	return t
}

func (t NodeType) String() string {
	return NodeTypeName[t]
}

const (
	NodeText NodeType = iota
	NodeArray
	NodeList
	NodeField
	NodeFilter
)

var NodeTypeName = map[NodeType]string{
	NodeText:   "NodeText",
	NodeArray:  "NodeArray",
	NodeList:   "NodeList",
	NodeField:  "NodeField",
	NodeFilter: "NodeFilter",
}

type Node interface {
	Type() NodeType
	String() string
}

// ListNode holds a sequence of nodes.
type ListNode struct {
	NodeType
	Nodes []Node // The element nodes in lexical order.
}

func newList() *ListNode {
	return &ListNode{NodeType: NodeList}
}

func (l *ListNode) append(n Node) {
	l.Nodes = append(l.Nodes, n)
}

func (l *ListNode) String() string {
	return fmt.Sprintf("%s", l.Type())
}

// TextNode holds plain text.
type TextNode struct {
	NodeType
	Text []byte // The text; may span newlines.
}

func newText(text string) *TextNode {
	return &TextNode{NodeType: NodeText, Text: []byte(text)}
}

func (t *TextNode) String() string {
	return fmt.Sprintf("%s: %s", t.Type(), t.Text)
}

// FieldNode holds filed of struct
type FieldNode struct {
	NodeType
	Value string
}

func newField(value string) *FieldNode {
	return &FieldNode{NodeType: NodeField, Value: value}
}

func (f *FieldNode) String() string {
	return fmt.Sprintf("%s: %s", f.Type(), f.Value)
}

type ArrayNode struct {
	NodeType
	Params [3]int //start, end, step
	Exist  [3]bool
}

func newArray(params [3]int, exist [3]bool) *ArrayNode {
	return &ArrayNode{
		NodeType: NodeArray,
		Params:   params,
		Exist:    exist,
	}
}

func (a *ArrayNode) String() string {
	return fmt.Sprintf("%s: %v %v", a.Type(), a.Params, a.Exist)
}

type FilterNode struct {
	NodeType
	Left, Operator, Right string
}

func newFilter(left, operator, right string) *FilterNode {
	return &FilterNode{
		NodeType: NodeFilter,
		Left:     left,
		Operator: operator,
		Right:    right,
	}
}

func (f *FilterNode) String() string {
	return fmt.Sprintf("%s: %s %s %s", f.Type(), f.Left, f.Operator, f.Right)
}
