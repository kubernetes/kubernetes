// Package ast declares the types used to represent syntax trees for HCL
// (HashiCorp Configuration Language)
package ast

import (
	"fmt"
	"strings"

	"github.com/hashicorp/hcl/hcl/token"
)

// Node is an element in the abstract syntax tree.
type Node interface {
	node()
	Pos() token.Pos
}

func (File) node()         {}
func (ObjectList) node()   {}
func (ObjectKey) node()    {}
func (ObjectItem) node()   {}
func (Comment) node()      {}
func (CommentGroup) node() {}
func (ObjectType) node()   {}
func (LiteralType) node()  {}
func (ListType) node()     {}

// File represents a single HCL file
type File struct {
	Node     Node            // usually a *ObjectList
	Comments []*CommentGroup // list of all comments in the source
}

func (f *File) Pos() token.Pos {
	return f.Node.Pos()
}

// ObjectList represents a list of ObjectItems. An HCL file itself is an
// ObjectList.
type ObjectList struct {
	Items []*ObjectItem
}

func (o *ObjectList) Add(item *ObjectItem) {
	o.Items = append(o.Items, item)
}

// Filter filters out the objects with the given key list as a prefix.
//
// The returned list of objects contain ObjectItems where the keys have
// this prefix already stripped off. This might result in objects with
// zero-length key lists if they have no children.
//
// If no matches are found, an empty ObjectList (non-nil) is returned.
func (o *ObjectList) Filter(keys ...string) *ObjectList {
	var result ObjectList
	for _, item := range o.Items {
		// If there aren't enough keys, then ignore this
		if len(item.Keys) < len(keys) {
			continue
		}

		match := true
		for i, key := range item.Keys[:len(keys)] {
			key := key.Token.Value().(string)
			if key != keys[i] && !strings.EqualFold(key, keys[i]) {
				match = false
				break
			}
		}
		if !match {
			continue
		}

		// Strip off the prefix from the children
		newItem := *item
		newItem.Keys = newItem.Keys[len(keys):]
		result.Add(&newItem)
	}

	return &result
}

// Children returns further nested objects (key length > 0) within this
// ObjectList. This should be used with Filter to get at child items.
func (o *ObjectList) Children() *ObjectList {
	var result ObjectList
	for _, item := range o.Items {
		if len(item.Keys) > 0 {
			result.Add(item)
		}
	}

	return &result
}

// Elem returns items in the list that are direct element assignments
// (key length == 0). This should be used with Filter to get at elements.
func (o *ObjectList) Elem() *ObjectList {
	var result ObjectList
	for _, item := range o.Items {
		if len(item.Keys) == 0 {
			result.Add(item)
		}
	}

	return &result
}

func (o *ObjectList) Pos() token.Pos {
	// always returns the uninitiliazed position
	return o.Items[0].Pos()
}

// ObjectItem represents a HCL Object Item. An item is represented with a key
// (or keys). It can be an assignment or an object (both normal and nested)
type ObjectItem struct {
	// keys is only one length long if it's of type assignment. If it's a
	// nested object it can be larger than one. In that case "assign" is
	// invalid as there is no assignments for a nested object.
	Keys []*ObjectKey

	// assign contains the position of "=", if any
	Assign token.Pos

	// val is the item itself. It can be an object,list, number, bool or a
	// string. If key length is larger than one, val can be only of type
	// Object.
	Val Node

	LeadComment *CommentGroup // associated lead comment
	LineComment *CommentGroup // associated line comment
}

func (o *ObjectItem) Pos() token.Pos {
	return o.Keys[0].Pos()
}

// ObjectKeys are either an identifier or of type string.
type ObjectKey struct {
	Token token.Token
}

func (o *ObjectKey) Pos() token.Pos {
	return o.Token.Pos
}

// LiteralType represents a literal of basic type. Valid types are:
// token.NUMBER, token.FLOAT, token.BOOL and token.STRING
type LiteralType struct {
	Token token.Token

	// associated line comment, only when used in a list
	LineComment *CommentGroup
}

func (l *LiteralType) Pos() token.Pos {
	return l.Token.Pos
}

// ListStatement represents a HCL List type
type ListType struct {
	Lbrack token.Pos // position of "["
	Rbrack token.Pos // position of "]"
	List   []Node    // the elements in lexical order
}

func (l *ListType) Pos() token.Pos {
	return l.Lbrack
}

func (l *ListType) Add(node Node) {
	l.List = append(l.List, node)
}

// ObjectType represents a HCL Object Type
type ObjectType struct {
	Lbrace token.Pos   // position of "{"
	Rbrace token.Pos   // position of "}"
	List   *ObjectList // the nodes in lexical order
}

func (o *ObjectType) Pos() token.Pos {
	return o.Lbrace
}

// Comment node represents a single //, # style or /*- style commment
type Comment struct {
	Start token.Pos // position of / or #
	Text  string
}

func (c *Comment) Pos() token.Pos {
	return c.Start
}

// CommentGroup node represents a sequence of comments with no other tokens and
// no empty lines between.
type CommentGroup struct {
	List []*Comment // len(List) > 0
}

func (c *CommentGroup) Pos() token.Pos {
	return c.List[0].Pos()
}

//-------------------------------------------------------------------
// GoStringer
//-------------------------------------------------------------------

func (o *ObjectKey) GoString() string { return fmt.Sprintf("*%#v", *o) }
