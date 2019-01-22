/*
Copyright 2016 Google Inc. All Rights Reserved.

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

// Rule-level API for inspecting and modifying a build.File syntax tree.

package build

import "strings"

// A Rule represents a single BUILD rule.
type Rule struct {
	Call *CallExpr
}

// Rules returns the rules in the file of the given kind (such as "go_library").
// If kind == "", Rules returns all rules in the file.
func (f *File) Rules(kind string) []*Rule {
	var all []*Rule
	for _, stmt := range f.Stmt {
		call, ok := stmt.(*CallExpr)
		if !ok {
			continue
		}
		rule := &Rule{call}
		if kind != "" && rule.Kind() != kind {
			continue
		}
		all = append(all, rule)
	}
	return all
}

// RuleAt returns the rule in the file that starts at the specified line, or null if no such rule.
func (f *File) RuleAt(linenum int) *Rule {
	for _, stmt := range f.Stmt {
		call, ok := stmt.(*CallExpr)
		if !ok {
			continue
		}
		start, end := call.X.Span()
		if start.Line <= linenum && linenum <= end.Line {
			return &Rule{call}
		}
	}
	return nil
}

// DelRules removes rules with the given kind and name from the file.
// An empty kind matches all kinds; an empty name matches all names.
// It returns the number of rules that were deleted.
func (f *File) DelRules(kind, name string) int {
	var i int
	for _, stmt := range f.Stmt {
		if call, ok := stmt.(*CallExpr); ok {
			r := &Rule{call}
			if (kind == "" || r.Kind() == kind) &&
				(name == "" || r.AttrString("name") == name) {
				continue
			}
		}
		f.Stmt[i] = stmt
		i++
	}
	n := len(f.Stmt) - i
	f.Stmt = f.Stmt[:i]
	return n
}

// Kind returns the rule's kind (such as "go_library").
// The kind of the rule may be given by a literal or it may be a sequence of dot expressions that
// begins with a literal, if the call expression does not conform to either of these forms, an
// empty string will be returned
func (r *Rule) Kind() string {
	var names []string
	expr := r.Call.X
	for {
		x, ok := expr.(*DotExpr)
		if !ok {
			break
		}
		names = append(names, x.Name)
		expr = x.X
	}
	x, ok := expr.(*LiteralExpr)
	if !ok {
		return ""
	}
	names = append(names, x.Token)
	// Reverse the elements since the deepest expression contains the leading literal
	for l, r := 0, len(names)-1; l < r; l, r = l+1, r-1 {
		names[l], names[r] = names[r], names[l]
	}
	return strings.Join(names, ".")
}

// SetKind changes rule's kind (such as "go_library").
func (r *Rule) SetKind(kind string) {
	names := strings.Split(kind, ".")
	var expr Expr
	expr = &LiteralExpr{Token: names[0]}
	for _, name := range names[1:] {
		expr = &DotExpr{X: expr, Name: name}
	}
	r.Call.X = expr
}

// Name returns the rule's target name.
// If the rule has no target name, Name returns the empty string.
func (r *Rule) Name() string {
	return r.AttrString("name")
}

// AttrKeys returns the keys of all the rule's attributes.
func (r *Rule) AttrKeys() []string {
	var keys []string
	for _, expr := range r.Call.List {
		if binExpr, ok := expr.(*BinaryExpr); ok && binExpr.Op == "=" {
			if keyExpr, ok := binExpr.X.(*LiteralExpr); ok {
				keys = append(keys, keyExpr.Token)
			}
		}
	}
	return keys
}

// AttrDefn returns the BinaryExpr defining the rule's attribute with the given key.
// That is, the result is a *BinaryExpr with Op == "=".
// If the rule has no such attribute, AttrDefn returns nil.
func (r *Rule) AttrDefn(key string) *BinaryExpr {
	for _, kv := range r.Call.List {
		as, ok := kv.(*BinaryExpr)
		if !ok || as.Op != "=" {
			continue
		}
		k, ok := as.X.(*LiteralExpr)
		if !ok || k.Token != key {
			continue
		}
		return as
	}
	return nil
}

// Attr returns the value of the rule's attribute with the given key
// (such as "name" or "deps").
// If the rule has no such attribute, Attr returns nil.
func (r *Rule) Attr(key string) Expr {
	as := r.AttrDefn(key)
	if as == nil {
		return nil
	}
	return as.Y
}

// DelAttr deletes the rule's attribute with the named key.
// It returns the old value of the attribute, or nil if the attribute was not found.
func (r *Rule) DelAttr(key string) Expr {
	list := r.Call.List
	for i, kv := range list {
		as, ok := kv.(*BinaryExpr)
		if !ok || as.Op != "=" {
			continue
		}
		k, ok := as.X.(*LiteralExpr)
		if !ok || k.Token != key {
			continue
		}
		copy(list[i:], list[i+1:])
		r.Call.List = list[:len(list)-1]
		return as.Y
	}
	return nil
}

// SetAttr sets the rule's attribute with the given key to value.
// If the rule has no attribute with the key, SetAttr appends
// one to the end of the rule's attribute list.
func (r *Rule) SetAttr(key string, val Expr) {
	as := r.AttrDefn(key)
	if as != nil {
		as.Y = val
		return
	}

	r.Call.List = append(r.Call.List,
		&BinaryExpr{
			X:  &LiteralExpr{Token: key},
			Op: "=",
			Y:  val,
		},
	)
}

// AttrLiteral returns the literal form of the rule's attribute
// with the given key (such as "cc_api_version"), only when
// that value is an identifier or number.
// If the rule has no such attribute or the attribute is not an identifier or number,
// AttrLiteral returns "".
func (r *Rule) AttrLiteral(key string) string {
	lit, ok := r.Attr(key).(*LiteralExpr)
	if !ok {
		return ""
	}
	return lit.Token
}

// AttrString returns the value of the rule's attribute
// with the given key (such as "name"), as a string.
// If the rule has no such attribute or the attribute has a non-string value,
// Attr returns the empty string.
func (r *Rule) AttrString(key string) string {
	str, ok := r.Attr(key).(*StringExpr)
	if !ok {
		return ""
	}
	return str.Value
}

// AttrStrings returns the value of the rule's attribute
// with the given key (such as "srcs"), as a []string.
// If the rule has no such attribute or the attribute is not
// a list of strings, AttrStrings returns a nil slice.
func (r *Rule) AttrStrings(key string) []string {
	return Strings(r.Attr(key))
}

// Strings returns expr as a []string.
// If expr is not a list of string literals,
// Strings returns a nil slice instead.
// If expr is an empty list of string literals,
// returns a non-nil empty slice.
// (this allows differentiating between these two cases)
func Strings(expr Expr) []string {
	list, ok := expr.(*ListExpr)
	if !ok {
		return nil
	}
	all := []string{} // not nil
	for _, l := range list.List {
		str, ok := l.(*StringExpr)
		if !ok {
			return nil
		}
		all = append(all, str.Value)
	}
	return all
}
