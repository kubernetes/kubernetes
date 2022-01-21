// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package decls

import exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"

// Scopes represents nested Decl sets where the Scopes value contains a Groups containing all
// identifiers in scope and an optional parent representing outer scopes.
// Each Groups value is a mapping of names to Decls in the ident and function namespaces.
// Lookups are performed such that bindings in inner scopes shadow those in outer scopes.
type Scopes struct {
	parent *Scopes
	scopes *Group
}

// NewScopes creates a new, empty Scopes.
// Some operations can't be safely performed until a Group is added with Push.
func NewScopes() *Scopes {
	return &Scopes{
		scopes: newGroup(),
	}
}

// Push creates a new Scopes value which references the current Scope as its parent.
func (s *Scopes) Push() *Scopes {
	return &Scopes{
		parent: s,
		scopes: newGroup(),
	}
}

// Pop returns the parent Scopes value for the current scope, or the current scope if the parent
// is nil.
func (s *Scopes) Pop() *Scopes {
	if s.parent != nil {
		return s.parent
	}
	// TODO: Consider whether this should be an error / panic.
	return s
}

// AddIdent adds the ident Decl in the current scope.
// Note: If the name collides with an existing identifier in the scope, the Decl is overwritten.
func (s *Scopes) AddIdent(decl *exprpb.Decl) {
	s.scopes.idents[decl.Name] = decl
}

// FindIdent finds the first ident Decl with a matching name in Scopes, or nil if one cannot be
// found.
// Note: The search is performed from innermost to outermost.
func (s *Scopes) FindIdent(name string) *exprpb.Decl {
	if ident, found := s.scopes.idents[name]; found {
		return ident
	}
	if s.parent != nil {
		return s.parent.FindIdent(name)
	}
	return nil
}

// FindIdentInScope finds the first ident Decl with a matching name in the current Scopes value, or
// nil if one does not exist.
// Note: The search is only performed on the current scope and does not search outer scopes.
func (s *Scopes) FindIdentInScope(name string) *exprpb.Decl {
	if ident, found := s.scopes.idents[name]; found {
		return ident
	}
	return nil
}

// AddFunction adds the function Decl to the current scope.
// Note: Any previous entry for a function in the current scope with the same name is overwritten.
func (s *Scopes) AddFunction(fn *exprpb.Decl) {
	s.scopes.functions[fn.Name] = fn
}

// FindFunction finds the first function Decl with a matching name in Scopes.
// The search is performed from innermost to outermost.
// Returns nil if no such function in Scopes.
func (s *Scopes) FindFunction(name string) *exprpb.Decl {
	if fn, found := s.scopes.functions[name]; found {
		return fn
	}
	if s.parent != nil {
		return s.parent.FindFunction(name)
	}
	return nil
}

// Group is a set of Decls that is pushed on or popped off a Scopes as a unit.
// Contains separate namespaces for idenifier and function Decls.
// (Should be named "Scope" perhaps?)
type Group struct {
	idents    map[string]*exprpb.Decl
	functions map[string]*exprpb.Decl
}

func newGroup() *Group {
	return &Group{
		idents:    make(map[string]*exprpb.Decl),
		functions: make(map[string]*exprpb.Decl),
	}
}
