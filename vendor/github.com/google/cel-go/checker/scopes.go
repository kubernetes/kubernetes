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

package checker

import (
	"github.com/google/cel-go/common/decls"
)

// Scopes represents nested Decl sets where the Scopes value contains a Groups containing all
// identifiers in scope and an optional parent representing outer scopes.
// Each Groups value is a mapping of names to Decls in the ident and function namespaces.
// Lookups are performed such that bindings in inner scopes shadow those in outer scopes.
type Scopes struct {
	parent *Scopes
	scopes *Group
}

// newScopes creates a new, empty Scopes.
// Some operations can't be safely performed until a Group is added with Push.
func newScopes() *Scopes {
	return &Scopes{
		scopes: newGroup(),
	}
}

// Copy creates a copy of the current Scopes values, including a copy of its parent if non-nil.
func (s *Scopes) Copy() *Scopes {
	cpy := newScopes()
	if s == nil {
		return cpy
	}
	if s.parent != nil {
		cpy.parent = s.parent.Copy()
	}
	cpy.scopes = s.scopes.copy()
	return cpy
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
func (s *Scopes) AddIdent(decl *decls.VariableDecl) {
	s.scopes.idents[decl.Name()] = decl
}

// FindIdent finds the first ident Decl with a matching name in Scopes, or nil if one cannot be
// found.
// Note: The search is performed from innermost to outermost.
func (s *Scopes) FindIdent(name string) *decls.VariableDecl {
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
func (s *Scopes) FindIdentInScope(name string) *decls.VariableDecl {
	if ident, found := s.scopes.idents[name]; found {
		return ident
	}
	return nil
}

// SetFunction adds the function Decl to the current scope.
// Note: Any previous entry for a function in the current scope with the same name is overwritten.
func (s *Scopes) SetFunction(fn *decls.FunctionDecl) {
	s.scopes.functions[fn.Name()] = fn
}

// FindFunction finds the first function Decl with a matching name in Scopes.
// The search is performed from innermost to outermost.
// Returns nil if no such function in Scopes.
func (s *Scopes) FindFunction(name string) *decls.FunctionDecl {
	if fn, found := s.scopes.functions[name]; found {
		return fn
	}
	if s.parent != nil {
		return s.parent.FindFunction(name)
	}
	return nil
}

// Group is a set of Decls that is pushed on or popped off a Scopes as a unit.
// Contains separate namespaces for identifier and function Decls.
// (Should be named "Scope" perhaps?)
type Group struct {
	idents    map[string]*decls.VariableDecl
	functions map[string]*decls.FunctionDecl
}

// copy creates a new Group instance with a shallow copy of the variables and functions.
// If callers need to mutate the exprpb.Decl definitions for a Function, they should copy-on-write.
func (g *Group) copy() *Group {
	cpy := &Group{
		idents:    make(map[string]*decls.VariableDecl, len(g.idents)),
		functions: make(map[string]*decls.FunctionDecl, len(g.functions)),
	}
	for n, id := range g.idents {
		cpy.idents[n] = id
	}
	for n, fn := range g.functions {
		cpy.functions[n] = fn
	}
	return cpy
}

// newGroup creates a new Group with empty maps for identifiers and functions.
func newGroup() *Group {
	return &Group{
		idents:    make(map[string]*decls.VariableDecl),
		functions: make(map[string]*decls.FunctionDecl),
	}
}
