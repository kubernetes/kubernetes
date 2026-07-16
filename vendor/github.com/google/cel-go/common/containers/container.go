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

// Package containers defines types and functions for resolving qualified names within a namespace
// or type provided to CEL.
package containers

import (
	"fmt"
	"strings"
	"unicode"

	"github.com/google/cel-go/common/ast"
)

var (
	// DefaultContainer has an empty container name.
	DefaultContainer *Container = nil

	// Empty map to search for aliases when needed.
	noAliases = make(map[string]string)
)

// NewContainer creates a new Container with the fully-qualified name.
func NewContainer(opts ...ContainerOption) (*Container, error) {
	var c *Container
	var err error
	for _, opt := range opts {
		c, err = opt(c)
		if err != nil {
			return nil, err
		}
	}
	return c, nil
}

// Container holds a reference to an optional qualified container name and set of aliases.
//
// The program container can be used to simplify variable, function, and type specification within
// CEL programs and behaves more or less like a C++ namespace. See ResolveCandidateNames for more
// details.
type Container struct {
	name    string
	aliases map[string]string
}

// Extend creates a new Container with the existing settings and applies a series of
// ContainerOptions to further configure the new container.
func (c *Container) Extend(opts ...ContainerOption) (*Container, error) {
	if c == nil {
		return NewContainer(opts...)
	}
	// Copy the name and aliases of the existing container.
	ext := &Container{name: c.Name()}
	if len(c.AliasSet()) > 0 {
		aliasSet := make(map[string]string, len(c.AliasSet()))
		for k, v := range c.AliasSet() {
			aliasSet[k] = v
		}
		ext.aliases = aliasSet
	}
	// Apply the new options to the container.
	var err error
	for _, opt := range opts {
		ext, err = opt(ext)
		if err != nil {
			return nil, err
		}
	}
	return ext, nil
}

// Name returns the fully-qualified name of the container.
//
// The name may conceptually be a namespace, package, or type.
func (c *Container) Name() string {
	if c == nil {
		return ""
	}
	return c.name
}

// ResolveCandidateNames returns the candidates name of namespaced identifiers in C++ resolution
// order.
//
// Names which shadow other names are returned first. If a name includes a leading dot ('.'),
// the name is treated as an absolute identifier which cannot be shadowed.
//
// Given a container name a.b.c.M.N and a type name R.s, this will deliver in order:
//
//	a.b.c.M.N.R.s
//	a.b.c.M.R.s
//	a.b.c.R.s
//	a.b.R.s
//	a.R.s
//	R.s
//
// If aliases or abbreviations are configured for the container, then alias names will take
// precedence over containerized names.
func (c *Container) ResolveCandidateNames(name string) []string {
	if strings.HasPrefix(name, ".") {
		qn := name[1:]
		alias, isAlias := c.findAlias(qn)
		if isAlias {
			return []string{alias}
		}
		return []string{qn}
	}
	alias, isAlias := c.findAlias(name)
	if isAlias {
		return []string{alias}
	}
	if c.Name() == "" {
		return []string{name}
	}
	nextCont := c.Name()
	candidates := []string{nextCont + "." + name}
	for i := strings.LastIndex(nextCont, "."); i >= 0; i = strings.LastIndex(nextCont, ".") {
		nextCont = nextCont[:i]
		candidates = append(candidates, nextCont+"."+name)
	}
	return append(candidates, name)
}

// AliasSet returns the alias to fully-qualified name mapping stored in the container.
func (c *Container) AliasSet() map[string]string {
	if c == nil || c.aliases == nil {
		return noAliases
	}
	return c.aliases
}

// findAlias takes a name as input and returns an alias expansion if one exists.
//
// If the name is qualified, the first component of the qualified name is checked against known
// aliases. Any alias that is found in a qualified name is expanded in the result:
//
//	alias: R -> my.alias.R
//	name: R.S.T
//	output: my.alias.R.S.T
//
// Note, the name must not have a leading dot.
func (c *Container) findAlias(name string) (string, bool) {
	// If an alias exists for the name, ensure it is searched last.
	simple := name
	qualifier := ""
	dot := strings.Index(name, ".")
	if dot >= 0 {
		simple = name[0:dot]
		qualifier = name[dot:]
	}
	alias, found := c.AliasSet()[simple]
	if !found {
		return "", false
	}
	return alias + qualifier, true
}

// ContainerOption specifies a functional configuration option for a Container.
//
// Note, ContainerOption implementations must be able to handle nil container inputs.
type ContainerOption func(*Container) (*Container, error)

// Abbrevs configures a set of simple names as abbreviations for fully-qualified names.
//
// An abbreviation (abbrev for short) is a simple name that expands to a fully-qualified name.
// Abbreviations can be useful when working with variables, functions, and especially types from
// multiple namespaces:
//
//	// CEL object construction
//	qual.pkg.version.ObjTypeName{
//	   field: alt.container.ver.FieldTypeName{value: ...}
//	}
//
// Only one the qualified names above may be used as the CEL container, so at least one of these
// references must be a long qualified name within an otherwise short CEL program. Using the
// following abbreviations, the program becomes much simpler:
//
//	// CEL Go option
//	Abbrevs("qual.pkg.version.ObjTypeName", "alt.container.ver.FieldTypeName")
//	// Simplified Object construction
//	ObjTypeName{field: FieldTypeName{value: ...}}
//
// There are a few rules for the qualified names and the simple abbreviations generated from them:
// - Qualified names must be dot-delimited, e.g. `package.subpkg.name`.
// - The last element in the qualified name is the abbreviation.
// - Abbreviations must not collide with each other.
// - The abbreviation must not collide with unqualified names in use.
//
// Abbreviations are distinct from container-based references in the following important ways:
//   - Abbreviations must expand to a fully-qualified name.
//   - Expanded abbreviations do not participate in namespace resolution.
//   - Abbreviation expansion is done instead of the container search for a matching identifier.
//   - Containers follow C++ namespace resolution rules with searches from the most qualified name
//     to the least qualified name.
//   - Container references within the CEL program may be relative, and are resolved to fully
//     qualified names at either type-check time or program plan time, whichever comes first.
//
// If there is ever a case where an identifier could be in both the container and as an
// abbreviation, the abbreviation wins as this will ensure that the meaning of a program is
// preserved between compilations even as the container evolves.
func Abbrevs(qualifiedNames ...string) ContainerOption {
	return func(c *Container) (*Container, error) {
		for _, qn := range qualifiedNames {
			qn = strings.TrimSpace(qn)
			for _, r := range qn {
				if !isIdentifierChar(r) {
					return nil, fmt.Errorf(
						"invalid qualified name: %s, wanted name of the form 'qualified.name'", qn)
				}
			}
			ind := strings.LastIndex(qn, ".")
			if ind <= 0 || ind >= len(qn)-1 {
				return nil, fmt.Errorf(
					"invalid qualified name: %s, wanted name of the form 'qualified.name'", qn)
			}
			alias := qn[ind+1:]
			var err error
			c, err = aliasAs("abbreviation", qn, alias)(c)
			if err != nil {
				return nil, err
			}
		}
		return c, nil
	}
}

// Alias associates a fully-qualified name with a user-defined alias.
//
// In general, Abbrevs is preferred to Alias since the names generated from the Abbrevs option
// are more easily traced back to source code. The Alias option is useful for propagating alias
// configuration from one Container instance to another, and may also be useful for remapping
// poorly chosen protobuf message / package names.
//
// Note: all of the rules that apply to Abbrevs also apply to Alias.
func Alias(qualifiedName, alias string) ContainerOption {
	return aliasAs("alias", qualifiedName, alias)
}

func aliasAs(kind, qualifiedName, alias string) ContainerOption {
	return func(c *Container) (*Container, error) {
		if len(alias) == 0 || strings.Contains(alias, ".") {
			return nil, fmt.Errorf(
				"%s must be non-empty and simple (not qualified): %s=%s", kind, kind, alias)
		}

		if qualifiedName[0:1] == "." {
			return nil, fmt.Errorf("qualified name must not begin with a leading '.': %s",
				qualifiedName)
		}
		ind := strings.LastIndex(qualifiedName, ".")
		if ind <= 0 || ind == len(qualifiedName)-1 {
			return nil, fmt.Errorf("%s must refer to a valid qualified name: %s",
				kind, qualifiedName)
		}
		aliasRef, found := c.AliasSet()[alias]
		if found {
			return nil, fmt.Errorf(
				"%s collides with existing reference: name=%s, %s=%s, existing=%s",
				kind, qualifiedName, kind, alias, aliasRef)
		}
		if strings.HasPrefix(c.Name(), alias+".") || c.Name() == alias {
			return nil, fmt.Errorf(
				"%s collides with container name: name=%s, %s=%s, container=%s",
				kind, qualifiedName, kind, alias, c.Name())
		}
		if c == nil {
			c = &Container{}
		}
		if c.aliases == nil {
			c.aliases = make(map[string]string)
		}
		c.aliases[alias] = qualifiedName
		return c, nil
	}
}

func isIdentifierChar(r rune) bool {
	return r <= unicode.MaxASCII && (r == '.' || r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r))
}

// Name sets the fully-qualified name of the Container.
func Name(name string) ContainerOption {
	return func(c *Container) (*Container, error) {
		if len(name) > 0 && name[0:1] == "." {
			return nil, fmt.Errorf("container name must not contain a leading '.': %s", name)
		}
		if c.Name() == name {
			return c, nil
		}
		if c == nil {
			return &Container{name: name}, nil
		}
		c.name = name
		return c, nil
	}
}

// ToQualifiedName converts an expression AST into a qualified name if possible, with a boolean
// 'found' value that indicates if the conversion is successful.
func ToQualifiedName(e ast.Expr) (string, bool) {
	switch e.Kind() {
	case ast.IdentKind:
		id := e.AsIdent()
		return id, true
	case ast.SelectKind:
		sel := e.AsSelect()
		// Test only expressions are not valid as qualified names.
		if sel.IsTestOnly() {
			return "", false
		}
		if qual, found := ToQualifiedName(sel.Operand()); found {
			return qual + "." + sel.FieldName(), true
		}
	}
	return "", false
}
