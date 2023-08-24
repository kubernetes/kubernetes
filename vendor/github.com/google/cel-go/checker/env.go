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
	"fmt"
	"strings"

	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/decls"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/parser"
)

type aggregateLiteralElementType int

const (
	dynElementType        aggregateLiteralElementType = iota
	homogenousElementType aggregateLiteralElementType = 1 << iota
)

var (
	crossTypeNumericComparisonOverloads = map[string]struct{}{
		// double <-> int | uint
		overloads.LessDoubleInt64:           {},
		overloads.LessDoubleUint64:          {},
		overloads.LessEqualsDoubleInt64:     {},
		overloads.LessEqualsDoubleUint64:    {},
		overloads.GreaterDoubleInt64:        {},
		overloads.GreaterDoubleUint64:       {},
		overloads.GreaterEqualsDoubleInt64:  {},
		overloads.GreaterEqualsDoubleUint64: {},
		// int <-> double | uint
		overloads.LessInt64Double:          {},
		overloads.LessInt64Uint64:          {},
		overloads.LessEqualsInt64Double:    {},
		overloads.LessEqualsInt64Uint64:    {},
		overloads.GreaterInt64Double:       {},
		overloads.GreaterInt64Uint64:       {},
		overloads.GreaterEqualsInt64Double: {},
		overloads.GreaterEqualsInt64Uint64: {},
		// uint <-> double | int
		overloads.LessUint64Double:          {},
		overloads.LessUint64Int64:           {},
		overloads.LessEqualsUint64Double:    {},
		overloads.LessEqualsUint64Int64:     {},
		overloads.GreaterUint64Double:       {},
		overloads.GreaterUint64Int64:        {},
		overloads.GreaterEqualsUint64Double: {},
		overloads.GreaterEqualsUint64Int64:  {},
	}
)

// Env is the environment for type checking.
//
// The Env is comprised of a container, type provider, declarations, and other related objects
// which can be used to assist with type-checking.
type Env struct {
	container           *containers.Container
	provider            types.Provider
	declarations        *Scopes
	aggLitElemType      aggregateLiteralElementType
	filteredOverloadIDs map[string]struct{}
}

// NewEnv returns a new *Env with the given parameters.
func NewEnv(container *containers.Container, provider types.Provider, opts ...Option) (*Env, error) {
	declarations := newScopes()
	declarations.Push()

	envOptions := &options{}
	for _, opt := range opts {
		if err := opt(envOptions); err != nil {
			return nil, err
		}
	}
	aggLitElemType := dynElementType
	if envOptions.homogeneousAggregateLiterals {
		aggLitElemType = homogenousElementType
	}
	filteredOverloadIDs := crossTypeNumericComparisonOverloads
	if envOptions.crossTypeNumericComparisons {
		filteredOverloadIDs = make(map[string]struct{})
	}
	if envOptions.validatedDeclarations != nil {
		declarations = envOptions.validatedDeclarations.Copy()
	}
	return &Env{
		container:           container,
		provider:            provider,
		declarations:        declarations,
		aggLitElemType:      aggLitElemType,
		filteredOverloadIDs: filteredOverloadIDs,
	}, nil
}

// AddIdents configures the checker with a list of variable declarations.
//
// If there are overlapping declarations, the method will error.
func (e *Env) AddIdents(declarations ...*decls.VariableDecl) error {
	errMsgs := make([]errorMsg, 0)
	for _, d := range declarations {
		errMsgs = append(errMsgs, e.addIdent(d))
	}
	return formatError(errMsgs)
}

// AddFunctions configures the checker with a list of function declarations.
//
// If there are overlapping declarations, the method will error.
func (e *Env) AddFunctions(declarations ...*decls.FunctionDecl) error {
	errMsgs := make([]errorMsg, 0)
	for _, d := range declarations {
		errMsgs = append(errMsgs, e.setFunction(d)...)
	}
	return formatError(errMsgs)
}

// LookupIdent returns a Decl proto for typeName as an identifier in the Env.
// Returns nil if no such identifier is found in the Env.
func (e *Env) LookupIdent(name string) *decls.VariableDecl {
	for _, candidate := range e.container.ResolveCandidateNames(name) {
		if ident := e.declarations.FindIdent(candidate); ident != nil {
			return ident
		}

		// Next try to import the name as a reference to a message type. If found,
		// the declaration is added to the outest (global) scope of the
		// environment, so next time we can access it faster.
		if t, found := e.provider.FindStructType(candidate); found {
			decl := decls.NewVariable(candidate, t)
			e.declarations.AddIdent(decl)
			return decl
		}

		// Next try to import this as an enum value by splitting the name in a type prefix and
		// the enum inside.
		if enumValue := e.provider.EnumValue(candidate); enumValue.Type() != types.ErrType {
			decl := decls.NewConstant(candidate, types.IntType, enumValue)
			e.declarations.AddIdent(decl)
			return decl
		}
	}
	return nil
}

// LookupFunction returns a Decl proto for typeName as a function in env.
// Returns nil if no such function is found in env.
func (e *Env) LookupFunction(name string) *decls.FunctionDecl {
	for _, candidate := range e.container.ResolveCandidateNames(name) {
		if fn := e.declarations.FindFunction(candidate); fn != nil {
			return fn
		}
	}
	return nil
}

// setFunction adds the function Decl to the Env.
// Adds a function decl if one doesn't already exist, then adds all overloads from the Decl.
// If overload overlaps with an existing overload, adds to the errors  in the Env instead.
func (e *Env) setFunction(fn *decls.FunctionDecl) []errorMsg {
	errMsgs := make([]errorMsg, 0)
	current := e.declarations.FindFunction(fn.Name())
	if current != nil {
		var err error
		current, err = current.Merge(fn)
		if err != nil {
			return append(errMsgs, errorMsg(err.Error()))
		}
	} else {
		current = fn
	}
	for _, overload := range current.OverloadDecls() {
		for _, macro := range parser.AllMacros {
			if macro.Function() == current.Name() &&
				macro.IsReceiverStyle() == overload.IsMemberFunction() &&
				macro.ArgCount() == len(overload.ArgTypes()) {
				errMsgs = append(errMsgs, overlappingMacroError(current.Name(), macro.ArgCount()))
			}
		}
		if len(errMsgs) > 0 {
			return errMsgs
		}
	}
	e.declarations.SetFunction(current)
	return errMsgs
}

// addIdent adds the Decl to the declarations in the Env.
// Returns a non-empty errorMsg if the identifier is already declared in the scope.
func (e *Env) addIdent(decl *decls.VariableDecl) errorMsg {
	current := e.declarations.FindIdentInScope(decl.Name())
	if current != nil {
		if current.DeclarationIsEquivalent(decl) {
			return ""
		}
		return overlappingIdentifierError(decl.Name())
	}
	e.declarations.AddIdent(decl)
	return ""
}

// isOverloadDisabled returns whether the overloadID is disabled in the current environment.
func (e *Env) isOverloadDisabled(overloadID string) bool {
	_, found := e.filteredOverloadIDs[overloadID]
	return found
}

// validatedDeclarations returns a reference to the validated variable and function declaration scope stack.
// must be copied before use.
func (e *Env) validatedDeclarations() *Scopes {
	return e.declarations
}

// enterScope creates a new Env instance with a new innermost declaration scope.
func (e *Env) enterScope() *Env {
	childDecls := e.declarations.Push()
	return &Env{
		declarations:   childDecls,
		container:      e.container,
		provider:       e.provider,
		aggLitElemType: e.aggLitElemType,
	}
}

// exitScope creates a new Env instance with the nearest outer declaration scope.
func (e *Env) exitScope() *Env {
	parentDecls := e.declarations.Pop()
	return &Env{
		declarations:   parentDecls,
		container:      e.container,
		provider:       e.provider,
		aggLitElemType: e.aggLitElemType,
	}
}

// errorMsg is a type alias meant to represent error-based return values which
// may be accumulated into an error at a later point in execution.
type errorMsg string

func overlappingIdentifierError(name string) errorMsg {
	return errorMsg(fmt.Sprintf("overlapping identifier for name '%s'", name))
}

func overlappingMacroError(name string, argCount int) errorMsg {
	return errorMsg(fmt.Sprintf(
		"overlapping macro for name '%s' with %d args", name, argCount))
}

func formatError(errMsgs []errorMsg) error {
	errStrs := make([]string, 0)
	if len(errMsgs) > 0 {
		for i := 0; i < len(errMsgs); i++ {
			if errMsgs[i] != "" {
				errStrs = append(errStrs, string(errMsgs[i]))
			}
		}
	}
	if len(errStrs) > 0 {
		return fmt.Errorf("%s", strings.Join(errStrs, "\n"))
	}
	return nil
}
