// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// NewEnv creates an empty Env instance with a fully qualified name that may be referenced
// within templates.
func NewEnv(name string) *Env {
	return &Env{
		Name:      name,
		Functions: []*Function{},
		Vars:      []*Var{},
		Types:     map[string]*DeclType{},
	}
}

// Env declares a set of variables, functions, and types available to a given set of CEL
// expressions.
//
// The Env name must be fully qualified as it will be referenced within template evaluators,
// validators, and possibly within the metadata of the instance rule schema.
//
// Note, the Types values currently only holds type definitions associated with a variable
// declaration. Any type mentioned in the environment which does not have a definition is
// treated as a reference to a type which must be supplied in the base CEL environment provided
// by the policy engine.
type Env struct {
	Name      string
	Container string
	Functions []*Function
	Vars      []*Var
	Types     map[string]*DeclType
}

// ExprEnvOptions returns a set of CEL environment options to be used when extending the base
// policy engine CEL environment.
func (e *Env) ExprEnvOptions() []cel.EnvOption {
	var opts []cel.EnvOption
	if e.Container != "" {
		opts = append(opts, cel.Container(e.Container))
	}
	if len(e.Vars) > 0 {
		vars := make([]*exprpb.Decl, len(e.Vars))
		for i, v := range e.Vars {
			vars[i] = v.ExprDecl()
		}
		opts = append(opts, cel.Declarations(vars...))
	}
	if len(e.Functions) > 0 {
		funcs := make([]*exprpb.Decl, len(e.Functions))
		for i, f := range e.Functions {
			funcs[i] = f.ExprDecl()
		}
		opts = append(opts, cel.Declarations(funcs...))
	}
	return opts
}

// NewVar creates a new variable with a name and a type.
func NewVar(name string, dt *DeclType) *Var {
	return &Var{
		Name: name,
		Type: dt,
	}
}

// Var represents a named instanced of a type.
type Var struct {
	Name string
	Type *DeclType
}

// ExprDecl produces a CEL proto declaration for the variable.
func (v *Var) ExprDecl() *exprpb.Decl {
	return decls.NewVar(v.Name, v.Type.ExprType())
}

// NewFunction creates a Function instance with a simple function name and a set of overload
// signatures.
func NewFunction(name string, overloads ...*Overload) *Function {
	return &Function{
		Name:      name,
		Overloads: overloads,
	}
}

// Function represents a simple name and a set of overload signatures.
type Function struct {
	Name      string
	Overloads []*Overload
}

// ExprDecl produces a CEL proto declaration for the function and its overloads.
func (f *Function) ExprDecl() *exprpb.Decl {
	overloadDecls := make([]*exprpb.Decl_FunctionDecl_Overload, len(f.Overloads))
	for i, o := range f.Overloads {
		overloadDecls[i] = o.overloadDecl()
	}
	return decls.NewFunction(f.Name, overloadDecls...)
}

// NewOverload returns a receiver-style overload declaration for a given function.
//
// The overload name must follow the conventions laid out within the CEL overloads.go file.
//
//     // Receiver-style overload name:
//     <receiver_type>_<func>_<arg_type0>_<arg_typeN>
//
// Within this function, the first type supplied is the receiver type, and the last type supplied
// is used as the return type. At least two types must be specified for a zero-arity receiver
// function.
func NewOverload(name string, first *DeclType, rest ...*DeclType) *Overload {
	argTypes := make([]*DeclType, 1+len(rest))
	argTypes[0] = first
	for i := 1; i < len(rest)+1; i++ {
		argTypes[i] = rest[i-1]
	}
	returnType := argTypes[len(argTypes)-1]
	argTypes = argTypes[0 : len(argTypes)-1]
	return newOverload(name, false, argTypes, returnType)
}

// NewFreeFunctionOverload returns a free function overload for a given function name.
//
// The overload name must follow the conventions laid out within the CEL overloads.go file:
//
//     // Free function style overload name:
//     <func>_<arg_type0>_<arg_typeN>
//
// When the function name is global, <func> will refer to the simple function name. When the
// function has a qualified name, replace the '.' characters in the fully-qualified name with
// underscores.
//
// Within this function, the last type supplied is used as the return type. At least one type must
// be specified for a zero-arity free function.
func NewFreeFunctionOverload(name string, first *DeclType, rest ...*DeclType) *Overload {
	argTypes := make([]*DeclType, 1+len(rest))
	argTypes[0] = first
	for i := 1; i < len(rest)+1; i++ {
		argTypes[i] = rest[i-1]
	}
	returnType := argTypes[len(argTypes)-1]
	argTypes = argTypes[0 : len(argTypes)-1]
	return newOverload(name, true, argTypes, returnType)
}

func newOverload(name string,
	freeFunction bool,
	argTypes []*DeclType,
	returnType *DeclType) *Overload {
	return &Overload{
		Name:         name,
		FreeFunction: freeFunction,
		Args:         argTypes,
		ReturnType:   returnType,
	}
}

// Overload represents a single function overload signature.
type Overload struct {
	Name         string
	FreeFunction bool
	Args         []*DeclType
	ReturnType   *DeclType
}

func (o *Overload) overloadDecl() *exprpb.Decl_FunctionDecl_Overload {
	typeParams := map[string]struct{}{}
	argExprTypes := make([]*exprpb.Type, len(o.Args))
	for i, a := range o.Args {
		if a.TypeParam {
			typeParams[a.TypeName()] = struct{}{}
		}
		argExprTypes[i] = a.ExprType()
	}
	returnType := o.ReturnType.ExprType()
	if len(typeParams) == 0 {
		if o.FreeFunction {
			return decls.NewOverload(o.Name, argExprTypes, returnType)
		}
		return decls.NewInstanceOverload(o.Name, argExprTypes, returnType)
	}
	typeParamNames := make([]string, 0, len(typeParams))
	for param := range typeParams {
		typeParamNames = append(typeParamNames, param)
	}
	if o.FreeFunction {
		return decls.NewParameterizedOverload(o.Name, argExprTypes, returnType, typeParamNames)
	}
	return decls.NewParameterizedInstanceOverload(o.Name, argExprTypes, returnType, typeParamNames)
}
