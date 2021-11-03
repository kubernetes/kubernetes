/*
Copyright 2021 The Kubernetes Authors.

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

package cel

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	expr "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	"google.golang.org/protobuf/proto"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	celmodel "k8s.io/apiextensions-apiserver/third_party/forked/celopenapi/model"
)

// ScopedTypeName is the placeholder type name used for the type of ScopedVarName if it is an object type.
const ScopedTypeName = "apiextensions.k8s.io.v1alpha1.ValidationExpressionSelf"

// ScopedVarName is the variable name assigned to the locally scoped data element of a CEL valid.
const ScopedVarName = "self"

// CompilationResult represents the cel compilation result for one rule
type CompilationResult struct {
	Rule      apiextensions.ValidationRule
	Program   cel.Program
	Error     Error
	RuleIndex int
}

// Compile compiles all the XValidations rules (without recursing into the schema) and returns a slice containing a compiled program for each provided CelRule, or an array of errors.
func Compile(s *schema.Structural) ([]CompilationResult, error) {
	if len(s.Extensions.XValidations) == 0 {
		return nil, nil
	}
	celRules := s.Extensions.XValidations

	var propDecls []*expr.Decl
	var root *celmodel.DeclType
	var ok bool
	env, err := cel.NewEnv()
	if err != nil {
		return nil, err
	}
	reg := celmodel.NewRegistry(env)
	rt, err := celmodel.NewRuleTypes(ScopedTypeName, s, reg)
	if err != nil {
		return nil, err
	}
	opts, err := rt.EnvOptions(env.TypeProvider())
	if err != nil {
		return nil, err
	}
	root, ok = rt.FindDeclType(ScopedTypeName)
	if !ok {
		root = celmodel.SchemaDeclType(s).MaybeAssignTypeName(ScopedTypeName)
	}
	// if the type is object, will traverse each field in the object tree and declare
	if root.IsObject() {
		for k, f := range root.Fields {
			if !(celmodel.IsRootReserved(k) || k == ScopedVarName) {
				propDecls = append(propDecls, decls.NewVar(k, f.Type.ExprType()))
			}
		}
	}
	propDecls = append(propDecls, decls.NewVar(ScopedVarName, root.ExprType()))
	opts = append(opts, cel.Declarations(propDecls...))
	env, err = env.Extend(opts...)
	if err != nil {
		return nil, err
	}
	compResults := make([]CompilationResult, len(celRules))
	for i, rule := range celRules {
		var compilationResult CompilationResult
		compilationResult.RuleIndex = i
		compilationResult.Rule = rule
		if rule.Rule == "" {
			compilationResult.Error = Error{ErrorTypeRequired, "rule is not specified"}
		} else {
			ast, issues := env.Compile(rule.Rule)
			if issues != nil {
				compilationResult.Error = Error{ErrorTypeInvalid, "compilation failed: " + issues.String()}
			} else if !proto.Equal(ast.ResultType(), decls.Bool) {
				compilationResult.Error = Error{ErrorTypeInvalid, "cel expression should evaluate to a bool"}
			} else {
				prog, err := env.Program(ast)
				if err != nil {
					compilationResult.Error = Error{ErrorTypeInvalid, "program instantiation failed: " + err.Error()}
				} else {
					compilationResult.Program = prog
				}
			}
		}

		compResults[i] = compilationResult
	}

	return compResults, nil
}
