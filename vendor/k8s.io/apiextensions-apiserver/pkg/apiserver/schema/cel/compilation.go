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
	"fmt"
	"strings"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/ext"

	expr "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
	"google.golang.org/protobuf/proto"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	celmodel "k8s.io/apiextensions-apiserver/third_party/forked/celopenapi/model"
)

// ScopedVarName is the variable name assigned to the locally scoped data element of a CEL valid.
const ScopedVarName = "self"

// CompilationResult represents the cel compilation result for one rule
type CompilationResult struct {
	Program cel.Program
	Error   *Error
}

// Compile compiles all the XValidations rules (without recursing into the schema) and returns a slice containing a
// CompilationResult for each ValidationRule, or an error.
// Each CompilationResult may contain:
/// - non-nil Program, nil Error: The program was compiled successfully
//  - nil Program, non-nil Error: Compilation resulted in an error
//  - nil Program, nil Error: The provided rule was empty so compilation was not attempted
func Compile(s *schema.Structural, isResourceRoot bool) ([]CompilationResult, error) {
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
	scopedTypeName := generateUniqueSelfTypeName()
	rt, err := celmodel.NewRuleTypes(scopedTypeName, s, isResourceRoot, reg)
	if err != nil {
		return nil, err
	}
	if rt == nil {
		return nil, nil
	}
	opts, err := rt.EnvOptions(env.TypeProvider())
	if err != nil {
		return nil, err
	}
	root, ok = rt.FindDeclType(scopedTypeName)
	if !ok {
		rootDecl := celmodel.SchemaDeclType(s, isResourceRoot)
		if rootDecl == nil {
			return nil, fmt.Errorf("rule declared on schema that does not support validation rules type: '%s' x-kubernetes-preserve-unknown-fields: '%t'", s.Type, s.XPreserveUnknownFields)
		}
		root = rootDecl.MaybeAssignTypeName(scopedTypeName)
	}
	propDecls = append(propDecls, decls.NewVar(ScopedVarName, root.ExprType()))
	opts = append(opts, cel.Declarations(propDecls...))
	opts = append(opts, ext.Strings())
	env, err = env.Extend(opts...)
	if err != nil {
		return nil, err
	}

	// compResults is the return value which saves a list of compilation results in the same order as x-kubernetes-validations rules.
	compResults := make([]CompilationResult, len(celRules))
	for i, rule := range celRules {
		var compilationResult CompilationResult
		if len(strings.TrimSpace(rule.Rule)) == 0 {
			// include a compilation result, but leave both program and error nil per documented return semantics of this
			// function
		} else {
			ast, issues := env.Compile(rule.Rule)
			if issues != nil {
				compilationResult.Error = &Error{ErrorTypeInvalid, "compilation failed: " + issues.String()}
			} else if !proto.Equal(ast.ResultType(), decls.Bool) {
				compilationResult.Error = &Error{ErrorTypeInvalid, "cel expression must evaluate to a bool"}
			} else {
				prog, err := env.Program(ast)
				if err != nil {
					compilationResult.Error = &Error{ErrorTypeInvalid, "program instantiation failed: " + err.Error()}
				} else {
					compilationResult.Program = prog
				}
			}
		}

		compResults[i] = compilationResult
	}

	return compResults, nil
}

// generateUniqueSelfTypeName creates a placeholder type name to use in a CEL programs for cases
// where we do not wish to expose a stable type name to CEL validator rule authors. For this to effectively prevent
// developers from depending on the generated name (i.e. using it in CEL programs), it must be changed each time a
// CRD is created or updated.
func generateUniqueSelfTypeName() string {
	return fmt.Sprintf("selfType%d", time.Now().Nanosecond())
}
