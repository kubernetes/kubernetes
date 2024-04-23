/*
Copyright 2022 The Kubernetes Authors.

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

	"github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
)

const (
	ObjectVarName                    = "object"
	OldObjectVarName                 = "oldObject"
	ParamsVarName                    = "params"
	RequestVarName                   = "request"
	NamespaceVarName                 = "namespaceObject"
	AuthorizerVarName                = "authorizer"
	RequestResourceAuthorizerVarName = "authorizer.requestResource"
	VariableVarName                  = "variables"
)

// BuildRequestType generates a DeclType for AdmissionRequest. This may be replaced with a utility that
// converts the native type definition to apiservercel.DeclType once such a utility becomes available.
// The 'uid' field is omitted since it is not needed for in-process admission review.
// The 'object' and 'oldObject' fields are omitted since they are exposed as root level CEL variables.
func BuildRequestType() *apiservercel.DeclType {
	field := func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField {
		return apiservercel.NewDeclField(name, declType, required, nil, nil)
	}
	fields := func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField {
		result := make(map[string]*apiservercel.DeclField, len(fields))
		for _, f := range fields {
			result[f.Name] = f
		}
		return result
	}
	gvkType := apiservercel.NewObjectType("kubernetes.GroupVersionKind", fields(
		field("group", apiservercel.StringType, true),
		field("version", apiservercel.StringType, true),
		field("kind", apiservercel.StringType, true),
	))
	gvrType := apiservercel.NewObjectType("kubernetes.GroupVersionResource", fields(
		field("group", apiservercel.StringType, true),
		field("version", apiservercel.StringType, true),
		field("resource", apiservercel.StringType, true),
	))
	userInfoType := apiservercel.NewObjectType("kubernetes.UserInfo", fields(
		field("username", apiservercel.StringType, false),
		field("uid", apiservercel.StringType, false),
		field("groups", apiservercel.NewListType(apiservercel.StringType, -1), false),
		field("extra", apiservercel.NewMapType(apiservercel.StringType, apiservercel.NewListType(apiservercel.StringType, -1), -1), false),
	))
	return apiservercel.NewObjectType("kubernetes.AdmissionRequest", fields(
		field("kind", gvkType, true),
		field("resource", gvrType, true),
		field("subResource", apiservercel.StringType, false),
		field("requestKind", gvkType, true),
		field("requestResource", gvrType, true),
		field("requestSubResource", apiservercel.StringType, false),
		field("name", apiservercel.StringType, true),
		field("namespace", apiservercel.StringType, false),
		field("operation", apiservercel.StringType, true),
		field("userInfo", userInfoType, true),
		field("dryRun", apiservercel.BoolType, false),
		field("options", apiservercel.DynType, false),
	))
}

// BuildNamespaceType generates a DeclType for Namespace.
// Certain nested fields in Namespace (e.g. managedFields, ownerReferences etc.) are omitted in the generated DeclType
// by design.
func BuildNamespaceType() *apiservercel.DeclType {
	field := func(name string, declType *apiservercel.DeclType, required bool) *apiservercel.DeclField {
		return apiservercel.NewDeclField(name, declType, required, nil, nil)
	}
	fields := func(fields ...*apiservercel.DeclField) map[string]*apiservercel.DeclField {
		result := make(map[string]*apiservercel.DeclField, len(fields))
		for _, f := range fields {
			result[f.Name] = f
		}
		return result
	}

	specType := apiservercel.NewObjectType("kubernetes.NamespaceSpec", fields(
		field("finalizers", apiservercel.NewListType(apiservercel.StringType, -1), true),
	))
	conditionType := apiservercel.NewObjectType("kubernetes.NamespaceCondition", fields(
		field("status", apiservercel.StringType, true),
		field("type", apiservercel.StringType, true),
		field("lastTransitionTime", apiservercel.TimestampType, true),
		field("message", apiservercel.StringType, true),
		field("reason", apiservercel.StringType, true),
	))
	statusType := apiservercel.NewObjectType("kubernetes.NamespaceStatus", fields(
		field("conditions", apiservercel.NewListType(conditionType, -1), true),
		field("phase", apiservercel.StringType, true),
	))
	metadataType := apiservercel.NewObjectType("kubernetes.NamespaceMetadata", fields(
		field("name", apiservercel.StringType, true),
		field("generateName", apiservercel.StringType, true),
		field("namespace", apiservercel.StringType, true),
		field("labels", apiservercel.NewMapType(apiservercel.StringType, apiservercel.StringType, -1), true),
		field("annotations", apiservercel.NewMapType(apiservercel.StringType, apiservercel.StringType, -1), true),
		field("UID", apiservercel.StringType, true),
		field("creationTimestamp", apiservercel.TimestampType, true),
		field("deletionGracePeriodSeconds", apiservercel.IntType, true),
		field("deletionTimestamp", apiservercel.TimestampType, true),
		field("generation", apiservercel.IntType, true),
		field("resourceVersion", apiservercel.StringType, true),
		field("finalizers", apiservercel.NewListType(apiservercel.StringType, -1), true),
	))
	return apiservercel.NewObjectType("kubernetes.Namespace", fields(
		field("metadata", metadataType, true),
		field("spec", specType, true),
		field("status", statusType, true),
	))
}

// CompilationResult represents a compiled validations expression.
type CompilationResult struct {
	Program            cel.Program
	Error              *apiservercel.Error
	ExpressionAccessor ExpressionAccessor
	OutputType         *cel.Type
}

// Compiler provides a CEL expression compiler configured with the desired admission related CEL variables and
// environment mode.
type Compiler interface {
	CompileCELExpression(expressionAccessor ExpressionAccessor, options OptionalVariableDeclarations, mode environment.Type) CompilationResult
}

type compiler struct {
	varEnvs variableDeclEnvs
}

func NewCompiler(env *environment.EnvSet) Compiler {
	return &compiler{varEnvs: mustBuildEnvs(env)}
}

type variableDeclEnvs map[OptionalVariableDeclarations]*environment.EnvSet

// CompileCELExpression returns a compiled CEL expression.
// perCallLimit was added for testing purpose only. Callers should always use const PerCallLimit from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func (c compiler) CompileCELExpression(expressionAccessor ExpressionAccessor, options OptionalVariableDeclarations, envType environment.Type) CompilationResult {
	resultError := func(errorString string, errType apiservercel.ErrorType, errors ...error) CompilationResult {
		return CompilationResult{
			Error: &apiservercel.Error{
				Type:   errType,
				Detail: errorString,
				Errors: errors,
			},
			ExpressionAccessor: expressionAccessor,
		}
	}

	env, err := c.varEnvs[options].Env(envType)
	if err != nil {
		return resultError(fmt.Sprintf("unexpected error loading CEL environment: %v", err), apiservercel.ErrorTypeInternal)
	}

	ast, issues := env.Compile(expressionAccessor.GetExpression())
	if issues != nil {
		return resultError("compilation failed: "+issues.String(), apiservercel.ErrorTypeInvalid, apiservercel.NewCompilationError(issues))
	}
	found := false
	returnTypes := expressionAccessor.ReturnTypes()
	for _, returnType := range returnTypes {
		if ast.OutputType() == returnType || cel.AnyType == returnType {
			found = true
			break
		}
	}
	if !found {
		var reason string
		if len(returnTypes) == 1 {
			reason = fmt.Sprintf("must evaluate to %v", returnTypes[0].String())
		} else {
			reason = fmt.Sprintf("must evaluate to one of %v", returnTypes)
		}

		return resultError(reason, apiservercel.ErrorTypeInvalid)
	}

	_, err = cel.AstToCheckedExpr(ast)
	if err != nil {
		// should be impossible since env.Compile returned no issues
		return resultError("unexpected compilation error: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	prog, err := env.Program(ast,
		cel.InterruptCheckFrequency(celconfig.CheckFrequency),
	)
	if err != nil {
		return resultError("program instantiation failed: "+err.Error(), apiservercel.ErrorTypeInternal)
	}
	return CompilationResult{
		Program:            prog,
		ExpressionAccessor: expressionAccessor,
		OutputType:         ast.OutputType(),
	}
}

func mustBuildEnvs(baseEnv *environment.EnvSet) variableDeclEnvs {
	requestType := BuildRequestType()
	namespaceType := BuildNamespaceType()
	envs := make(variableDeclEnvs, 8) // since the number of variable combinations is small, pre-build a environment for each
	for _, hasParams := range []bool{false, true} {
		for _, hasAuthorizer := range []bool{false, true} {
			for _, strictCost := range []bool{false, true} {
				var envOpts []cel.EnvOption
				if hasParams {
					envOpts = append(envOpts, cel.Variable(ParamsVarName, cel.DynType))
				}
				if hasAuthorizer {
					envOpts = append(envOpts,
						cel.Variable(AuthorizerVarName, library.AuthorizerType),
						cel.Variable(RequestResourceAuthorizerVarName, library.ResourceCheckType))
				}
				envOpts = append(envOpts,
					cel.Variable(ObjectVarName, cel.DynType),
					cel.Variable(OldObjectVarName, cel.DynType),
					cel.Variable(NamespaceVarName, namespaceType.CelType()),
					cel.Variable(RequestVarName, requestType.CelType()))

				extended, err := baseEnv.Extend(
					environment.VersionedOptions{
						// Feature epoch was actually 1.26, but we artificially set it to 1.0 because these
						// options should always be present.
						IntroducedVersion: version.MajorMinor(1, 0),
						EnvOptions:        envOpts,
						DeclTypes: []*apiservercel.DeclType{
							namespaceType,
							requestType,
						},
					},
				)
				if err != nil {
					panic(fmt.Sprintf("environment misconfigured: %v", err))
				}
				if strictCost {
					extended, err = extended.Extend(environment.StrictCostOpt)
					if err != nil {
						panic(fmt.Sprintf("environment misconfigured: %v", err))
					}
				}
				envs[OptionalVariableDeclarations{HasParams: hasParams, HasAuthorizer: hasAuthorizer, StrictCost: strictCost}] = extended
			}
		}
	}
	return envs
}
