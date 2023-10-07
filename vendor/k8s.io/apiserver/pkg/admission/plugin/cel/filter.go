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
	"context"
	"fmt"
	"math"
	"reflect"
	"time"

	"github.com/google/cel-go/interpreter"

	admissionv1 "k8s.io/api/admission/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

// filterCompiler implement the interface FilterCompiler.
type filterCompiler struct {
}

func NewFilterCompiler() FilterCompiler {
	return &filterCompiler{}
}

type evaluationActivation struct {
	object, oldObject, params, request, authorizer, requestResourceAuthorizer interface{}
}

// ResolveName returns a value from the activation by qualified name, or false if the name
// could not be found.
func (a *evaluationActivation) ResolveName(name string) (interface{}, bool) {
	switch name {
	case ObjectVarName:
		return a.object, true
	case OldObjectVarName:
		return a.oldObject, true
	case ParamsVarName:
		return a.params, true // params may be null
	case RequestVarName:
		return a.request, true
	case AuthorizerVarName:
		return a.authorizer, a.authorizer != nil
	case RequestResourceAuthorizerVarName:
		return a.requestResourceAuthorizer, a.requestResourceAuthorizer != nil
	default:
		return nil, false
	}
}

// Parent returns the parent of the current activation, may be nil.
// If non-nil, the parent will be searched during resolve calls.
func (a *evaluationActivation) Parent() interpreter.Activation {
	return nil
}

// Compile compiles the cel expressions defined in the ExpressionAccessors into a Filter
func (c *filterCompiler) Compile(expressionAccessors []ExpressionAccessor, options OptionalVariableDeclarations, perCallLimit uint64) Filter {
	compilationResults := make([]CompilationResult, len(expressionAccessors))
	for i, expressionAccessor := range expressionAccessors {
		if expressionAccessor == nil {
			continue
		}
		compilationResults[i] = CompileCELExpression(expressionAccessor, options, perCallLimit)
	}
	return NewFilter(compilationResults)
}

// filter implements the Filter interface
type filter struct {
	compilationResults []CompilationResult
}

func NewFilter(compilationResults []CompilationResult) Filter {
	return &filter{
		compilationResults,
	}
}

func convertObjectToUnstructured(obj interface{}) (*unstructured.Unstructured, error) {
	if obj == nil || reflect.ValueOf(obj).IsNil() {
		return &unstructured.Unstructured{Object: nil}, nil
	}
	ret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{Object: ret}, nil
}

func objectToResolveVal(r runtime.Object) (interface{}, error) {
	if r == nil || reflect.ValueOf(r).IsNil() {
		return nil, nil
	}
	v, err := convertObjectToUnstructured(r)
	if err != nil {
		return nil, err
	}
	return v.Object, nil
}

// ForInput evaluates the compiled CEL expressions converting them into CELEvaluations
// errors per evaluation are returned on the Evaluation object
// runtimeCELCostBudget was added for testing purpose only. Callers should always use const RuntimeCELCostBudget from k8s.io/apiserver/pkg/apis/cel/config.go as input.
func (f *filter) ForInput(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *admissionv1.AdmissionRequest, inputs OptionalVariableBindings, runtimeCELCostBudget int64) ([]EvaluationResult, int64, error) {
	// TODO: replace unstructured with ref.Val for CEL variables when native type support is available
	evaluations := make([]EvaluationResult, len(f.compilationResults))
	var err error

	oldObjectVal, err := objectToResolveVal(versionedAttr.VersionedOldObject)
	if err != nil {
		return nil, -1, err
	}
	objectVal, err := objectToResolveVal(versionedAttr.VersionedObject)
	if err != nil {
		return nil, -1, err
	}
	var paramsVal, authorizerVal, requestResourceAuthorizerVal any
	if inputs.VersionedParams != nil {
		paramsVal, err = objectToResolveVal(inputs.VersionedParams)
		if err != nil {
			return nil, -1, err
		}
	}

	if inputs.Authorizer != nil {
		authorizerVal = library.NewAuthorizerVal(versionedAttr.GetUserInfo(), inputs.Authorizer)
		requestResourceAuthorizerVal = library.NewResourceAuthorizerVal(versionedAttr.GetUserInfo(), inputs.Authorizer, versionedAttr)
	}

	requestVal, err := convertObjectToUnstructured(request)
	if err != nil {
		return nil, -1, err
	}
	va := &evaluationActivation{
		object:                    objectVal,
		oldObject:                 oldObjectVal,
		params:                    paramsVal,
		request:                   requestVal.Object,
		authorizer:                authorizerVal,
		requestResourceAuthorizer: requestResourceAuthorizerVal,
	}

	remainingBudget := runtimeCELCostBudget
	for i, compilationResult := range f.compilationResults {
		var evaluation = &evaluations[i]
		if compilationResult.ExpressionAccessor == nil { // in case of placeholder
			continue
		}
		evaluation.ExpressionAccessor = compilationResult.ExpressionAccessor
		if compilationResult.Error != nil {
			evaluation.Error = &cel.Error{
				Type:   cel.ErrorTypeInvalid,
				Detail: fmt.Sprintf("compilation error: %v", compilationResult.Error),
			}
			continue
		}
		if compilationResult.Program == nil {
			evaluation.Error = &cel.Error{
				Type:   cel.ErrorTypeInternal,
				Detail: fmt.Sprintf("unexpected internal error compiling expression"),
			}
			continue
		}
		t1 := time.Now()
		evalResult, evalDetails, err := compilationResult.Program.ContextEval(ctx, va)
		elapsed := time.Since(t1)
		evaluation.Elapsed = elapsed
		if evalDetails == nil {
			return nil, -1, &cel.Error{
				Type:   cel.ErrorTypeInternal,
				Detail: fmt.Sprintf("runtime cost could not be calculated for expression: %v, no further expression will be run", compilationResult.ExpressionAccessor.GetExpression()),
			}
		} else {
			rtCost := evalDetails.ActualCost()
			if rtCost == nil {
				return nil, -1, &cel.Error{
					Type:   cel.ErrorTypeInvalid,
					Detail: fmt.Sprintf("runtime cost could not be calculated for expression: %v, no further expression will be run", compilationResult.ExpressionAccessor.GetExpression()),
				}
			} else {
				if *rtCost > math.MaxInt64 || int64(*rtCost) > remainingBudget {
					return nil, -1, &cel.Error{
						Type:   cel.ErrorTypeInvalid,
						Detail: fmt.Sprintf("validation failed due to running out of cost budget, no further validation rules will be run"),
					}
				}
				remainingBudget -= int64(*rtCost)
			}
		}
		if err != nil {
			evaluation.Error = &cel.Error{
				Type:   cel.ErrorTypeInvalid,
				Detail: fmt.Sprintf("expression '%v' resulted in error: %v", compilationResult.ExpressionAccessor.GetExpression(), err),
			}
		} else {
			evaluation.EvalResult = evalResult
		}
	}

	return evaluations, remainingBudget, nil
}

// TODO: to reuse https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apiserver/pkg/admission/plugin/webhook/request/admissionreview.go#L154
func CreateAdmissionRequest(attr admission.Attributes) *admissionv1.AdmissionRequest {
	// FIXME: how to get resource GVK, GVR and subresource?
	gvk := attr.GetKind()
	gvr := attr.GetResource()
	subresource := attr.GetSubresource()

	requestGVK := attr.GetKind()
	requestGVR := attr.GetResource()
	requestSubResource := attr.GetSubresource()

	aUserInfo := attr.GetUserInfo()
	var userInfo authenticationv1.UserInfo
	if aUserInfo != nil {
		userInfo = authenticationv1.UserInfo{
			Extra:    make(map[string]authenticationv1.ExtraValue),
			Groups:   aUserInfo.GetGroups(),
			UID:      aUserInfo.GetUID(),
			Username: aUserInfo.GetName(),
		}
		// Convert the extra information in the user object
		for key, val := range aUserInfo.GetExtra() {
			userInfo.Extra[key] = authenticationv1.ExtraValue(val)
		}
	}

	dryRun := attr.IsDryRun()

	return &admissionv1.AdmissionRequest{
		Kind: metav1.GroupVersionKind{
			Group:   gvk.Group,
			Kind:    gvk.Kind,
			Version: gvk.Version,
		},
		Resource: metav1.GroupVersionResource{
			Group:    gvr.Group,
			Resource: gvr.Resource,
			Version:  gvr.Version,
		},
		SubResource: subresource,
		RequestKind: &metav1.GroupVersionKind{
			Group:   requestGVK.Group,
			Kind:    requestGVK.Kind,
			Version: requestGVK.Version,
		},
		RequestResource: &metav1.GroupVersionResource{
			Group:    requestGVR.Group,
			Resource: requestGVR.Resource,
			Version:  requestGVR.Version,
		},
		RequestSubResource: requestSubResource,
		Name:               attr.GetName(),
		Namespace:          attr.GetNamespace(),
		Operation:          admissionv1.Operation(attr.GetOperation()),
		UserInfo:           userInfo,
		// Leave Object and OldObject unset since we don't provide access to them via request
		DryRun: &dryRun,
		Options: runtime.RawExtension{
			Object: attr.GetOperationOptions(),
		},
	}
}

// CompilationErrors returns a list of all the errors from the compilation of the evaluator
func (e *filter) CompilationErrors() []error {
	compilationErrors := []error{}
	for _, result := range e.compilationResults {
		if result.Error != nil {
			compilationErrors = append(compilationErrors, result.Error)
		}
	}
	return compilationErrors
}
