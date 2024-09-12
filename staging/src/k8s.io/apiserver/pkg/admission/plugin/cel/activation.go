/*
Copyright 2024 The Kubernetes Authors.

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
	"time"

	admissionv1 "k8s.io/api/admission/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

// newActivation creates an activation for CEL admission plugins from the given request, admission chain and
// variable binding information.
func newActivation(ctx context.Context, versionedAttr *admission.VersionedAttributes, request *admissionv1.AdmissionRequest, inputs OptionalVariableBindings, namespace *v1.Namespace) (*evaluationActivation, error) {
	// if this activation supports composition, we will need the compositionCtx. It may be nil.
	compositionCtx, _ := ctx.(CompositionContext)

	var err error

	oldObjectVal, err := objectToResolveVal(versionedAttr.VersionedOldObject)
	if err != nil {
		return nil, err
	}
	objectVal, err := objectToResolveVal(versionedAttr.VersionedObject)
	if err != nil {
		return nil, err
	}
	var paramsVal, authorizerVal, requestResourceAuthorizerVal any
	if inputs.VersionedParams != nil {
		paramsVal, err = objectToResolveVal(inputs.VersionedParams)
		if err != nil {
			return nil, err
		}
	}

	if inputs.Authorizer != nil {
		authorizerVal = library.NewAuthorizerVal(versionedAttr.GetUserInfo(), inputs.Authorizer)
		requestResourceAuthorizerVal = library.NewResourceAuthorizerVal(versionedAttr.GetUserInfo(), inputs.Authorizer, versionedAttr)
	}

	requestVal, err := convertObjectToUnstructured(request)
	if err != nil {
		return nil, err
	}
	namespaceVal, err := objectToResolveVal(namespace)
	if err != nil {
		return nil, err
	}
	va := &evaluationActivation{
		object:                    objectVal,
		oldObject:                 oldObjectVal,
		params:                    paramsVal,
		request:                   requestVal.Object,
		namespace:                 namespaceVal,
		authorizer:                authorizerVal,
		requestResourceAuthorizer: requestResourceAuthorizerVal,
	}

	// composition is an optional feature that only applies for ValidatingAdmissionPolicy and MutatingAdmissionPolicy.
	if compositionCtx != nil {
		va.variables = compositionCtx.Variables(va)
	}
	return va, nil
}

// evaluateWithActivation evaluates a compiled CEL admission plugin expression using the provided activation and CEL
// runtime cost budget.
func evaluateWithActivation(ctx context.Context, activation *evaluationActivation, compilationResult CompilationResult, remainingBudget int64) (EvaluationResult, int64, error) {
	// if this evaluation supports composition, we will need the compositionCtx. It may be nil.
	compositionCtx, _ := ctx.(CompositionContext)

	var evaluation = EvaluationResult{}
	if compilationResult.ExpressionAccessor == nil { // in case of placeholder
		return evaluation, remainingBudget, nil
	}

	evaluation.ExpressionAccessor = compilationResult.ExpressionAccessor
	if compilationResult.Error != nil {
		evaluation.Error = &cel.Error{
			Type:   cel.ErrorTypeInvalid,
			Detail: fmt.Sprintf("compilation error: %v", compilationResult.Error),
			Cause:  compilationResult.Error,
		}
		return evaluation, remainingBudget, nil
	}
	if compilationResult.Program == nil {
		evaluation.Error = &cel.Error{
			Type:   cel.ErrorTypeInternal,
			Detail: "unexpected internal error compiling expression",
		}
		return evaluation, remainingBudget, nil
	}
	t1 := time.Now()
	evalResult, evalDetails, err := compilationResult.Program.ContextEval(ctx, activation)
	// budget may be spent due to lazy evaluation of composited variables
	if compositionCtx != nil {
		compositionCost := compositionCtx.GetAndResetCost()
		if compositionCost > remainingBudget {
			return evaluation, -1, &cel.Error{
				Type:   cel.ErrorTypeInvalid,
				Detail: "validation failed due to running out of cost budget, no further validation rules will be run",
				Cause:  cel.ErrOutOfBudget,
			}
		}
		remainingBudget -= compositionCost
	}
	elapsed := time.Since(t1)
	evaluation.Elapsed = elapsed
	if evalDetails == nil {
		return evaluation, -1, &cel.Error{
			Type:   cel.ErrorTypeInternal,
			Detail: fmt.Sprintf("runtime cost could not be calculated for expression: %v, no further expression will be run", compilationResult.ExpressionAccessor.GetExpression()),
		}
	} else {
		rtCost := evalDetails.ActualCost()
		if rtCost == nil {
			return evaluation, -1, &cel.Error{
				Type:   cel.ErrorTypeInvalid,
				Detail: fmt.Sprintf("runtime cost could not be calculated for expression: %v, no further expression will be run", compilationResult.ExpressionAccessor.GetExpression()),
				Cause:  cel.ErrOutOfBudget,
			}
		} else {
			if *rtCost > math.MaxInt64 || int64(*rtCost) > remainingBudget {
				return evaluation, -1, &cel.Error{
					Type:   cel.ErrorTypeInvalid,
					Detail: "validation failed due to running out of cost budget, no further validation rules will be run",
					Cause:  cel.ErrOutOfBudget,
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
	return evaluation, remainingBudget, nil
}
