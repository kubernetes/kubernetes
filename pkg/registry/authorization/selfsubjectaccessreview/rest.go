/*
Copyright 2016 The Kubernetes Authors.

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

package selfsubjectaccessreview

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	authorizationvalidation "k8s.io/kubernetes/pkg/apis/authorization/validation"
	authorizationutil "k8s.io/kubernetes/pkg/registry/authorization/util"
)

type REST struct {
	authorizer authorizer.Authorizer
	scheme     *runtime.Scheme
}

func NewREST(authorizer authorizer.Authorizer, scheme *runtime.Scheme) *REST {
	return &REST{authorizer, scheme}
}

func (r *REST) NamespaceScoped() bool {
	return false
}

func (r *REST) New() runtime.Object {
	return &authorizationapi.SelfSubjectAccessReview{}
}

// Destroy cleans up resources on shutdown.
func (r *REST) Destroy() {
	// Given no underlying store, we don't destroy anything
	// here explicitly.
}

var _ rest.SingularNameProvider = &REST{}

func (r *REST) GetSingularName() string {
	return "selfsubjectaccessreview"
}

func (r *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	selfSAR, ok := obj.(*authorizationapi.SelfSubjectAccessReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a SelfSubjectAccessReview: %#v", obj))
	}

	// Clear status so it's not taken into account during input validation.
	// This is important, as we cannot make validation stricter; in k8s 1.36 and before, the client was able to pass a bogus status without a validation error.
	selfSAR.Status = authorizationapi.SubjectAccessReviewStatus{}

	// Clear the options for opting into conditions-awareness when the feature gate is off.
	// This means that we fallback to the conditions-unaware Authorize when the feature gate is
	// off, even though both the client and authorizer might have supported conditions.
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) {
		selfSAR.Spec.AuthorizationOptions = nil
	}

	if errs := authorizationvalidation.ValidateSelfSubjectAccessReviewCreate(ctx, r.scheme, selfSAR); len(errs) > 0 {
		return nil, apierrors.NewInvalid(authorizationapi.Kind(selfSAR.Kind), "", errs)
	}

	userToCheck, exists := genericapirequest.UserFrom(ctx)
	if !exists {
		return nil, apierrors.NewBadRequest("no user present on request")
	}

	if createValidation != nil {
		if err := createValidation(ctx, obj.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	var authorizationAttributes authorizer.AttributesRecord
	if selfSAR.Spec.ResourceAttributes != nil {
		authorizationAttributes = authorizationutil.ResourceAttributesFrom(userToCheck, *selfSAR.Spec.ResourceAttributes)
	} else {
		authorizationAttributes = authorizationutil.NonResourceAttributesFrom(userToCheck, *selfSAR.Spec.NonResourceAttributes)
	}

	// Only activate Conditional Authorization if both the server and client supports it
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) && selfSAR.Spec.AuthorizationOptions.SupportsConditionalAuthorization() {
		conditionsAwareDecision := r.authorizer.ConditionsAwareAuthorize(ctx, authorizationAttributes)
		selfSAR.Status = authorizationutil.ConditionsAwareDecisionToSARStatus(ctx, authorizationAttributes, conditionsAwareDecision)

	} else if selfSAR.Spec.AuthorizationOptions.SupportsUnconditionalAuthorization() {
		// conditions-unaware flow, feature gate is off or client does not support conditions
		decision, reason, evaluationErr := r.authorizer.Authorize(ctx, authorizationAttributes)

		selfSAR.Status = authorizationapi.SubjectAccessReviewStatus{
			Allowed:         (decision == authorizer.DecisionAllow),
			Denied:          (decision == authorizer.DecisionDeny),
			Reason:          reason,
			EvaluationError: authorizationutil.BuildEvaluationError(evaluationErr, authorizationAttributes),
		}
	} else {
		// the HandledDecisionTypes was neither [Allow, ConditionsMap, Deny, NoOpinion, Union] or [Allow, Deny, NoOpinion], reject it.
		return nil, apierrors.NewBadRequest(fmt.Sprintf("unsupported client-handled decision types: %v", sets.List(selfSAR.Spec.AuthorizationOptions.GetHandledDecisionTypes())))
	}

	return selfSAR, nil
}
