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

package localsubjectaccessreview

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
}

func NewREST(authorizer authorizer.Authorizer) *REST {
	return &REST{authorizer}
}

func (r *REST) NamespaceScoped() bool {
	return true
}

func (r *REST) New() runtime.Object {
	return &authorizationapi.LocalSubjectAccessReview{}
}

// Destroy cleans up resources on shutdown.
func (r *REST) Destroy() {
	// Given no underlying store, we don't destroy anything
	// here explicitly.
}

var _ rest.SingularNameProvider = &REST{}

func (r *REST) GetSingularName() string {
	return "localsubjectaccessreview"
}

func (r *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	localSubjectAccessReview, ok := obj.(*authorizationapi.LocalSubjectAccessReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a LocaLocalSubjectAccessReview: %#v", obj))
	}
	// clear fields if the featuregate is disabled
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
		if localSubjectAccessReview.Spec.ResourceAttributes != nil {
			localSubjectAccessReview.Spec.ResourceAttributes.FieldSelector = nil
			localSubjectAccessReview.Spec.ResourceAttributes.LabelSelector = nil
		}
	}

	if errs := authorizationvalidation.ValidateLocalSubjectAccessReview(localSubjectAccessReview); len(errs) > 0 {
		return nil, apierrors.NewInvalid(authorizationapi.Kind(localSubjectAccessReview.Kind), "", errs)
	}
	namespace := genericapirequest.NamespaceValue(ctx)
	if len(namespace) == 0 {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("namespace is required on this type: %v", namespace))
	}
	if namespace != localSubjectAccessReview.Namespace {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("spec.resourceAttributes.namespace must match namespace: %v", namespace))
	}

	if createValidation != nil {
		if err := createValidation(ctx, obj.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	authorizationAttributes := authorizationutil.AuthorizationAttributesFrom(localSubjectAccessReview.Spec)
	decision, reason, evaluationErr := r.authorizer.Authorize(ctx, authorizationAttributes)

	localSubjectAccessReview.Status = authorizationapi.SubjectAccessReviewStatus{
		Allowed: (decision == authorizer.DecisionAllow),
		Denied:  (decision == authorizer.DecisionDeny),
		Reason:  reason,
	}
	localSubjectAccessReview.Status.EvaluationError = authorizationutil.BuildEvaluationError(evaluationErr, authorizationAttributes)

	return localSubjectAccessReview, nil
}
