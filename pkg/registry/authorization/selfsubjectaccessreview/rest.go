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
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
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

func (r *REST) New() runtime.Object {
	return &authorizationapi.SelfSubjectAccessReview{}
}

func (r *REST) Create(ctx genericapirequest.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	selfSAR, ok := obj.(*authorizationapi.SelfSubjectAccessReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a SelfSubjectAccessReview: %#v", obj))
	}
	if errs := authorizationvalidation.ValidateSelfSubjectAccessReview(selfSAR); len(errs) > 0 {
		return nil, apierrors.NewInvalid(authorizationapi.Kind(selfSAR.Kind), "", errs)
	}
	userToCheck, exists := genericapirequest.UserFrom(ctx)
	if !exists {
		return nil, apierrors.NewBadRequest("no user present on request")
	}

	var authorizationAttributes authorizer.AttributesRecord
	if selfSAR.Spec.ResourceAttributes != nil {
		authorizationAttributes = authorizationutil.ResourceAttributesFrom(userToCheck, *selfSAR.Spec.ResourceAttributes)
	} else {
		authorizationAttributes = authorizationutil.NonResourceAttributesFrom(userToCheck, *selfSAR.Spec.NonResourceAttributes)
	}

	allowed, reason, evaluationErr := r.authorizer.Authorize(authorizationAttributes)

	selfSAR.Status = authorizationapi.SubjectAccessReviewStatus{
		Allowed: allowed,
		Reason:  reason,
	}
	if evaluationErr != nil {
		selfSAR.Status.EvaluationError = evaluationErr.Error()
	}

	return selfSAR, nil
}
