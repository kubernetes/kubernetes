/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package subjectaccessreview

import (
	"fmt"

	kapi "k8s.io/kubernetes/pkg/api"
	kapierrors "k8s.io/kubernetes/pkg/api/errors"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization.k8s.io"
	authorizationvalidation "k8s.io/kubernetes/pkg/apis/authorization.k8s.io/validation"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	authorizationutil "k8s.io/kubernetes/pkg/registry/authorization/util"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// REST implements the RESTStorage interface in terms of an Registry.
type REST struct {
	authorizer authorizer.Authorizer
}

// NewREST creates a new REST for policies.
func NewREST(authorizer authorizer.Authorizer) *REST {
	return &REST{authorizer}
}

// New creates a new ResourceAccessReview object
func (r *REST) New() runtime.Object {
	return &authorizationapi.SubjectAccessReview{}
}

// Create registers a given new ResourceAccessReview instance to r.registry.
func (r *REST) Create(ctx kapi.Context, obj runtime.Object) (runtime.Object, error) {
	subjectAccessReview, ok := obj.(*authorizationapi.SubjectAccessReview)
	if !ok {
		return nil, kapierrors.NewBadRequest(fmt.Sprintf("not a SubjectAccessReview: %#v", obj))
	}
	if errs := authorizationvalidation.ValidateSubjectAccessReview(subjectAccessReview); len(errs) > 0 {
		return nil, kapierrors.NewInvalid(authorizationapi.Kind(subjectAccessReview.Kind), "", errs)
	}
	if subjectAccessReview.Spec == nil {
		return nil, kapierrors.NewInvalid(authorizationapi.Kind(subjectAccessReview.Kind), "", field.ErrorList{field.Required(field.NewPath("spec"), "")})
	}

	userToCheck := &user.DefaultInfo{
		Name:   subjectAccessReview.Spec.User,
		Groups: subjectAccessReview.Spec.Groups,
	}

	var authorizationAttributes authorizer.AttributesRecord
	if subjectAccessReview.Spec.ResourceAttributes != nil {
		authorizationAttributes = authorizationutil.ResourceAttributesFrom(userToCheck, *subjectAccessReview.Spec.ResourceAttributes)
	} else {
		authorizationAttributes = authorizationutil.NonResourceAttributesFrom(userToCheck, *subjectAccessReview.Spec.NonResourceAttributes)
	}

	denyError := r.authorizer.Authorize(authorizationAttributes)

	subjectAccessReview.Status = &authorizationapi.SubjectAccessReviewStatus{}
	subjectAccessReview.Status.Allowed = (denyError == nil)
	if denyError != nil {
		subjectAccessReview.Status.Reason = denyError.Error()
	}

	// clear Spec so we don't serialize it back
	subjectAccessReview.Spec = nil

	return subjectAccessReview, nil
}
