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

package selfsubjectreview

import (
	"context"
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
)

var _ interface {
	rest.Creater
	rest.NamespaceScopedStrategy
	rest.Scoper
	rest.Storage
} = &REST{}

// REST implements a RESTStorage for selfsubjectreviews.
type REST struct {
}

// NewREST returns a RESTStorage object that will work against selfsubjectreviews.
func NewREST() *REST {
	return &REST{}
}

// NamespaceScoped fulfill rest.Scoper
func (r *REST) NamespaceScoped() bool {
	return false
}

// New creates a new selfsubjectreview object.
func (r *REST) New() runtime.Object {
	return &authenticationapi.SelfSubjectReview{}
}

// Destroy cleans up resources on shutdown.
func (r *REST) Destroy() {
	// Given no underlying store, we don't destroy anything
	// here explicitly.
}

// Create returns attributes of the subject making the request.
func (r *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	if createValidation != nil {
		if err := createValidation(ctx, obj.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	_, ok := obj.(*authenticationapi.SelfSubjectReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a SelfSubjectReview: %#v", obj))
	}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return nil, apierrors.NewBadRequest("no user present on request")
	}

	extra := user.GetExtra()

	selfSR := &authenticationapi.SelfSubjectReview{
		ObjectMeta: metav1.ObjectMeta{
			CreationTimestamp: metav1.NewTime(time.Now()),
		},
		Status: authenticationapi.SelfSubjectReviewStatus{
			UserInfo: authenticationapi.UserInfo{
				Username: user.GetName(),
				UID:      user.GetUID(),
				Groups:   user.GetGroups(),
				Extra:    make(map[string]authenticationapi.ExtraValue, len(extra)),
			},
		},
	}
	for key, attr := range extra {
		selfSR.Status.UserInfo.Extra[key] = attr
	}

	return selfSR, nil
}

var _ rest.SingularNameProvider = &REST{}

func (r *REST) GetSingularName() string {
	return "selfsubjectreview"
}
