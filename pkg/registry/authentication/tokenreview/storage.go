/*
Copyright 2015 The Kubernetes Authors.

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

package tokenreview

import (
	"fmt"
	"net/http"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/authentication"
)

type REST struct {
	tokenAuthenticator authenticator.Request
}

func NewREST(tokenAuthenticator authenticator.Request) *REST {
	return &REST{tokenAuthenticator: tokenAuthenticator}
}

func (r *REST) New() runtime.Object {
	return &authentication.TokenReview{}
}

func (r *REST) Create(ctx genericapirequest.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	tokenReview, ok := obj.(*authentication.TokenReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a TokenReview: %#v", obj))
	}
	namespace := genericapirequest.NamespaceValue(ctx)
	if len(namespace) != 0 {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("namespace is not allowed on this type: %v", namespace))
	}

	if len(tokenReview.Spec.Token) == 0 {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("token is required for TokenReview in authentication"))
	}

	if r.tokenAuthenticator == nil {
		return tokenReview, nil
	}

	// create a header that contains nothing but the token
	fakeReq := &http.Request{Header: http.Header{}}
	fakeReq.Header.Add("Authorization", "Bearer "+tokenReview.Spec.Token)

	tokenUser, ok, err := r.tokenAuthenticator.AuthenticateRequest(fakeReq)
	tokenReview.Status.Authenticated = ok
	if err != nil {
		tokenReview.Status.Error = err.Error()
	}
	if tokenUser != nil {
		tokenReview.Status.User = authentication.UserInfo{
			Username: tokenUser.GetName(),
			UID:      tokenUser.GetUID(),
			Groups:   tokenUser.GetGroups(),
			Extra:    map[string]authentication.ExtraValue{},
		}
		for k, v := range tokenUser.GetExtra() {
			tokenReview.Status.User.Extra[k] = authentication.ExtraValue(v)
		}
	}

	return tokenReview, nil
}
