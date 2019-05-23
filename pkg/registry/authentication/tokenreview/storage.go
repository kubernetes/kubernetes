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
	"context"
	"errors"
	"fmt"
	"net/http"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/apis/authentication"
)

var badAuthenticatorAuds = apierrors.NewInternalError(errors.New("error validating audiences"))

type REST struct {
	tokenAuthenticator authenticator.Request
	apiAudiences       []string
}

func NewREST(tokenAuthenticator authenticator.Request, apiAudiences []string) *REST {
	return &REST{
		tokenAuthenticator: tokenAuthenticator,
		apiAudiences:       apiAudiences,
	}
}

func (r *REST) NamespaceScoped() bool {
	return false
}

func (r *REST) New() runtime.Object {
	return &authentication.TokenReview{}
}

func (r *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
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

	if createValidation != nil {
		if err := createValidation(ctx, obj.DeepCopyObject()); err != nil {
			return nil, err
		}
	}

	if r.tokenAuthenticator == nil {
		return tokenReview, nil
	}

	// create a header that contains nothing but the token
	fakeReq := &http.Request{Header: http.Header{}}
	fakeReq.Header.Add("Authorization", "Bearer "+tokenReview.Spec.Token)

	auds := tokenReview.Spec.Audiences
	if len(auds) == 0 {
		auds = r.apiAudiences
	}
	if len(auds) > 0 {
		fakeReq = fakeReq.WithContext(authenticator.WithAudiences(fakeReq.Context(), auds))
	}

	resp, ok, err := r.tokenAuthenticator.AuthenticateRequest(fakeReq)
	tokenReview.Status.Authenticated = ok
	if err != nil {
		tokenReview.Status.Error = err.Error()
	}

	if len(auds) > 0 && resp != nil && len(authenticator.Audiences(auds).Intersect(resp.Audiences)) == 0 {
		klog.Errorf("error validating audience. want=%q got=%q", auds, resp.Audiences)
		return nil, badAuthenticatorAuds
	}

	if resp != nil && resp.User != nil {
		tokenReview.Status.User = authentication.UserInfo{
			Username: resp.User.GetName(),
			UID:      resp.User.GetUID(),
			Groups:   resp.User.GetGroups(),
			Extra:    map[string]authentication.ExtraValue{},
		}
		for k, v := range resp.User.GetExtra() {
			tokenReview.Status.User.Extra[k] = authentication.ExtraValue(v)
		}
		tokenReview.Status.Audiences = resp.Audiences
	}

	return tokenReview, nil
}
