/*
Copyright 2018 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"

	authenticationapiv1 "k8s.io/api/authentication/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	authenticationvalidation "k8s.io/kubernetes/pkg/apis/authentication/validation"
	api "k8s.io/kubernetes/pkg/apis/core"
	token "k8s.io/kubernetes/pkg/serviceaccount"
)

func (r *TokenREST) New() runtime.Object {
	return &authenticationapi.TokenRequest{}
}

type TokenREST struct {
	svcaccts             getter
	pods                 getter
	secrets              getter
	issuer               token.TokenGenerator
	auds                 authenticator.Audiences
	maxExpirationSeconds int64
}

var _ = rest.NamedCreater(&TokenREST{})
var _ = rest.GroupVersionKindProvider(&TokenREST{})

var gvk = schema.GroupVersionKind{
	Group:   authenticationapiv1.SchemeGroupVersion.Group,
	Version: authenticationapiv1.SchemeGroupVersion.Version,
	Kind:    "TokenRequest",
}

func (r *TokenREST) Create(ctx context.Context, name string, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	if err := createValidation(obj); err != nil {
		return nil, err
	}

	out := obj.(*authenticationapi.TokenRequest)

	if errs := authenticationvalidation.ValidateTokenRequest(out); len(errs) != 0 {
		return nil, errors.NewInvalid(gvk.GroupKind(), "", errs)
	}

	svcacctObj, err := r.svcaccts.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	svcacct := svcacctObj.(*api.ServiceAccount)

	var (
		pod    *api.Pod
		secret *api.Secret
	)

	if ref := out.Spec.BoundObjectRef; ref != nil {
		var uid types.UID

		gvk := schema.FromAPIVersionAndKind(ref.APIVersion, ref.Kind)
		switch {
		case gvk.Group == "" && gvk.Kind == "Pod":
			newCtx := newContext(ctx, "pods", ref.Name, gvk)
			podObj, err := r.pods.Get(newCtx, ref.Name, &metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			pod = podObj.(*api.Pod)
			if name != pod.Spec.ServiceAccountName {
				return nil, errors.NewBadRequest(fmt.Sprintf("cannot bind token for serviceaccount %q to pod running with different serviceaccount name.", name))
			}
			uid = pod.UID
		case gvk.Group == "" && gvk.Kind == "Secret":
			newCtx := newContext(ctx, "secrets", ref.Name, gvk)
			secretObj, err := r.secrets.Get(newCtx, ref.Name, &metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			secret = secretObj.(*api.Secret)
			uid = secret.UID
		default:
			return nil, errors.NewBadRequest(fmt.Sprintf("cannot bind token to object of type %s", gvk.String()))
		}
		if ref.UID != "" && uid != ref.UID {
			return nil, errors.NewConflict(schema.GroupResource{Group: gvk.Group, Resource: gvk.Kind}, ref.Name, fmt.Errorf("the UID in the bound object reference (%s) does not match the UID in record. The object might have been deleted and then recreated", ref.UID))
		}
	}
	if len(out.Spec.Audiences) == 0 {
		out.Spec.Audiences = r.auds
	}

	if r.maxExpirationSeconds > 0 && out.Spec.ExpirationSeconds > r.maxExpirationSeconds {
		//only positive value is valid
		out.Spec.ExpirationSeconds = r.maxExpirationSeconds
	}

	sc, pc := token.Claims(*svcacct, pod, secret, out.Spec.ExpirationSeconds, out.Spec.Audiences)
	tokdata, err := r.issuer.GenerateToken(sc, pc)
	if err != nil {
		return nil, fmt.Errorf("failed to generate token: %v", err)
	}

	out.Status = authenticationapi.TokenRequestStatus{
		Token:               tokdata,
		ExpirationTimestamp: metav1.Time{Time: sc.Expiry.Time()},
	}
	return out, nil
}

func (r *TokenREST) GroupVersionKind(schema.GroupVersion) schema.GroupVersionKind {
	return gvk
}

type getter interface {
	Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error)
}

// newContext return a copy of ctx in which new RequestInfo is set
func newContext(ctx context.Context, resource, name string, gvk schema.GroupVersionKind) context.Context {
	oldInfo, found := genericapirequest.RequestInfoFrom(ctx)
	if !found {
		return ctx
	}
	newInfo := genericapirequest.RequestInfo{
		IsResourceRequest: true,
		Verb:              "get",
		Namespace:         oldInfo.Namespace,
		Resource:          resource,
		Name:              name,
		Parts:             []string{resource, name},
		APIGroup:          gvk.Group,
		APIVersion:        gvk.Version,
	}
	return genericapirequest.WithRequestInfo(ctx, &newInfo)
}
