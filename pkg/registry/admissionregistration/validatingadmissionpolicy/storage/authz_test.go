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

package storage

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/registry/rest/resttest"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
)

func TestAuthorization(t *testing.T) {
	for _, tc := range []struct {
		name             string
		userInfo         user.Info
		auth             AuthFunc
		resourceResolver resolver.ResourceResolverFunc
		expectErr        bool
	}{
		{
			name:      "superuser",
			userInfo:  &user.DefaultInfo{Groups: []string{user.SystemPrivilegedGroup}},
			expectErr: false, // success despite always-denying authorizer
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				return authorizer.DecisionDeny, "", nil
			},
		},
		{
			name:     "authorized",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				if a.GetResource() == "replicalimits" {
					return authorizer.DecisionAllow, "", nil
				}
				return authorizer.DecisionDeny, "", nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "rules.example.com",
					Version:  "v1",
					Resource: "replicalimits",
				}, nil
			},
			expectErr: false,
		},
		{
			name:     "denied",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				if a.GetResource() == "configmaps" {
					return authorizer.DecisionAllow, "", nil
				}
				return authorizer.DecisionDeny, "", nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "rules.example.com",
					Version:  "v1",
					Resource: "replicalimits",
				}, nil
			},
			expectErr: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			storage, server := newStorage(t, tc.auth, tc.resourceResolver)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()
			test := resttest.New(t, storage).ClusterScope()
			t.Run("create", func(t *testing.T) {
				ctx := request.WithUser(test.TestContext(), tc.userInfo)
				_, err := storage.Create(ctx, validValidatingAdmissionPolicy(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
				if (err != nil) != tc.expectErr {
					t.Errorf("expected error: %v but got error: %v", tc.expectErr, err)
				}
			})
			t.Run("update", func(t *testing.T) {
				ctx := request.WithUser(test.TestContext(), tc.userInfo)
				obj := validValidatingAdmissionPolicy()
				_, _, err := storage.Update(ctx, obj.Name, rest.DefaultUpdatedObjectInfo(obj, func(ctx context.Context, newObj runtime.Object, oldObj runtime.Object) (transformedNewObj runtime.Object, err error) {
					object := oldObj.(*admissionregistration.ValidatingAdmissionPolicy)
					object.Labels = map[string]string{"c": "d"}
					return object, nil
				}), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
				if (err != nil) != tc.expectErr {
					t.Errorf("expected error: %v but got error: %v", tc.expectErr, err)
				}
			})
		})
	}
}

type AuthFunc func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error)

func (f AuthFunc) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return f(ctx, a)
}
