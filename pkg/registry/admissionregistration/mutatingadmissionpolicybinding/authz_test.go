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

package mutatingadmissionpolicybinding

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"
)

func TestAuthorization(t *testing.T) {
	for _, tc := range []struct {
		name              string
		userInfo          user.Info
		auth              AuthFunc
		policyGetter      PolicyGetterFunc
		resourceResolver  resolver.ResourceResolverFunc
		expectErrContains string
	}{
		{
			name:     "superuser", // success despite always-denying authorizer
			userInfo: &user.DefaultInfo{Groups: []string{user.SystemPrivilegedGroup}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				return authorizer.DecisionDeny, "", nil
			},
		},
		{
			name:     "authorized",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				if a.GetResource() == "configmaps" {
					return authorizer.DecisionAllow, "", nil
				}
				return authorizer.DecisionDeny, "", nil
			},
			policyGetter: func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
				return &admissionregistration.MutatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "replicalimit-policy.example.com"},
					Spec: admissionregistration.MutatingAdmissionPolicySpec{
						ParamKind: &admissionregistration.ParamKind{Kind: "ConfigMap", APIVersion: "v1"},
					},
				}, nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "",
					Version:  "v1",
					Resource: "configmaps",
				}, nil
			},
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
			policyGetter: func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
				return &admissionregistration.MutatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "replicalimit-policy.example.com"},
					Spec: admissionregistration.MutatingAdmissionPolicySpec{
						ParamKind: &admissionregistration.ParamKind{Kind: "Params", APIVersion: "foo.example.com/v1"},
					},
				}, nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "foo.example.com",
					Version:  "v1",
					Resource: "params",
				}, nil
			},
			expectErrContains: "permission on the object referenced by paramRef",
		},
		{
			name:     "deny but relevant fields not updated",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				return authorizer.DecisionDeny, "", nil
			},
			policyGetter: func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
				return &admissionregistration.MutatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "replicalimit-policy.example.com"},
					Spec: admissionregistration.MutatingAdmissionPolicySpec{
						ParamKind: &admissionregistration.ParamKind{Kind: "Params", APIVersion: "foo.example.com/v1"},
					},
				}, nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "foo.example.com",
					Version:  "v1",
					Resource: "params",
				}, nil
			},
		},
		{
			name:     "unable to parse paramRef",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				if a.GetResource() == "configmaps" {
					return authorizer.DecisionAllow, "", nil
				}
				return authorizer.DecisionDeny, "", nil
			},
			policyGetter: func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
				return &admissionregistration.MutatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "replicalimit-policy.example.com"},
					Spec: admissionregistration.MutatingAdmissionPolicySpec{
						ParamKind: &admissionregistration.ParamKind{Kind: "Params", APIVersion: "foo.example.com/v1"},
					},
				}, nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "",
					Version:  "v1",
					Resource: "configmaps",
				}, nil
			},
			expectErrContains: "unable to parse paramKind &{foo.example.com/v1 Params} to determine minimum required permissions",
		},
		{
			name:     "unable to resolve param",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				if a.GetResource() == "configmaps" {
					return authorizer.DecisionAllow, "", nil
				}
				return authorizer.DecisionDeny, "", nil
			},
			policyGetter: func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
				return &admissionregistration.MutatingAdmissionPolicy{
					ObjectMeta: metav1.ObjectMeta{Name: "replicalimit-policy.example.com"},
					Spec: admissionregistration.MutatingAdmissionPolicySpec{
						ParamKind: &admissionregistration.ParamKind{Kind: "Params", APIVersion: "foo.example.com/v1"},
					},
				}, nil
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{}, &meta.NoKindMatchError{GroupKind: gvk.GroupKind(), SearchedVersions: []string{gvk.Version}}
			},
			expectErrContains: "unable to resolve paramKind &{foo.example.com/v1 Params} to determine minimum required permissions",
		},
		{
			name:     "unable to get policy",
			userInfo: &user.DefaultInfo{Groups: []string{user.AllAuthenticated}},
			auth: func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
				if a.GetResource() == "configmaps" {
					return authorizer.DecisionAllow, "", nil
				}
				return authorizer.DecisionDeny, "", nil
			},
			policyGetter: func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
				return nil, fmt.Errorf("no such policy")
			},
			resourceResolver: func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
				return schema.GroupVersionResource{
					Group:    "",
					Version:  "v1",
					Resource: "configmaps",
				}, nil
			},
			expectErrContains: "unable to get policy replicalimit-policy.example.com to determine minimum required permissions",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			strategy := NewStrategy(tc.auth, tc.policyGetter, tc.resourceResolver)
			t.Run("create", func(t *testing.T) {
				ctx := request.WithUser(context.Background(), tc.userInfo)
				for _, obj := range validPolicyBindings() {
					errs := strategy.Validate(ctx, obj)
					if len(errs) > 0 && !strings.Contains(errors.Join(errs.ToAggregate().Errors()...).Error(), tc.expectErrContains) {
						t.Errorf("expected error to contain: %v but got error: %v", tc.expectErrContains, errs)
					}
				}
			})
			t.Run("update", func(t *testing.T) {
				ctx := request.WithUser(context.Background(), tc.userInfo)
				for _, obj := range validPolicyBindings() {
					objWithChangedParamRef := obj.DeepCopy()
					if pr := objWithChangedParamRef.Spec.ParamRef; pr != nil {
						if len(pr.Name) > 0 {
							pr.Name = "changed"
						}

						if pr.Selector != nil {
							pr.Selector = &metav1.LabelSelector{
								MatchLabels: map[string]string{
									"changed": "value",
								},
							}
						}

						if len(pr.Namespace) > 0 {
							pr.Namespace = "othernamespace"
						}

						if pr.ParameterNotFoundAction == nil || *pr.ParameterNotFoundAction == admissionregistration.AllowAction {
							v := admissionregistration.DenyAction
							pr.ParameterNotFoundAction = &v
						} else {
							v := admissionregistration.AllowAction
							pr.ParameterNotFoundAction = &v
						}
					}
					errs := strategy.ValidateUpdate(ctx, obj, objWithChangedParamRef)
					if len(errs) > 0 && !strings.Contains(errors.Join(errs.ToAggregate().Errors()...).Error(), tc.expectErrContains) {
						t.Errorf("expected error to contain: %v but got error: %v", tc.expectErrContains, errs)
					}
				}
			})
		})
	}
}

type AuthFunc func(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error)

func (f AuthFunc) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	return f(ctx, a)
}

type PolicyGetterFunc func(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error)

func (f PolicyGetterFunc) GetMutatingAdmissionPolicy(ctx context.Context, name string) (*admissionregistration.MutatingAdmissionPolicy, error) {
	return f(ctx, name)
}
