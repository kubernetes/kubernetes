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

package policybased

import (
	"context"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/apis/rbac"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	"k8s.io/kubernetes/pkg/registry/rbac/validation"
)

func TestEscalation(t *testing.T) {
	createContext := request.WithRequestInfo(request.WithNamespace(context.TODO(), "myns"), &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "create",
		APIGroup:          "rbac.authorization.k8s.io",
		APIVersion:        "v1",
		Namespace:         "myns",
		Resource:          "roles",
		Name:              "",
	})
	updateContext := request.WithRequestInfo(request.WithNamespace(context.TODO(), "myns"), &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "update",
		APIGroup:          "rbac.authorization.k8s.io",
		APIVersion:        "v1",
		Namespace:         "myns",
		Resource:          "roles",
		Name:              "myrole",
	})

	superuser := &user.DefaultInfo{Name: "superuser", Groups: []string{"system:masters"}}
	bob := &user.DefaultInfo{Name: "bob"}
	steve := &user.DefaultInfo{Name: "steve"}
	alice := &user.DefaultInfo{Name: "alice"}

	authzCalled := 0
	fakeStorage := &fakeStorage{}
	fakeAuthorizer := authorizer.AuthorizerFunc(func(attr authorizer.Attributes) (authorizer.Decision, string, error) {
		authzCalled++
		if attr.GetUser().GetName() == "steve" {
			return authorizer.DecisionAllow, "", nil
		}
		return authorizer.DecisionNoOpinion, "", nil
	})
	fakeRuleResolver, _ := validation.NewTestRuleResolver(
		nil,
		nil,
		[]*rbacv1.ClusterRole{{ObjectMeta: metav1.ObjectMeta{Name: "alice-role"}, Rules: []rbacv1.PolicyRule{{APIGroups: []string{"*"}, Resources: []string{"*"}, Verbs: []string{"*"}}}}},
		[]*rbacv1.ClusterRoleBinding{{RoleRef: rbacv1.RoleRef{Name: "alice-role", APIGroup: "rbac.authorization.k8s.io", Kind: "ClusterRole"}, Subjects: []rbacv1.Subject{{Name: "alice", Kind: "User", APIGroup: "rbac.authorization.k8s.io"}}}},
	)

	role := &rbac.Role{
		ObjectMeta: metav1.ObjectMeta{Name: "myrole", Namespace: "myns"},
		Rules:      []rbac.PolicyRule{{APIGroups: []string{""}, Verbs: []string{"get"}, Resources: []string{"pods"}}},
	}

	s := NewStorage(fakeStorage, fakeAuthorizer, fakeRuleResolver)

	testcases := []struct {
		name          string
		user          user.Info
		expectAllowed bool
		expectAuthz   bool
	}{
		// superuser doesn't even trigger an authz check, and is allowed
		{
			name:          "superuser",
			user:          superuser,
			expectAuthz:   false,
			expectAllowed: true,
		},
		// bob triggers an authz check, is disallowed by the authorizer, and has no RBAC permissions, so is not allowed
		{
			name:          "bob",
			user:          bob,
			expectAuthz:   true,
			expectAllowed: false,
		},
		// steve triggers an authz check, is allowed by the authorizer, and has no RBAC permissions, but is still allowed
		{
			name:          "steve",
			user:          steve,
			expectAuthz:   true,
			expectAllowed: true,
		},
		// alice triggers an authz check, is denied by the authorizer, but has RBAC permissions in the fakeRuleResolver, so is allowed
		{
			name:          "alice",
			user:          alice,
			expectAuthz:   true,
			expectAllowed: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			authzCalled, fakeStorage.created, fakeStorage.updated = 0, 0, 0
			_, err := s.Create(request.WithUser(createContext, tc.user), role, nil, nil)

			if tc.expectAllowed {
				if err != nil {
					t.Error(err)
					return
				}
				if fakeStorage.created != 1 {
					t.Errorf("unexpected calls to underlying storage.Create: %d", fakeStorage.created)
					return
				}
			} else {
				if !errors.IsForbidden(err) {
					t.Errorf("expected forbidden, got %v", err)
					return
				}
				if fakeStorage.created != 0 {
					t.Errorf("unexpected calls to underlying storage.Create: %d", fakeStorage.created)
					return
				}
			}

			if tc.expectAuthz != (authzCalled > 0) {
				t.Fatalf("expected authz=%v, saw %d calls", tc.expectAuthz, authzCalled)
			}

			authzCalled, fakeStorage.created, fakeStorage.updated = 0, 0, 0
			_, _, err = s.Update(request.WithUser(updateContext, tc.user), role.Name, rest.DefaultUpdatedObjectInfo(role), nil, nil, false, nil)

			if tc.expectAllowed {
				if err != nil {
					t.Error(err)
					return
				}
				if fakeStorage.updated != 1 {
					t.Errorf("unexpected calls to underlying storage.Update: %d", fakeStorage.updated)
					return
				}
			} else {
				if !errors.IsForbidden(err) {
					t.Errorf("expected forbidden, got %v", err)
					return
				}
				if fakeStorage.updated != 0 {
					t.Errorf("unexpected calls to underlying storage.Update: %d", fakeStorage.updated)
					return
				}
			}

			if tc.expectAuthz != (authzCalled > 0) {
				t.Fatalf("expected authz=%v, saw %d calls", tc.expectAuthz, authzCalled)
			}
		})
	}
}

type fakeStorage struct {
	updated int
	created int
	rest.StandardStorage
}

func (f *fakeStorage) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	f.created++
	return nil, nil
}

func (f *fakeStorage) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	obj, err := objInfo.UpdatedObject(ctx, &rbac.Role{})
	if err != nil {
		return obj, false, err
	}
	f.updated++
	return nil, false, nil
}
