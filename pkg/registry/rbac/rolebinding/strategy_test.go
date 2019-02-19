/*
Copyright 2019 The Kubernetes Authors.
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

package rolebinding

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"testing"
)

func TestRoleBindingStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("RoleBinding must be namespace scoped")
	}
	if !Strategy.AllowCreateOnUpdate() {
		t.Errorf("RoleBinding should allow create on update")
	}

	oldRoleBinding := &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-role-binding",
			Namespace: metav1.NamespaceDefault,
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "test-valid-role-binding",
				Namespace: metav1.NamespaceDefault,
			},
		},
		RoleRef: rbac.RoleRef{
			Name:     "test-valid-role-binding",
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
		},
	}

	Strategy.PrepareForCreate(ctx, oldRoleBinding)

	errs := Strategy.Validate(ctx, oldRoleBinding)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newRoleBinding := &rbac.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-role-binding-2",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "4",
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "test-valid-role-binding",
				Namespace: metav1.NamespaceDefault,
			},
		},
		RoleRef: rbac.RoleRef{
			Name:     "test-valid-role-binding-2",
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
		},
	}

	errs = Strategy.Validate(ctx, newRoleBinding)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	Strategy.PrepareForUpdate(ctx, newRoleBinding, oldRoleBinding)

	errs = Strategy.ValidateUpdate(ctx, newRoleBinding, oldRoleBinding)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
