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

package clusterrolebinding

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"testing"
)

func TestClusterRoleBindingStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("ClusterRoleBinding must not be namespace scoped")
	}
	if !Strategy.AllowCreateOnUpdate() {
		t.Errorf("ClusterRoleBinding should allow create on update")
	}

	oldClusterRoleBinding := &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-cluster-role-binding",
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "test-valid-cluster-role-binding",
				Namespace: metav1.NamespaceDefault,
			},
		},
		RoleRef: rbac.RoleRef{
			Name:     "test-valid-cluster-role-binding",
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
		},
	}

	Strategy.PrepareForCreate(ctx, oldClusterRoleBinding)

	errs := Strategy.Validate(ctx, oldClusterRoleBinding)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newClusterRoleBinding := &rbac.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-cluster-role-binding-2",
			ResourceVersion: "4",
		},
		Subjects: []rbac.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "test-valid-cluster-role-binding",
				Namespace: metav1.NamespaceDefault,
			},
		},
		RoleRef: rbac.RoleRef{
			Name:     "test-valid-cluster-role-binding-2",
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
		},
	}

	errs = Strategy.Validate(ctx, newClusterRoleBinding)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	Strategy.PrepareForUpdate(ctx, newClusterRoleBinding, oldClusterRoleBinding)

	errs = Strategy.ValidateUpdate(ctx, newClusterRoleBinding, oldClusterRoleBinding)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
