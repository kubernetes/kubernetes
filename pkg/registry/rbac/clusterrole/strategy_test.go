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

package clusterrole

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"testing"
)

func TestClusterRoleStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("ClusterRole must not be namespace scoped")
	}
	if !Strategy.AllowCreateOnUpdate() {
		t.Errorf("ClusterRole should allow create on update")
	}

	oldClusterRole := &rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-clusterRole",
		},
		Rules: []rbac.PolicyRule{
			{
				APIGroups:     []string{""},
				ResourceNames: []string{},
				Resources: []string{
					"endpoints",
					"nodes",
					"pods",
					"services",
				},
				Verbs: []string{
					"get",
					"list",
					"watch",
				},
			},
		},
	}

	Strategy.PrepareForCreate(ctx, oldClusterRole)

	errs := Strategy.Validate(ctx, oldClusterRole)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newClusterRole := &rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-clusterRole-2",
			ResourceVersion: "4",
		},
		Rules: []rbac.PolicyRule{
			{
				APIGroups:     []string{""},
				ResourceNames: []string{},
				Resources: []string{
					"endpoints",
					"nodes",
					"pods",
					"services",
				},
				Verbs: []string{
					"get",
					"list",
					"watch",
				},
			},
		},
	}

	errs = Strategy.Validate(ctx, newClusterRole)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	Strategy.PrepareForUpdate(ctx, newClusterRole, oldClusterRole)

	errs = Strategy.ValidateUpdate(ctx, newClusterRole, oldClusterRole)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
