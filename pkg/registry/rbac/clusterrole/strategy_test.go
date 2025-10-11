/*
Copyright 2025 The Kubernetes Authors.

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
package clusterrole_test

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/registry/rbac/clusterrole"
)

func TestClusterRole_DeclarativeValidation_VerbsRequired(t *testing.T) {
	ctx := context.Background()
	strategy := clusterrole.Strategy

	role := &rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: "test-role"},
		Rules: []rbac.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods"},
			},
		},
	}

	errs := strategy.Validate(ctx, role)
	if len(errs) == 0 {
		t.Errorf("expected validation error for missing Verbs, got none")
	}
}
