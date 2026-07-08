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

package auth

import (
	"strings"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
)

// singleObjectVisitor visits exactly one in-memory object, without touching the
// filesystem or a REST builder, so RunReconcile can be exercised directly.
type singleObjectVisitor struct {
	object runtime.Object
}

func (v singleObjectVisitor) Visit(fn resource.VisitorFunc) error {
	return fn(&resource.Info{Object: v.object}, nil)
}

// TestRunReconcileReturnsErrorOnForbidden is a regression test for a nil-pointer
// panic: the *rbacv1.RoleBinding and *rbacv1.ClusterRoleBinding cases in
// RunReconcile called result.RoleBinding.GetObject() without checking the error
// returned by the reconcile Run(). When the reconcile Get was forbidden, result
// was nil and the dereference panicked. RunReconcile must instead return the error.
// See https://github.com/kubernetes/kubernetes/issues/XXXXX (argoproj/argo-cd#28607).
func TestRunReconcileReturnsErrorOnForbidden(t *testing.T) {
	forbidden := func(gr schema.GroupResource, name string) clienttesting.ReactionFunc {
		return func(action clienttesting.Action) (bool, runtime.Object, error) {
			return true, nil, apierrors.NewForbidden(gr, name, nil)
		}
	}

	testCases := []struct {
		name     string
		object   runtime.Object
		verb     string
		resource string
		grName   string
	}{
		{
			name: "RoleBinding forbidden",
			object: &rbacv1.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-rb", Namespace: "test-ns"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "test-role"},
			},
			verb:     "get",
			resource: "rolebindings",
			grName:   "test-rb",
		},
		{
			name: "ClusterRoleBinding forbidden",
			object: &rbacv1.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-crb"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "test-clusterrole"},
			},
			verb:     "get",
			resource: "clusterrolebindings",
			grName:   "test-crb",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			client.PrependReactor(tc.verb, tc.resource,
				forbidden(schema.GroupResource{Group: rbacv1.GroupName, Resource: tc.resource}, tc.grName))

			streams, _, _, _ := genericiooptions.NewTestIOStreams()
			o := NewReconcileOptions(streams)
			o.RBACClient = client.RbacV1()
			o.NamespaceClient = client.CoreV1()
			o.Visitor = singleObjectVisitor{object: tc.object}

			// Before the fix this panics with a nil-pointer dereference.
			// After the fix it must return the forbidden error.
			err := o.RunReconcile()
			if err == nil {
				t.Fatalf("expected a forbidden error, got nil")
			}
			if !apierrors.IsForbidden(err) && !strings.Contains(err.Error(), "forbidden") {
				t.Fatalf("expected a forbidden error, got: %v", err)
			}
		})
	}
}
