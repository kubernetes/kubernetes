/*
Copyright The Kubernetes Authors.

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
	"errors"
	"io"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
)

// singleObjectVisitor implements Visit(), so RunReconcile can be exercised directly.
type singleObjectVisitor struct {
	object runtime.Object
}

func (v singleObjectVisitor) Visit(fn resource.VisitorFunc) error {
	return fn(&resource.Info{Object: v.object}, nil)
}

func TestRunReconcile(t *testing.T) {
	testCases := []struct {
		name          string
		object        runtime.Object
		reactVerb     string
		reactResource string
		wantErr       string
	}{
		{
			name: "RoleBinding reconciles successfully",
			object: &rbacv1.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-rb", Namespace: "test-ns"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "test-role"},
			},
		},
		{
			name: "ClusterRoleBinding reconciles successfully",
			object: &rbacv1.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-crb"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "test-clusterrole"},
			},
		},
		{
			name: "RoleBinding forbidden returns error",
			object: &rbacv1.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-rb", Namespace: "test-ns"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "test-role"},
			},
			reactVerb:     "get",
			reactResource: "rolebindings",
			wantErr:       "forbidden",
		},
		{
			name: "ClusterRoleBinding forbidden returns error",
			object: &rbacv1.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-crb"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "test-clusterrole"},
			},
			reactVerb:     "get",
			reactResource: "clusterrolebindings",
			wantErr:       "forbidden",
		},
		{
			name: "unsupported version returns error",
			object: &rbacv1beta1.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-rb", Namespace: "test-ns"},
			},
			wantErr: "only rbac.authorization.k8s.io/v1 is supported",
		},
		{
			name:   "unknown type is skipped",
			object: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-pod"}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			if tc.reactResource != "" {
				client.PrependReactor(tc.reactVerb, tc.reactResource,
					func(action clienttesting.Action) (bool, runtime.Object, error) {
						return true, nil, apierrors.NewForbidden(action.GetResource().GroupResource(), "", errors.New("forbidden"))
					})
			}

			streams, _, _, _ := genericiooptions.NewTestIOStreams()
			o := NewReconcileOptions(streams)
			o.RBACClient = client.RbacV1()
			o.NamespaceClient = client.CoreV1()
			o.PrintObject = func(_ runtime.Object, _ io.Writer) error { return nil }
			o.Visitor = singleObjectVisitor{object: tc.object}

			err := o.RunReconcile()
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
			}
		})
	}
}
