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

	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	corev1fake "k8s.io/client-go/kubernetes/typed/core/v1/fake"
	rbacv1fake "k8s.io/client-go/kubernetes/typed/rbac/v1/fake"
	clienttesting "k8s.io/client-go/testing"
)

type staticInfoVisitor struct {
	info *resource.Info
}

func (v *staticInfoVisitor) Visit(fn resource.VisitorFunc) error {
	return fn(v.info, nil)
}

func TestReconcileBindingReturnsError(t *testing.T) {
	tests := map[string]struct {
		obj             runtime.Object
		failingResource string
	}{
		"role binding": {
			obj: &rbacv1.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-binding", Namespace: metav1.NamespaceDefault},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: "test-role"},
				Subjects:   []rbacv1.Subject{{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "test-user"}},
			},
			failingResource: "rolebindings",
		},
		"cluster role binding": {
			obj: &rbacv1.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{Name: "test-cluster-binding"},
				RoleRef:    rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "ClusterRole", Name: "test-cluster-role"},
				Subjects:   []rbacv1.Subject{{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: "test-user"}},
			},
			failingResource: "clusterrolebindings",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			rbacClient := &rbacv1fake.FakeRbacV1{Fake: &clienttesting.Fake{}}
			groupResource := schema.GroupResource{Group: rbacv1.GroupName, Resource: tc.failingResource}
			rbacClient.PrependReactor("get", tc.failingResource, func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewForbidden(groupResource, "test", errors.New("rejected by test"))
			})

			o := &ReconcileOptions{
				FilenameOptions: &resource.FilenameOptions{},
				Visitor:         &staticInfoVisitor{info: &resource.Info{Object: tc.obj}},
				RBACClient:      rbacClient,
				NamespaceClient: &corev1fake.FakeCoreV1{Fake: &clienttesting.Fake{}},
				PrintObject: func(_ runtime.Object, _ io.Writer) error {
					return nil
				},
				IOStreams: genericiooptions.NewTestIOStreamsDiscard(),
			}
			if err := o.Validate(); err != nil {
				t.Fatalf("unexpected validation error: %v", err)
			}

			err := o.RunReconcile()
			if err == nil {
				t.Fatal("expected an error when the server rejects the reconcile request")
			}
			if !strings.Contains(err.Error(), "forbidden") {
				t.Errorf("expected the underlying forbidden error, got: %v", err)
			}
			if !strings.Contains(err.Error(), tc.failingResource) {
				t.Errorf("expected the error to reference %q, got: %v", tc.failingResource, err)
			}
		})
	}
}
