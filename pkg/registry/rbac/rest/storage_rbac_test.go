/*
Copyright 2021 The Kubernetes Authors.

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

package rest

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	rbacapiv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/plugin/pkg/auth/authorizer/rbac/bootstrappolicy"
)

func TestEnsureRBACPolicy(t *testing.T) {
	tests := []struct {
		name string
		// policy defaults to the cluster-scoped bootstrap policy.
		policy *PolicyData
		// change edits the cluster before ensureRBACPolicy runs again.
		change func(*testing.T, clienttesting.ObjectTracker)
		// changeAfterList edits the cluster after ensureRBACPolicy lists it.
		changeAfterList func(*testing.T, clienttesting.ObjectTracker)
		// wantRequests are the requests ensureRBACPolicy sends, in order.
		wantRequests []string
	}{
		{
			// Without drift, no bootstrap object needs its own Get.
			name:         "already bootstrapped",
			wantRequests: []string{"list clusterroles", "list clusterrolebindings"},
		},
		{
			name: "drifted objects are read live and repaired unless protected",
			change: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				updateObject(t, tracker, "clusterroles", "cluster-admin", func(role *rbacapiv1.ClusterRole) {
					role.Rules = nil
				})
				updateObject(t, tracker, "clusterroles", "system:monitoring", func(role *rbacapiv1.ClusterRole) {
					role.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
					role.Rules = nil
				})
				updateObject(t, tracker, "clusterrolebindings", "system:basic-user", func(binding *rbacapiv1.ClusterRoleBinding) {
					binding.RoleRef.Name = "custom-role"
				})
			},
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"get clusterroles/cluster-admin", "update clusterroles/cluster-admin",
				"get clusterroles/system:monitoring", // protected, so not updated
				"get clusterrolebindings/system:basic-user", "delete clusterrolebindings/system:basic-user", "create clusterrolebindings/system:basic-user",
			},
		},
		{
			name: "objects protected after the List are not written",
			change: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				updateObject(t, tracker, "clusterroles", "cluster-admin", func(role *rbacapiv1.ClusterRole) {
					role.Rules = nil
				})
				updateObject(t, tracker, "clusterrolebindings", "system:basic-user", func(binding *rbacapiv1.ClusterRoleBinding) {
					binding.RoleRef.Name = "custom-role"
				})
			},
			changeAfterList: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				updateObject(t, tracker, "clusterroles", "cluster-admin", func(role *rbacapiv1.ClusterRole) {
					role.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
				})
				updateObject(t, tracker, "clusterrolebindings", "system:basic-user", func(binding *rbacapiv1.ClusterRoleBinding) {
					binding.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
				})
			},
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"get clusterroles/cluster-admin",
				"get clusterrolebindings/system:basic-user",
			},
		},
		{
			name: "legacy roles and bindings are copied",
			policy: &PolicyData{
				ClusterRolesToAggregate:    bootstrappolicy.ClusterRolesToAggregate(),
				ClusterRoleBindingsToSplit: bootstrappolicy.ClusterRoleBindingsToSplit(),
			},
			change: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				for _, obj := range []runtime.Object{
					&rbacapiv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: "admin"}},
					&rbacapiv1.ClusterRoleBinding{ObjectMeta: metav1.ObjectMeta{Name: "system:discovery"}},
				} {
					if err := tracker.Add(obj); err != nil {
						t.Fatal(err)
					}
				}
			},
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"get clusterroles/system:aggregate-to-admin", "get clusterroles/admin", "create clusterroles/system:aggregate-to-admin",
				"get clusterrolebindings/system:discovery", "get clusterrolebindings/system:public-info-viewer", "create clusterrolebindings/system:public-info-viewer",
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			policy := tc.policy
			if policy == nil {
				// Leave out the namespaced Roles and RoleBindings. Every run reads
				// each of them with its own Get, so they would add the same
				// requests to every case's wantRequests.
				policy = &PolicyData{
					ClusterRoles:               append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...),
					ClusterRoleBindings:        append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...),
					ClusterRolesToAggregate:    bootstrappolicy.ClusterRolesToAggregate(),
					ClusterRoleBindingsToSplit: bootstrappolicy.ClusterRoleBindingsToSplit(),
				}
			}
			client := fake.NewSimpleClientset()
			if done, err := ensureRBACPolicy(policy, client); err != nil || !done {
				t.Fatalf("bootstrap ensureRBACPolicy() = (%v, %v), want (true, nil)", done, err)
			}
			if tc.change != nil {
				tc.change(t, client.Tracker())
			}
			if tc.changeAfterList != nil {
				// ensureRBACPolicy lists ClusterRoles, then ClusterRoleBindings, then
				// reconciles. Making the change right after the second List leaves
				// both List results out of date before any object is reconciled.
				// The change goes through the tracker because calling the fake
				// client from inside a reactor deadlocks.
				client.PrependReactor("list", "clusterrolebindings", func(action clienttesting.Action) (bool, runtime.Object, error) {
					handled, list, err := clienttesting.ObjectReaction(client.Tracker())(action)
					tc.changeAfterList(t, client.Tracker())
					return handled, list, err
				})
			}

			client.ClearActions()
			if done, err := ensureRBACPolicy(policy, client); err != nil || !done {
				t.Fatalf("ensureRBACPolicy() = (%v, %v), want (true, nil)", done, err)
			}
			var requests []string
			for _, action := range client.Actions() {
				request := action.GetVerb() + " " + action.GetResource().Resource
				switch action := action.(type) {
				case clienttesting.GetAction: // also matches Deletes
					request += "/" + action.GetName()
				case clienttesting.CreateAction: // also matches Updates
					request += "/" + action.GetObject().(metav1.Object).GetName()
				}
				requests = append(requests, request)
			}
			if diff := cmp.Diff(tc.wantRequests, requests); diff != "" {
				t.Errorf("requests (-want +got):\n%s", diff)
			}
		})
	}
}

// updateObject edits the named ClusterRole or ClusterRoleBinding in tracker.
func updateObject[T runtime.Object](t *testing.T, tracker clienttesting.ObjectTracker, resource, name string, change func(T)) {
	t.Helper()
	gvr := rbacapiv1.SchemeGroupVersion.WithResource(resource)
	obj, err := tracker.Get(gvr, "", name)
	if err != nil {
		t.Fatal(err)
	}
	change(obj.(T))
	if err := tracker.Update(gvr, obj, ""); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkEnsureRBACPolicy(b *testing.B) {
	newPolicy := func() *PolicyData {
		return &PolicyData{
			ClusterRoles:               append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...),
			ClusterRoleBindings:        append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...),
			Roles:                      bootstrappolicy.NamespaceRoles(),
			RoleBindings:               bootstrappolicy.NamespaceRoleBindings(),
			ClusterRolesToAggregate:    bootstrappolicy.ClusterRolesToAggregate(),
			ClusterRoleBindingsToSplit: bootstrappolicy.ClusterRoleBindingsToSplit(),
		}
	}
	b.Run("empty-cluster", func(b *testing.B) {
		b.ReportAllocs()
		requests := 0
		for n := 0; n < b.N; n++ {
			b.StopTimer()
			policy := newPolicy()
			coreClientSet := fake.NewSimpleClientset()
			b.StartTimer()
			_, _ = ensureRBACPolicy(policy, coreClientSet)
			requests += len(coreClientSet.Actions())
		}
		b.ReportMetric(float64(requests)/float64(b.N), "requests/op")
	})
	b.Run("already-bootstrapped", func(b *testing.B) {
		policy := newPolicy()
		coreClientSet := fake.NewSimpleClientset()
		_, _ = ensureRBACPolicy(policy, coreClientSet)
		b.ReportAllocs()
		b.ResetTimer()
		requests := 0
		for n := 0; n < b.N; n++ {
			coreClientSet.ClearActions()
			_, _ = ensureRBACPolicy(policy, coreClientSet)
			requests += len(coreClientSet.Actions())
		}
		b.ReportMetric(float64(requests)/float64(b.N), "requests/op")
	})
}
