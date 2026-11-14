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
	"errors"
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
	kubeSystemRole := bootstrappolicy.NamespaceRoles()["kube-system"][0]
	tests := []struct {
		name string
		// policy defaults to the bootstrap policy.
		policy *PolicyData
		// change edits the cluster before ensureRBACPolicy runs again.
		change func(*testing.T, clienttesting.ObjectTracker)
		// changeAfterList edits the cluster right after ensureRBACPolicy lists it.
		changeAfterList map[string]func(*testing.T, clienttesting.ObjectTracker)
		// wantRequests are the requests ensureRBACPolicy sends, in order.
		wantRequests []string
		// wantFailRequest is a List request that the test makes fail.
		wantFailRequest string
	}{
		{
			// Without drift, no bootstrap object needs its own Get.
			name: "already bootstrapped",
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"list roles in kube-system",
				"list rolebindings in kube-system",
			},
		},
		{
			name: "drifted objects are read live and repaired unless protected",
			change: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				updateObject(t, tracker, "clusterroles", "", "cluster-admin", func(role *rbacapiv1.ClusterRole) {
					role.Rules = nil
				})
				updateObject(t, tracker, "clusterroles", "", "system:monitoring", func(role *rbacapiv1.ClusterRole) {
					role.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
					role.Rules = nil
				})
				updateObject(t, tracker, "clusterrolebindings", "", "system:basic-user", func(binding *rbacapiv1.ClusterRoleBinding) {
					binding.RoleRef.Name = "custom-role"
				})
				updateObject(t, tracker, "roles", "kube-system", "extension-apiserver-authentication-reader", func(role *rbacapiv1.Role) {
					role.Rules = nil
				})
				updateObject(t, tracker, "rolebindings", "kube-system", "system:controller:bootstrap-signer", func(binding *rbacapiv1.RoleBinding) {
					binding.RoleRef.Name = "custom-role"
				})
			},
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"get clusterroles/cluster-admin", "update clusterroles/cluster-admin",
				"get clusterroles/system:monitoring", // protected, so not updated
				"get clusterrolebindings/system:basic-user", "delete clusterrolebindings/system:basic-user", "create clusterrolebindings/system:basic-user",
				"list roles in kube-system",
				"get roles/extension-apiserver-authentication-reader in kube-system", "update roles/extension-apiserver-authentication-reader in kube-system",
				"list rolebindings in kube-system",
				"get rolebindings/system:controller:bootstrap-signer in kube-system", "delete rolebindings/system:controller:bootstrap-signer in kube-system",
				"get namespaces/kube-system", "create rolebindings/system:controller:bootstrap-signer in kube-system",
			},
		},
		{
			name: "objects protected after the List are not written",
			change: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				updateObject(t, tracker, "clusterroles", "", "cluster-admin", func(role *rbacapiv1.ClusterRole) {
					role.Rules = nil
				})
				updateObject(t, tracker, "clusterrolebindings", "", "system:basic-user", func(binding *rbacapiv1.ClusterRoleBinding) {
					binding.RoleRef.Name = "custom-role"
				})
				updateObject(t, tracker, "roles", "kube-system", "extension-apiserver-authentication-reader", func(role *rbacapiv1.Role) {
					role.Rules = nil
				})
				updateObject(t, tracker, "rolebindings", "kube-system", "system::leader-locking-kube-scheduler", func(binding *rbacapiv1.RoleBinding) {
					binding.RoleRef.Name = "custom-role"
				})
			},
			changeAfterList: map[string]func(*testing.T, clienttesting.ObjectTracker){
				"list clusterrolebindings": func(t *testing.T, tracker clienttesting.ObjectTracker) {
					updateObject(t, tracker, "clusterroles", "", "cluster-admin", func(role *rbacapiv1.ClusterRole) {
						role.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
					})
					updateObject(t, tracker, "clusterrolebindings", "", "system:basic-user", func(binding *rbacapiv1.ClusterRoleBinding) {
						binding.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
					})
				},
				"list roles in kube-system": func(t *testing.T, tracker clienttesting.ObjectTracker) {
					updateObject(t, tracker, "roles", "kube-system", "extension-apiserver-authentication-reader", func(role *rbacapiv1.Role) {
						role.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
					})
				},
				"list rolebindings in kube-system": func(t *testing.T, tracker clienttesting.ObjectTracker) {
					updateObject(t, tracker, "rolebindings", "kube-system", "system::leader-locking-kube-scheduler", func(binding *rbacapiv1.RoleBinding) {
						binding.Annotations[rbacapiv1.AutoUpdateAnnotationKey] = "false"
					})
				},
			},
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"get clusterroles/cluster-admin",
				"get clusterrolebindings/system:basic-user",
				"list roles in kube-system",
				"get roles/extension-apiserver-authentication-reader in kube-system",
				"list rolebindings in kube-system",
				"get rolebindings/system::leader-locking-kube-scheduler in kube-system",
			},
		},
		{
			name: "a Role under the wrong namespace key is read live",
			policy: &PolicyData{
				Roles: map[string][]rbacapiv1.Role{"kube-public": {kubeSystemRole}},
			},
			change: func(t *testing.T, tracker clienttesting.ObjectTracker) {
				inKubePublic := kubeSystemRole.DeepCopy()
				inKubePublic.Namespace = "kube-public"
				if err := tracker.Add(inKubePublic); err != nil {
					t.Fatal(err)
				}
				updateObject(t, tracker, "roles", "kube-system", kubeSystemRole.Name, func(role *rbacapiv1.Role) {
					role.Rules = nil
				})
			},
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"list roles in kube-public",
				"get roles/extension-apiserver-authentication-reader in kube-system", "update roles/extension-apiserver-authentication-reader in kube-system",
			},
		},
		{
			name:            "a failed Role List falls back to live reads",
			wantFailRequest: "list roles in kube-system",
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"list roles in kube-system",
				"get roles/extension-apiserver-authentication-reader in kube-system",
				"get roles/system:controller:bootstrap-signer in kube-system",
				"get roles/system:controller:cloud-provider in kube-system",
				"get roles/system:controller:token-cleaner in kube-system",
				"get roles/system::leader-locking-kube-controller-manager in kube-system",
				"get roles/system::leader-locking-kube-scheduler in kube-system",
				"list rolebindings in kube-system",
			},
		},
		{
			name:            "a failed RoleBinding List falls back to live reads",
			wantFailRequest: "list rolebindings in kube-system",
			wantRequests: []string{
				"list clusterroles", "list clusterrolebindings",
				"list roles in kube-system",
				"list rolebindings in kube-system",
				"get rolebindings/system::extension-apiserver-authentication-reader in kube-system",
				"get rolebindings/system::leader-locking-kube-controller-manager in kube-system",
				"get rolebindings/system::leader-locking-kube-scheduler in kube-system",
				"get rolebindings/system:controller:bootstrap-signer in kube-system",
				"get rolebindings/system:controller:cloud-provider in kube-system",
				"get rolebindings/system:controller:token-cleaner in kube-system",
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
				policy = &PolicyData{
					ClusterRoles:               append(bootstrappolicy.ClusterRoles(), bootstrappolicy.ControllerRoles()...),
					ClusterRoleBindings:        append(bootstrappolicy.ClusterRoleBindings(), bootstrappolicy.ControllerRoleBindings()...),
					Roles:                      map[string][]rbacapiv1.Role{"kube-system": bootstrappolicy.NamespaceRoles()["kube-system"]},
					RoleBindings:               map[string][]rbacapiv1.RoleBinding{"kube-system": bootstrappolicy.NamespaceRoleBindings()["kube-system"]},
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
			client.PrependReactor("list", "*", func(action clienttesting.Action) (bool, runtime.Object, error) {
				request := requestString(action)
				if request == tc.wantFailRequest {
					return true, nil, errors.New("list failed")
				}
				handled, list, err := clienttesting.ObjectReaction(client.Tracker())(action)
				if change := tc.changeAfterList[request]; change != nil {
					// The change goes through the tracker because calling the fake
					// client from inside a reactor deadlocks.
					change(t, client.Tracker())
				}
				return handled, list, err
			})

			client.ClearActions()
			if done, err := ensureRBACPolicy(policy, client); err != nil || !done {
				t.Fatalf("ensureRBACPolicy() = (%v, %v), want (true, nil)", done, err)
			}
			var requests []string
			for _, action := range client.Actions() {
				requests = append(requests, requestString(action))
			}
			if diff := cmp.Diff(tc.wantRequests, requests); diff != "" {
				t.Errorf("requests (-want +got):\n%s", diff)
			}
		})
	}
}

// requestString describes a client request as "verb resource[/name][ in namespace]".
func requestString(action clienttesting.Action) string {
	request := action.GetVerb() + " " + action.GetResource().Resource
	switch action := action.(type) {
	case clienttesting.GetAction: // also matches Deletes
		request += "/" + action.GetName()
	case clienttesting.CreateAction: // also matches Updates
		request += "/" + action.GetObject().(metav1.Object).GetName()
	}
	if namespace := action.GetNamespace(); namespace != "" {
		request += " in " + namespace
	}
	return request
}

// updateObject edits the named RBAC object in tracker. namespace is empty for
// ClusterRoles and ClusterRoleBindings.
func updateObject[T runtime.Object](t *testing.T, tracker clienttesting.ObjectTracker, resource, namespace, name string, change func(T)) {
	t.Helper()
	gvr := rbacapiv1.SchemeGroupVersion.WithResource(resource)
	obj, err := tracker.Get(gvr, namespace, name)
	if err != nil {
		t.Fatal(err)
	}
	change(obj.(T))
	if err := tracker.Update(gvr, obj, namespace); err != nil {
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
