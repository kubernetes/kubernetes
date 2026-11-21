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
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	rbacapiv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
			if len(requests) < 2 {
				t.Fatalf("missing initial policy Lists: %v", requests)
			}
			if diff := cmp.Diff(tc.wantRequests[:2], requests[:2]); diff != "" {
				t.Errorf("initial Lists (-want +got):\n%s", diff)
			}
			// Different objects can now interleave, but each object's live-read
			// and write sequence must remain exactly as before.
			groupRequests := func(requests []string) map[string][]string {
				byObject := map[string][]string{}
				for _, request := range requests {
					verb, object, _ := strings.Cut(request, " ")
					byObject[object] = append(byObject[object], verb)
				}
				return byObject
			}
			if diff := cmp.Diff(groupRequests(tc.wantRequests[2:]), groupRequests(requests[2:])); diff != "" {
				t.Errorf("per-object requests (-want +got):\n%s", diff)
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

func TestReconcileBootstrapPolicyBoundsAndErrors(t *testing.T) {
	objects := make([]int, 12)
	for i := range objects {
		objects[i] = i
	}
	var active, maximum atomic.Int32
	visits := make([]atomic.Int32, len(objects))
	started := make(chan int, len(objects))
	release := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	done := make(chan []error, 1)
	wantErr := errors.New("reconcile failure")
	go func() {
		done <- reconcileBootstrapPolicy(objects, func(index int) error {
			visits[index].Add(1)
			n := active.Add(1)
			defer active.Add(-1)
			for old := maximum.Load(); n > old; old = maximum.Load() {
				if maximum.CompareAndSwap(old, n) {
					break
				}
			}
			started <- index
			<-release
			if index == 3 {
				return wantErr
			}
			return nil
		})
	}()
	for range 4 {
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("four independent objects did not reconcile concurrently")
		}
	}
	select {
	case <-started:
		t.Fatal("more than four reconciliations ran concurrently")
	case <-time.After(20 * time.Millisecond):
	}
	releaseOnce.Do(func() { close(release) })
	select {
	case errs := <-done:
		if len(errs) != len(objects) || !errors.Is(errs[3], wantErr) {
			t.Fatalf("lost or misattributed reconciliation error: %v", errs)
		}
		for i := range objects {
			if visits[i].Load() != 1 || (i != 3 && errs[i] != nil) {
				t.Errorf("object %d: visits=%d error=%v", i, visits[i].Load(), errs[i])
			}
		}
	case <-time.After(5 * time.Second):
		t.Fatal("reconciliation did not finish")
	}
	if maximum.Load() != 4 {
		t.Errorf("maximum concurrent reconciliations = %d, want 4", maximum.Load())
	}
}

func TestBootstrapRolesCompleteBeforeBindings(t *testing.T) {
	policy := &PolicyData{}
	for i := range 8 {
		name := fmt.Sprintf("role-%d", i)
		policy.ClusterRoles = append(policy.ClusterRoles, rbacapiv1.ClusterRole{ObjectMeta: metav1.ObjectMeta{Name: name}})
		policy.ClusterRoleBindings = append(policy.ClusterRoleBindings, rbacapiv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			RoleRef:    rbacapiv1.RoleRef{APIGroup: rbacapiv1.GroupName, Kind: "ClusterRole", Name: name},
		})
	}
	client := fake.NewSimpleClientset()
	client.PrependReactor("create", "clusterrolebindings", func(clienttesting.Action) (bool, runtime.Object, error) {
		for _, role := range policy.ClusterRoles {
			if _, err := client.Tracker().Get(rbacapiv1.SchemeGroupVersion.WithResource("clusterroles"), "", role.Name); err != nil {
				return true, nil, fmt.Errorf("binding phase started before role %s completed: %w", role.Name, err)
			}
		}
		return false, nil, nil
	})
	if done, err := ensureRBACPolicy(policy, client); err != nil || !done {
		t.Fatalf("bootstrap = (%v, %v), want (true, nil)", done, err)
	}
}

func TestBootstrapPartialFailureStillReconcilesOtherObjects(t *testing.T) {
	policy := &PolicyData{ClusterRoles: []rbacapiv1.ClusterRole{
		{ObjectMeta: metav1.ObjectMeta{Name: "denied"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "allowed"}},
	}}
	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "clusterroles", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.(clienttesting.GetAction).GetName() == "denied" {
			return true, nil, apierrors.NewForbidden(rbacapiv1.Resource("clusterroles"), "denied", errors.New("denied for test"))
		}
		return false, nil, nil
	})
	if done, err := ensureRBACPolicy(policy, client); err != nil || done {
		t.Fatalf("partial failure must request another attempt: done=%v err=%v", done, err)
	}
	if _, err := client.Tracker().Get(rbacapiv1.SchemeGroupVersion.WithResource("clusterroles"), "", "allowed"); err != nil {
		t.Fatalf("one failure prevented another independent role from reconciling: %v", err)
	}
}

func TestBootstrapNamespaceChecksRemainLive(t *testing.T) {
	policy := &PolicyData{Roles: bootstrappolicy.NamespaceRoles(), RoleBindings: bootstrappolicy.NamespaceRoleBindings()}
	client := fake.NewSimpleClientset()
	if done, err := ensureRBACPolicy(policy, client); err != nil || !done {
		t.Fatalf("bootstrap = (%v, %v)", done, err)
	}
	want := 0
	for _, roles := range policy.Roles {
		want += len(roles)
	}
	for _, bindings := range policy.RoleBindings {
		want += len(bindings)
	}
	got := 0
	for _, action := range client.Actions() {
		if action.GetVerb() == "get" && action.GetResource().Resource == "namespaces" {
			got++
		}
	}
	if got != want {
		t.Errorf("namespace probes were cached or omitted: got %d live Gets, want %d", got, want)
	}
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
