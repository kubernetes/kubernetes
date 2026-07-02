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

package storage

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func restoreContext() context.Context {
	return genericapirequest.WithUser(genericapirequest.WithNamespace(context.Background(), "team-a"), &user.DefaultInfo{Name: "alice"})
}

func preparedRestore(t *testing.T) (*api.Pod, *nodev1alpha1.PodCheckpoint) {
	t.Helper()
	return preparedRestoreFromPod(t, validNewPod())
}

func preparedRestoreFromPod(t *testing.T, source *api.Pod) (*api.Pod, *nodev1alpha1.PodCheckpoint) {
	t.Helper()
	pod := source.DeepCopy()
	pod.Namespace = "team-a"
	var live v1.Pod
	if err := legacyscheme.Scheme.Convert(pod, &live, nil); err != nil {
		t.Fatal(err)
	}
	checkpoint := &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "cp-1", Namespace: pod.Namespace, UID: "checkpoint-uid"},
		Status: nodev1alpha1.PodCheckpointStatus{
			NodeName:                ptr.To("node-1"),
			Conditions:              []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue}},
			CheckpointedPodTemplate: checkpointutil.SanitizePodTemplate(&live),
		},
	}
	live.Spec = *checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	checkpointutil.AddRestoreNodeAffinity(&live.Spec, "node-1")
	if err := legacyscheme.Scheme.Convert(&live, pod, nil); err != nil {
		t.Fatal(err)
	}
	pod.Spec.RestoreFrom = &api.CheckpointReference{Name: checkpoint.Name, UID: checkpoint.UID}
	pod.Finalizers = []string{nodeapi.PodRestoreProtectionFinalizer}
	return pod, checkpoint
}

func TestRestorePolicyCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	for _, tc := range []struct {
		name                 string
		mutatePod            func(*api.Pod)
		mutateCheckpoint     func(*nodev1alpha1.PodCheckpoint)
		ctx                  func() context.Context
		decision             authorizer.Decision
		authzErr             error
		readErr              error
		missingAuthorizer    bool
		missingClient        bool
		cancelDuringRead     bool
		wantError            string
		wantInvalid          bool
		wantAuthz, wantReads int
	}{
		{name: "missing checkpoint UID", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) { p.Spec.RestoreFrom.UID = "" }, wantError: "spec.restoreFrom.uid", wantAuthz: 1, wantReads: 1},
		{name: "replacement checkpoint", decision: authorizer.DecisionAllow, mutateCheckpoint: func(c *nodev1alpha1.PodCheckpoint) { c.UID = "replacement" }, wantError: "spec.restoreFrom.uid", wantAuthz: 1, wantReads: 1},
		{name: "prepared restore", decision: authorizer.DecisionAllow, wantAuthz: 1, wantReads: 1},
		{name: "ordinary pod", mutatePod: func(p *api.Pod) { p.Spec.RestoreFrom = nil }},
		{name: "denied before read", decision: authorizer.DecisionDeny, wantError: "not authorized", wantAuthz: 1},
		{name: "no opinion before read", decision: authorizer.DecisionNoOpinion, wantError: "not authorized", wantAuthz: 1},
		{name: "authorization error", authzErr: errors.New("authorization backend unavailable"), wantError: "authorization backend unavailable", wantAuthz: 1},
		{name: "missing authorizer", missingAuthorizer: true, wantError: "requires an authorizer"},
		{name: "missing user", ctx: func() context.Context { return genericapirequest.WithNamespace(context.Background(), "team-a") }, wantError: "requires a user"},
		{name: "missing namespace", ctx: func() context.Context {
			return genericapirequest.WithUser(context.Background(), &user.DefaultInfo{Name: "alice"})
		}, wantError: "requires a namespace"},
		{name: "canceled", ctx: func() context.Context { ctx, cancel := context.WithCancel(restoreContext()); cancel(); return ctx }, wantError: "context canceled"},
		{name: "canceled during checkpoint read", decision: authorizer.DecisionAllow, cancelDuringRead: true, wantError: "context canceled", wantAuthz: 1, wantReads: 1},
		{name: "missing checkpoint client", decision: authorizer.DecisionAllow, missingClient: true, wantError: "requires a PodCheckpoint client", wantAuthz: 1},
		{name: "checkpoint read error", decision: authorizer.DecisionAllow, readErr: errors.New("checkpoint unavailable"), wantError: "checkpoint unavailable", wantAuthz: 1, wantReads: 1},
		{name: "checkpoint missing", decision: authorizer.DecisionAllow, readErr: apierrors.NewNotFound(nodev1alpha1.Resource("podcheckpoints"), "cp-1"), wantError: "not found", wantAuthz: 1, wantReads: 1},
		{name: "deleting checkpoint", decision: authorizer.DecisionAllow, mutateCheckpoint: func(c *nodev1alpha1.PodCheckpoint) { now := metav1.Now(); c.DeletionTimestamp = &now }, wantError: "being deleted", wantAuthz: 1, wantReads: 1},
		{name: "not ready", decision: authorizer.DecisionAllow, mutateCheckpoint: func(c *nodev1alpha1.PodCheckpoint) { c.Status.Conditions = nil }, wantError: "not ready", wantAuthz: 1, wantReads: 1},
		{name: "missing captured template", decision: authorizer.DecisionAllow, mutateCheckpoint: func(c *nodev1alpha1.PodCheckpoint) { c.Status.CheckpointedPodTemplate = nil }, wantError: "checkpointedPodTemplate", wantAuthz: 1, wantReads: 1},
		{name: "missing assigned node", decision: authorizer.DecisionAllow, mutateCheckpoint: func(c *nodev1alpha1.PodCheckpoint) { c.Status.NodeName = nil }, wantError: "status.nodeName", wantAuthz: 1, wantReads: 1},
		{name: "empty assigned node", decision: authorizer.DecisionAllow, mutateCheckpoint: func(c *nodev1alpha1.PodCheckpoint) { c.Status.NodeName = ptr.To("") }, wantError: "status.nodeName", wantAuthz: 1, wantReads: 1},
		{name: "missing protection", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) { p.Finalizers = nil }, wantError: "must retain", wantAuthz: 1, wantReads: 1},
		{name: "explicit nodeName", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) { p.Spec.NodeName = "node-1" }, wantError: "must not set spec.nodeName", wantAuthz: 1, wantReads: 1},
		{name: "missing affinity", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) { p.Spec.Affinity = nil }, wantError: "must match", wantInvalid: true, wantAuthz: 1, wantReads: 1},
		{name: "wrong node pin", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) {
			p.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[0].MatchFields[0].Values = []string{"other-node"}
		}, wantError: "must match", wantInvalid: true, wantAuthz: 1, wantReads: 1},
		{name: "changed workload", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) { p.Spec.Containers[0].Image = "different-image" }, wantError: "must match", wantInvalid: true, wantAuthz: 1, wantReads: 1},
		{name: "changed final reference", decision: authorizer.DecisionAllow, mutatePod: func(p *api.Pod) { p.Spec.RestoreFrom.Name = "cp-2" }, wantAuthz: 1, wantReads: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod, checkpoint := preparedRestore(t)
			if tc.mutatePod != nil {
				tc.mutatePod(pod)
			}
			if tc.mutateCheckpoint != nil {
				tc.mutateCheckpoint(checkpoint)
			}
			authzCalls, reads := 0, 0
			ctx := restoreContext()
			if tc.ctx != nil {
				ctx = tc.ctx()
			}
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			policy := &restorePolicy{authorizer: authorizer.AuthorizerFunc(func(actual context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
				authzCalls++
				if actual != ctx || a.GetUser().GetName() != "alice" || a.GetNamespace() != pod.Namespace || a.GetVerb() != "restore" || a.GetAPIGroup() != "node.k8s.io" || a.GetAPIVersion() != "v1alpha1" || a.GetResource() != "podcheckpoints" || a.GetName() != pod.Spec.RestoreFrom.Name || !a.IsResourceRequest() {
					t.Errorf("unexpected restore authorization: %#v", a)
				}
				return tc.decision, "policy", tc.authzErr
			})}
			client := fake.NewClientset()
			client.PrependReactor("get", "podcheckpoints", func(action clienttesting.Action) (bool, runtime.Object, error) {
				reads++
				if tc.cancelDuringRead {
					cancel()
				}
				if authzCalls == 0 {
					t.Error("checkpoint read preceded restore authorization")
				}
				if action.GetNamespace() != pod.Namespace || action.(clienttesting.GetAction).GetName() != pod.Spec.RestoreFrom.Name {
					t.Errorf("unexpected checkpoint lookup: %#v", action)
				}
				return true, checkpoint, tc.readErr
			})
			policy.checkpoints = client.NodeV1alpha1()
			if tc.missingAuthorizer {
				policy.authorizer = nil
			}
			if tc.missingClient {
				policy.checkpoints = nil
			}
			checkpointBefore := checkpoint.DeepCopy()
			podBefore := pod.DeepCopy()
			err := policy.validateCreate(ctx, pod)
			if tc.wantError == "" {
				if err != nil {
					t.Fatalf("unexpected validation error: %v", err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) || (tc.wantInvalid && !apierrors.IsInvalid(err)) || (!tc.wantInvalid && !apierrors.IsForbidden(err)) {
				t.Errorf("expected typed error containing %q, got %v", tc.wantError, err)
			}
			if authzCalls != tc.wantAuthz || reads != tc.wantReads {
				t.Errorf("authorization/read counts = %d/%d, want %d/%d", authzCalls, reads, tc.wantAuthz, tc.wantReads)
			}
			if diff := cmp.Diff(checkpointBefore, checkpoint); diff != "" {
				t.Errorf("checkpoint mutated: %s", diff)
			}
			if diff := cmp.Diff(podBefore, pod); diff != "" {
				t.Errorf("pod mutated: %s", diff)
			}
		})
	}
}

func TestRestorePolicyUpdate(t *testing.T) {
	for _, gate := range []bool{false, true} {
		t.Run(map[bool]string{false: "gate off", true: "gate on"}[gate], func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, gate)
			for _, tc := range []struct {
				name                           string
				protected, remove, restore     bool
				decision                       authorizer.Decision
				authzErr                       error
				missingAuthorizer, missingUser bool
				wantCalls                      int
				wantDenied                     bool
			}{
				{name: "unprotected", remove: true},
				{name: "retained", protected: true},
				{name: "denied", protected: true, remove: true, restore: true, wantCalls: 1, wantDenied: true},
				{name: "without restore spec", protected: true, remove: true, wantCalls: 1, wantDenied: true},
				{name: "allowed", protected: true, remove: true, decision: authorizer.DecisionAllow, wantCalls: 1},
				{name: "authorization error", protected: true, remove: true, authzErr: errors.New("backend unavailable"), wantCalls: 1, wantDenied: true},
				{name: "missing authorizer", protected: true, remove: true, missingAuthorizer: true, wantDenied: true},
				{name: "missing user", protected: true, remove: true, missingUser: true, wantDenied: true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					oldPod := validNewPod()
					oldPod.Namespace = "team-a"
					if tc.protected {
						oldPod.Finalizers = []string{nodeapi.PodRestoreProtectionFinalizer, "example.com/unrelated"}
					}
					if tc.restore {
						oldPod.Spec.RestoreFrom = &api.CheckpointReference{Name: "gone"}
					}
					pod := oldPod.DeepCopy()
					if tc.remove {
						pod.Finalizers = []string{"example.com/unrelated"}
					}
					calls := 0
					policy := &restorePolicy{authorizer: authorizer.AuthorizerFunc(func(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
						calls++
						if a.GetUser().GetName() != "alice" || a.GetVerb() != "update" || a.GetNamespace() != oldPod.Namespace || a.GetName() != oldPod.Name || a.GetResource() != "pods" || a.GetSubresource() != "finalizers" || a.GetAPIGroup() != "" || a.GetAPIVersion() != "v1" {
							t.Errorf("unexpected finalizer authorization: %#v", a)
						}
						return tc.decision, "policy", tc.authzErr
					})}
					if tc.missingAuthorizer {
						policy.authorizer = nil
					}
					ctx := restoreContext()
					if tc.missingUser {
						ctx = genericapirequest.WithNamespace(context.Background(), "team-a")
					}
					err := policy.validateUpdate(ctx, pod, oldPod)
					if tc.wantDenied != apierrors.IsForbidden(err) || (!tc.wantDenied && err != nil) {
						t.Errorf("expected forbidden=%v, got %v", tc.wantDenied, err)
					}
					if calls != tc.wantCalls {
						t.Errorf("authorization count = %d, want %d", calls, tc.wantCalls)
					}
				})
			}
		})
	}
}

func TestRestoreStorageInitialization(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	if _, err := NewStorage(nil, nil, nil, nil, nil, nil); err == nil || !strings.Contains(err.Error(), "authorizer") {
		t.Fatalf("expected missing authorizer error, got %v", err)
	}
	allow := authorizer.AuthorizerFunc(func(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
		return authorizer.DecisionAllow, "", nil
	})
	if _, err := NewStorage(nil, nil, nil, nil, allow, nil); err == nil || !strings.Contains(err.Error(), "PodCheckpoint client") {
		t.Fatalf("expected missing checkpoint client error, got %v", err)
	}
}

func TestRestoreValidationCallbacks(t *testing.T) {
	storage, _, statusStorage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)

	for _, tc := range []struct {
		name                    string
		apply, ordinary, denied bool
		callbackError, dryRun   bool
	}{
		{name: "create"},
		{name: "apply-create", apply: true},
		{name: "create-denied", denied: true},
		{name: "apply-create-denied", apply: true, denied: true},
		{name: "create-callback-error", callbackError: true},
		{name: "apply-create-callback-error", apply: true, callbackError: true},
		{name: "ordinary-create", ordinary: true},
		{name: "ordinary-apply-create", ordinary: true, apply: true},
		{name: "create-dry-run", dryRun: true},
		{name: "apply-create-dry-run", apply: true, dryRun: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod, checkpoint := preparedRestore(t)
			pod.Name = tc.name
			if tc.ordinary {
				pod.Spec.RestoreFrom = nil
			}
			ctx := genericapirequest.WithUser(genericregistrytest.NewNamespaceScopeContext(storage.Store, pod.Namespace), &user.DefaultInfo{Name: "alice"})
			authzCalls, reads, callbacks := 0, 0, 0
			storage.restorePolicy.authorizer = authorizer.AuthorizerFunc(func(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
				authzCalls++
				if tc.denied {
					return authorizer.DecisionDeny, "policy", nil
				}
				return authorizer.DecisionAllow, "", nil
			})
			client := fake.NewClientset(checkpoint)
			client.PrependReactor("get", "podcheckpoints", func(clienttesting.Action) (bool, runtime.Object, error) {
				reads++
				return false, nil, nil
			})
			storage.restorePolicy.checkpoints = client.NodeV1alpha1()
			callbackErr := errors.New("incoming create validation rejected the pod")
			validate := func(actual context.Context, obj runtime.Object) error {
				callbacks++
				if actual != ctx || obj.(*api.Pod).Status.Phase != api.PodPending {
					t.Error("incoming create validation did not receive the normalized object and request context")
				}
				if tc.callbackError {
					return callbackErr
				}
				return nil
			}
			var dryRun []string
			if tc.dryRun {
				dryRun = []string{metav1.DryRunAll}
			}
			var err error
			if tc.apply {
				var created bool
				_, created, err = storage.Update(ctx, pod.Name, rest.DefaultUpdatedObjectInfo(pod), validate, rest.ValidateAllObjectUpdateFunc, true, &metav1.UpdateOptions{DryRun: dryRun})
				if err == nil && !created {
					t.Error("apply-on-create did not create the pod")
				}
			} else {
				_, err = storage.Create(ctx, pod, validate, &metav1.CreateOptions{DryRun: dryRun})
			}
			switch {
			case tc.denied:
				if !apierrors.IsForbidden(err) || callbacks != 0 || reads != 0 {
					t.Errorf("denial did not stop callbacks and checkpoint reads: err=%v callbacks=%d reads=%d", err, callbacks, reads)
				}
			case tc.callbackError:
				if !errors.Is(err, callbackErr) || callbacks != 1 {
					t.Errorf("incoming callback error was lost: %v (calls=%d)", err, callbacks)
				}
			default:
				if err != nil || callbacks != 1 {
					t.Errorf("expected allowed callback, got %v (calls=%d)", err, callbacks)
				}
			}
			if tc.ordinary && (authzCalls != 0 || reads != 0) {
				t.Errorf("ordinary pod performed restore work: authorizations=%d reads=%d", authzCalls, reads)
			}
			_, getErr := storage.Get(ctx, pod.Name, &metav1.GetOptions{})
			if tc.denied || tc.callbackError || tc.dryRun {
				if !apierrors.IsNotFound(getErr) {
					t.Errorf("rejected or dry-run pod persisted: %v", getErr)
				}
			} else if getErr != nil {
				t.Errorf("allowed pod was not persisted: %v", getErr)
			}
		})
	}

	for _, tc := range []struct {
		name                          string
		status, denied, callbackError bool
		retain, deleting              bool
	}{
		{name: "update"},
		{name: "status", status: true},
		{name: "update-denied", denied: true},
		{name: "status-denied", status: true, denied: true},
		{name: "update-callback-error", callbackError: true},
		{name: "status-callback-error", status: true, callbackError: true},
		{name: "metadata-after-checkpoint-deletion", retain: true},
		{name: "status-after-checkpoint-deletion", status: true, retain: true},
		{name: "deleting-update-denied", deleting: true, denied: true},
		{name: "deleting-update", deleting: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod, _ := preparedRestore(t)
			pod.Name = tc.name
			ctx := genericapirequest.WithUser(genericregistrytest.NewNamespaceScopeContext(storage.Store, pod.Namespace), &user.DefaultInfo{Name: "alice"})
			if tc.status {
				ctx = genericapirequest.WithUser(genericregistrytest.NewNamespaceScopeContext(storage.Store, pod.Namespace, "status"), &user.DefaultInfo{Name: "alice"})
			}
			// Seed the old object directly to represent a restore whose checkpoint is gone.
			key, err := storage.KeyFunc(ctx, pod.Name)
			if err != nil {
				t.Fatal(err)
			}
			if err := storage.Storage.Create(ctx, key, pod, nil, 0, false); err != nil {
				t.Fatal(err)
			}
			if tc.deleting {
				if _, _, err := storage.Delete(ctx, pod.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)}); err != nil {
					t.Fatal(err)
				}
			}
			obj, err := storage.Get(ctx, pod.Name, &metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			pod = obj.(*api.Pod)
			if !tc.retain {
				pod.Finalizers = nil
			}
			pod.Labels = map[string]string{"changed": "true"}
			calls, callbacks := 0, 0
			storage.restorePolicy.checkpoints = nil
			storage.restorePolicy.authorizer = authorizer.AuthorizerFunc(func(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
				calls++
				if tc.denied {
					return authorizer.DecisionDeny, "policy", nil
				}
				return authorizer.DecisionAllow, "", nil
			})
			callbackErr := errors.New("incoming update validation rejected the pod")
			validate := func(actual context.Context, obj, old runtime.Object) error {
				callbacks++
				if actual != ctx || len(old.(*api.Pod).Finalizers) != 1 {
					t.Error("incoming update callback lost context or old object")
				}
				if tc.callbackError {
					return callbackErr
				}
				return nil
			}
			if tc.status {
				_, _, err = statusStorage.Update(ctx, pod.Name, rest.DefaultUpdatedObjectInfo(pod), rest.ValidateAllObjectFunc, validate, false, &metav1.UpdateOptions{})
			} else {
				_, _, err = storage.Update(ctx, pod.Name, rest.DefaultUpdatedObjectInfo(pod), rest.ValidateAllObjectFunc, validate, false, &metav1.UpdateOptions{})
			}
			switch {
			case tc.denied:
				if !apierrors.IsForbidden(err) || callbacks != 0 {
					t.Errorf("expected finalizer removal denial before incoming callback, got %v (calls=%d)", err, callbacks)
				}
			case tc.callbackError:
				if !errors.Is(err, callbackErr) || callbacks != 1 {
					t.Errorf("incoming update callback error was lost: %v (calls=%d)", err, callbacks)
				}
			default:
				if err != nil || callbacks == 0 {
					t.Errorf("expected successful update callback, got %v (calls=%d)", err, callbacks)
				}
			}
			if tc.retain && calls != 0 {
				t.Errorf("ordinary update performed restore authorization %d times", calls)
			}
			stored, getErr := storage.Get(ctx, pod.Name, &metav1.GetOptions{})
			switch {
			case tc.deleting && !tc.denied:
				if !apierrors.IsNotFound(getErr) {
					t.Errorf("authorized finalizer removal did not complete deletion: %v", getErr)
				}
			case getErr != nil:
				t.Errorf("pod disappeared after update: %v", getErr)
			default:
				if wantProtected := tc.retain || tc.denied || tc.callbackError; (len(stored.(*api.Pod).Finalizers) != 0) != wantProtected {
					t.Errorf("stored finalizer protection = %v, want %v", stored.(*api.Pod).Finalizers, wantProtected)
				}
			}
		})
	}

	t.Run("status-never-creates", func(t *testing.T) {
		pod := validNewPod()
		pod.Name = "status-missing"
		pod.Namespace = "team-a"
		ctx := genericregistrytest.NewNamespaceScopeContext(storage.Store, pod.Namespace, "status")
		if _, _, err := statusStorage.Update(ctx, pod.Name, rest.DefaultUpdatedObjectInfo(pod), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, true, &metav1.UpdateOptions{}); !apierrors.IsNotFound(err) {
			t.Errorf("status allowed create-on-update: %v", err)
		}
	})
}

func TestRestoreSpecComparison(t *testing.T) {
	source := validNewPod()
	source.Spec.NodeName = "node-1"
	source.Spec.EphemeralContainers = []api.EphemeralContainer{{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "debug", Image: "debug"}}}
	source.Spec.Containers[0].Image = "captured-private-image"
	source.Spec.Containers[0].Resources.Requests = api.ResourceList{api.ResourceCPU: apiresource.MustParse("2")}
	source.Spec.Containers[0].Args = []string{"captured-private-argument"}
	source.Spec.Containers[0].Env = []api.EnvVar{
		{Name: "PRIVATE_VALUE", Value: "captured-private-environment"},
		{Name: "POD_NAME", ValueFrom: &api.EnvVarSource{FieldRef: &api.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.name"}}},
	}
	source.Spec.InitContainers = []api.Container{
		{Name: "init", Image: "init@sha256:captured-private-digest"},
		{Name: "sidecar", Image: "sidecar@sha256:captured-private-digest", RestartPolicy: ptr.To(api.ContainerRestartPolicyAlways)},
	}
	source.Spec.Tolerations = []api.Toleration{{Key: "private", Value: "captured-private-toleration"}}
	for _, tc := range []struct {
		name       string
		mutate     func(*api.Pod)
		wantFields []string
	}{
		{name: "captured resized resources", mutate: func(*api.Pod) {}},
		{name: "equivalent resource quantity", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Resources.Requests[api.ResourceCPU] = apiresource.MustParse("2000m")
		}},
		{name: "scheduling gates", mutate: func(p *api.Pod) { p.Spec.SchedulingGates = []api.PodSchedulingGate{{Name: "example.com/ready"}} }},
		{name: "new metadata with unchanged downward API", mutate: func(p *api.Pod) {
			p.Name = "different-name"
			p.Labels = map[string]string{"different": "metadata"}
		}},
		{name: "image mismatch", mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "changed-private-image" }, wantFields: []string{"spec.containers[0].image"}},
		{name: "init digest mismatch", mutate: func(p *api.Pod) { p.Spec.InitContainers[0].Image = "init@sha256:changed-private-digest" }, wantFields: []string{"spec.initContainers[0].image"}},
		{name: "restartable init tag mismatch", mutate: func(p *api.Pod) { p.Spec.InitContainers[1].Image = "sidecar:changed-private-tag" }, wantFields: []string{"spec.initContainers[1].image"}},
		{name: "resource mismatch", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Resources.Requests[api.ResourceCPU] = apiresource.MustParse("1")
		}, wantFields: []string{"spec.containers[0].resources.requests[cpu]"}},
		{name: "argument mismatch", mutate: func(p *api.Pod) { p.Spec.Containers[0].Args[0] = "changed-private-argument" }, wantFields: []string{"spec.containers[0].args[0]"}},
		{name: "environment mismatch", mutate: func(p *api.Pod) { p.Spec.Containers[0].Env[0].Value = "changed-private-environment" }, wantFields: []string{"spec.containers[0].env[0].value"}},
		{name: "downward API declaration mismatch", mutate: func(p *api.Pod) { p.Spec.Containers[0].Env[1].ValueFrom.FieldRef.FieldPath = "metadata.namespace" }, wantFields: []string{"spec.containers[0].env[1].valueFrom.fieldRef.fieldPath"}},
		{name: "toleration mismatch", mutate: func(p *api.Pod) { p.Spec.Tolerations[0].Value = "changed-private-toleration" }, wantFields: []string{"spec.tolerations[0].value"}},
		{name: "multiple mismatches", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Image = "changed-private-image"
			p.Spec.Containers[0].Env[0].Value = "changed-private-environment"
		}, wantFields: []string{"spec.containers[0].env[0].value", "spec.containers[0].image"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod, checkpoint := preparedRestoreFromPod(t, source)
			tc.mutate(pod)
			err := validatePodSpecMatchesCheckpoint(pod, checkpoint.Name, checkpoint)
			if len(tc.wantFields) == 0 {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if !apierrors.IsInvalid(err) {
				t.Fatalf("expected invalid Pod error, got %v", err)
			}
			status := err.(apierrors.APIStatus).Status()
			var fields []string
			for _, cause := range status.Details.Causes {
				fields = append(fields, cause.Field)
				if cause.Type != metav1.CauseTypeForbidden {
					t.Errorf("unexpected cause type: %v", cause.Type)
				}
			}
			if diff := cmp.Diff(tc.wantFields, fields); diff != "" {
				t.Errorf("changed fields (-want +got): %s", diff)
			}
			if strings.Contains(err.Error(), "captured-private-") || strings.Contains(err.Error(), "changed-private-") {
				t.Errorf("comparison error disclosed workload values: %v", err)
			}
		})
	}
}

func TestRestoreSchedulingComparison(t *testing.T) {
	for _, terms := range [][]api.NodeSelectorTerm{nil, {{}}} {
		source := validNewPod()
		source.Spec.Affinity = &api.Affinity{NodeAffinity: &api.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{NodeSelectorTerms: terms}}}
		pod, checkpoint := preparedRestoreFromPod(t, source)
		if err := validatePodSpecMatchesCheckpoint(pod, checkpoint.Name, checkpoint); err != nil {
			t.Fatal(err)
		}
		pod.Spec.Affinity = nil
		if !apierrors.IsInvalid(validatePodSpecMatchesCheckpoint(pod, checkpoint.Name, checkpoint)) {
			t.Error("dropping unsatisfiable captured affinity was allowed")
		}
	}
	for _, identity := range []api.NodeSelectorRequirement{
		{Key: "metadata.name", Operator: api.NodeSelectorOpIn, Values: []string{"node-1"}},
		{Key: "metadata.name", Operator: api.NodeSelectorOpNotIn, Values: []string{"other-node"}},
	} {
		for _, mutation := range []string{"unchanged", "drop zone", "change identity", "remove pin"} {
			t.Run(string(identity.Operator)+"/"+mutation, func(t *testing.T) {
				source := validNewPod()
				source.Spec.Affinity = &api.Affinity{NodeAffinity: &api.NodeAffinity{RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{NodeSelectorTerms: []api.NodeSelectorTerm{
					{MatchFields: []api.NodeSelectorRequirement{identity}},
					{MatchExpressions: []api.NodeSelectorRequirement{{Key: "zone", Operator: api.NodeSelectorOpIn, Values: []string{"prod"}}}},
				}}}}
				pod, checkpoint := preparedRestoreFromPod(t, source)
				originalCheckpoint := checkpoint.DeepCopy()
				terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				switch mutation {
				case "drop zone":
					terms.NodeSelectorTerms = terms.NodeSelectorTerms[:1]
				case "change identity":
					terms.NodeSelectorTerms[0].MatchFields[0].Values = []string{"third-node"}
				case "remove pin":
					terms.NodeSelectorTerms[1].MatchFields = nil
				}
				err := validatePodSpecMatchesCheckpoint(pod, checkpoint.Name, checkpoint)
				if mutation == "unchanged" && err != nil {
					t.Fatal(err)
				} else if mutation != "unchanged" && !apierrors.IsInvalid(err) {
					t.Errorf("expected Invalid after %s, got %v", mutation, err)
				}
				if diff := cmp.Diff(originalCheckpoint, checkpoint); diff != "" {
					t.Errorf("comparison mutated checkpoint: %s", diff)
				}
			})
		}
	}
}
