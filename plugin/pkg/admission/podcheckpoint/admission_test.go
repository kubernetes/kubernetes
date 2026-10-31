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

package podcheckpoint

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	node "k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestValidateSourcePod(t *testing.T) {
	for _, tc := range []struct {
		name        string
		mutate      func(*v1.Pod, *node.PodCheckpoint)
		missing     bool
		disabled    bool
		operation   admission.Operation
		subresource string
		wantErr     string
	}{
		{name: "running"},
		{name: "matching-uid", mutate: func(p *v1.Pod, c *node.PodCheckpoint) { c.Spec.SourcePod.UID = &p.UID }},
		{name: "wrong-uid", mutate: func(_ *v1.Pod, c *node.PodCheckpoint) { c.Spec.SourcePod.UID = ptr.To(types.UID("other")) }, wantErr: "UID"},
		{name: "unassigned", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Spec.NodeName = "" }, wantErr: "assigned to a node and Running"},
		{name: "pending", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Status.Phase = v1.PodPending }, wantErr: "Running"},
		{name: "succeeded", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Status.Phase = v1.PodSucceeded }, wantErr: "Running"},
		{name: "failed", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.Status.Phase = v1.PodFailed }, wantErr: "Running"},
		{name: "deleting", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.DeletionTimestamp = new(metav1.Now()) }, wantErr: "being deleted"},
		{name: "missing", missing: true, wantErr: "cannot read source"},
		{name: "missing-reference", mutate: func(_ *v1.Pod, c *node.PodCheckpoint) { c.Spec.SourcePod = nil }, wantErr: "sourcePod.name"},
		{name: "disabled", missing: true, disabled: true},
		{name: "update-does-not-revalidate-source", missing: true, operation: admission.Update},
		{name: "status-does-not-revalidate-source", missing: true, operation: admission.Update, subresource: "status"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, !tc.disabled)
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns", UID: "source-uid"}, Spec: v1.PodSpec{NodeName: "node"}, Status: v1.PodStatus{Phase: v1.PodRunning}}
			checkpoint := &node.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"}, Spec: node.PodCheckpointSpec{SourcePod: &node.PodReference{Name: "source"}}}
			if tc.mutate != nil {
				tc.mutate(pod, checkpoint)
			}
			var objects []runtime.Object
			if !tc.missing {
				objects = append(objects, pod)
			}
			client := fake.NewClientset(objects...)
			p := &Plugin{Handler: admission.NewHandler(admission.Create)}
			p.SetExternalKubeClientSet(client)
			p.InspectFeatureGates(utilfeature.DefaultFeatureGate)
			if err := p.ValidateInitialization(); err != nil {
				t.Fatal(err)
			}
			op := tc.operation
			if op == "" {
				op = admission.Create
			}
			a := admission.NewAttributesRecord(checkpoint, nil, node.SchemeGroupVersion.WithKind("PodCheckpoint"), "ns", "checkpoint", node.SchemeGroupVersion.WithResource("podcheckpoints"), tc.subresource, op, nil, false, nil)
			err := p.Validate(context.Background(), a, nil)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("want error containing %q, got %v", tc.wantErr, err)
			}
			if tc.disabled || op != admission.Create || tc.subresource != "" {
				if len(client.Actions()) != 0 {
					t.Fatalf("unexpected API requests: %v", client.Actions())
				}
			}
		})
	}
}

func TestValidateCheckpointRuntimeOptions(t *testing.T) {
	for _, tc := range []struct {
		name         string
		options      map[string]string
		useClass     bool
		missingClass bool
		policy       *nodev1.RuntimeClassPodCheckpointOptions
		disabled     bool
		wantErr      string
	}{
		{name: "no options needs no RuntimeClass"},
		{name: "empty options needs no RuntimeClass", options: map[string]string{}},
		{name: "options without RuntimeClass", options: map[string]string{"tcp": "close"}, wantErr: "spec.runtimeClassName"},
		{name: "missing RuntimeClass", options: map[string]string{"tcp": "close"}, useClass: true, missingClass: true, wantErr: "cannot read RuntimeClass"},
		{name: "missing policy", options: map[string]string{"tcp": "close"}, useClass: true, wantErr: `runtime option "tcp" is not allowed`},
		{name: "empty checkpoint allowlist", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpointOptions{}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "allowed checkpoint key", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpointOptions{AllowedCheckpointOptions: []string{"tcp"}}},
		{name: "disallowed checkpoint key", options: map[string]string{"device-map": "sensitive-value"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpointOptions{AllowedCheckpointOptions: []string{"tcp"}}, wantErr: `runtime option "device-map" is not allowed`},
		{name: "restore list cannot authorize checkpoint", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpointOptions{AllowedRestoreOptions: []string{"tcp"}}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "feature disabled ignores options", options: map[string]string{"tcp": "close"}, disabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, !tc.disabled)
			pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns"}, Spec: v1.PodSpec{NodeName: "node"}, Status: v1.PodStatus{Phase: v1.PodRunning}}
			objects := []runtime.Object{pod}
			if tc.useClass {
				className := "checkpoint-runtime"
				pod.Spec.RuntimeClassName = &className
				if !tc.missingClass {
					objects = append(objects, &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: className}, Handler: "runtime", PodCheckpointOptions: tc.policy})
				}
			}
			checkpoint := &node.PodCheckpoint{ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"}, Spec: node.PodCheckpointSpec{SourcePod: &node.PodReference{Name: "source"}, CheckpointOptions: tc.options}}
			original := checkpoint.DeepCopy()
			client := fake.NewClientset(objects...)
			p := &Plugin{Handler: admission.NewHandler(admission.Create)}
			p.SetExternalKubeClientSet(client)
			p.InspectFeatureGates(utilfeature.DefaultFeatureGate)
			attrs := admission.NewAttributesRecord(checkpoint, nil, node.SchemeGroupVersion.WithKind("PodCheckpoint"), "ns", "checkpoint", node.SchemeGroupVersion.WithResource("podcheckpoints"), "", admission.Create, nil, false, nil)
			err := p.Validate(context.Background(), attrs, nil)
			if tc.wantErr == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, tc.wantErr)
				require.NotContains(t, err.Error(), "sensitive-value")
			}
			require.Equal(t, original, checkpoint, "validation must not transform user options")
			if tc.disabled {
				require.Empty(t, client.Actions())
			}
			if len(tc.options) == 0 {
				for _, action := range client.Actions() {
					require.NotEqual(t, "runtimeclasses", action.GetResource().Resource)
				}
			}
		})
	}
}

func TestCheckpointingUpdatesPreserveWorkload(t *testing.T) {
	for _, tc := range []struct {
		name             string
		subresource      string
		mutate           func(*api.Pod)
		checkpointStatus api.ConditionStatus
		gateDisabled     bool
		allowed          bool
		wantFields       []string
	}{
		{name: "image during checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "changed-private-image" }, wantFields: []string{"spec.containers[0].image"}},
		{name: "resize during checkpoint", checkpointStatus: api.ConditionTrue, subresource: "resize", mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Resources.Requests = api.ResourceList{api.ResourceCPU: resource.MustParse("2")}
		}, wantFields: []string{"spec.containers[0].resources.requests[cpu]"}},
		{name: "ephemeral container during checkpoint", checkpointStatus: api.ConditionTrue, subresource: "ephemeralcontainers", mutate: func(p *api.Pod) {
			p.Spec.EphemeralContainers = []api.EphemeralContainer{{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "debug"}}}
		}, wantFields: []string{"spec.ephemeralContainers[0]"}},
		{name: "arguments during checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) { p.Spec.Containers[0].Args[0] = "changed-private-argument" }, wantFields: []string{"spec.containers[0].args[0]"}},
		{name: "environment during checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) { p.Spec.Containers[0].Env[0].Value = "changed-private-environment" }, wantFields: []string{"spec.containers[0].env[0].value"}},
		{name: "multiple changes during checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Image = "changed-private-image"
			p.Spec.Containers[0].Env[0].Value = "changed-private-environment"
		}, wantFields: []string{"spec.containers[0].image", "spec.containers[0].env[0].value"}},
		{name: "metadata during checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) { p.Labels = map[string]string{"app": "test"} }, allowed: true},
		{name: "scheduling gate removal during checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) { p.Spec.SchedulingGates = nil }, allowed: true},
		{name: "status update during checkpoint", checkpointStatus: api.ConditionTrue, subresource: "status", mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
		{name: "binding during checkpoint", checkpointStatus: api.ConditionTrue, subresource: "binding", mutate: func(p *api.Pod) { p.Spec.NodeName = "node-1" }, allowed: true},
		{name: "new status cannot bypass active checkpoint", checkpointStatus: api.ConditionTrue, mutate: func(p *api.Pod) {
			p.Spec.Containers[0].Image = "changed-private-image"
			p.Status.Conditions = []api.PodCondition{{Type: api.PodCheckpointing, Status: api.ConditionFalse}}
		}, wantFields: []string{"spec.containers[0].image"}},
		{name: "image after checkpoint", checkpointStatus: api.ConditionFalse, mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
		{name: "image with unknown checkpoint state", checkpointStatus: api.ConditionUnknown, mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
		{name: "image without checkpoint state", mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
		{name: "feature disabled", checkpointStatus: api.ConditionTrue, gateDisabled: true, mutate: func(p *api.Pod) { p.Spec.Containers[0].Image = "other" }, allowed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, !tc.gateDisabled)
			old := &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns"},
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name: "app", Image: "captured-private-image",
						Args: []string{"captured-private-argument"},
						Env:  []api.EnvVar{{Name: "PRIVATE_VALUE", Value: "captured-private-environment"}},
					}},
					SchedulingGates: []api.PodSchedulingGate{{Name: "example.com/ready"}},
				},
			}
			if tc.checkpointStatus != "" {
				old.Status.Conditions = []api.PodCondition{{
					Type: api.PodCheckpointing, Status: tc.checkpointStatus, Message: "Checkpointing for PodCheckpoint \"source-checkpoint\"",
				}}
			}
			pod := old.DeepCopy()
			tc.mutate(pod)
			p := &Plugin{Handler: admission.NewHandler(admission.Create, admission.Update)}
			p.SetExternalKubeClientSet(fake.NewClientset())
			p.InspectFeatureGates(utilfeature.DefaultFeatureGate)
			require.NoError(t, p.ValidateInitialization())
			require.True(t, p.Handles(admission.Update))
			a := admission.NewAttributesRecord(pod, old, api.Kind("Pod").WithVersion("version"), "ns", "source", api.Resource("pods").WithVersion("version"), tc.subresource, admission.Update, nil, false, nil)
			err := p.Validate(context.Background(), a, admission.NewObjectInterfacesFromScheme(legacyscheme.Scheme))
			if tc.allowed {
				require.NoError(t, err)
				return
			}
			require.True(t, apierrors.IsForbidden(err), "want forbidden, got %v", err)
			require.ErrorContains(t, err, "while Checkpointing=True")
			require.ErrorContains(t, err, "source-checkpoint")
			for _, path := range tc.wantFields {
				require.ErrorContains(t, err, path)
			}
			require.NotContains(t, err.Error(), "captured-private-")
			require.NotContains(t, err.Error(), "changed-private-")
		})
	}
}
