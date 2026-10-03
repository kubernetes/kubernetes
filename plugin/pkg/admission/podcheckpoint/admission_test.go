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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
		{name: "deleting", mutate: func(p *v1.Pod, _ *node.PodCheckpoint) { p.DeletionTimestamp = ptr.To(metav1.Now()) }, wantErr: "being deleted"},
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
		policy       *nodev1.RuntimeClassPodCheckpoint
		disabled     bool
		wantErr      string
	}{
		{name: "no options needs no RuntimeClass"},
		{name: "empty options needs no RuntimeClass", options: map[string]string{}},
		{name: "options without RuntimeClass", options: map[string]string{"tcp": "close"}, wantErr: "spec.runtimeClassName"},
		{name: "missing RuntimeClass", options: map[string]string{"tcp": "close"}, useClass: true, missingClass: true, wantErr: "cannot read RuntimeClass"},
		{name: "missing policy", options: map[string]string{"tcp": "close"}, useClass: true, wantErr: `runtime option "tcp" is not allowed`},
		{name: "empty checkpoint allowlist", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{}, wantErr: `runtime option "tcp" is not allowed`},
		{name: "allowed checkpoint key", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}},
		{name: "disallowed checkpoint key", options: map[string]string{"device-map": "sensitive-value"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedCheckpointOptions: []string{"tcp"}}, wantErr: `runtime option "device-map" is not allowed`},
		{name: "restore list cannot authorize checkpoint", options: map[string]string{"tcp": "close"}, useClass: true, policy: &nodev1.RuntimeClassPodCheckpoint{AllowedRestoreOptions: []string{"tcp"}}, wantErr: `runtime option "tcp" is not allowed`},
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
					objects = append(objects, &nodev1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: className}, Handler: "runtime", PodCheckpoint: tc.policy})
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
