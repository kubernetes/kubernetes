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
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestPodCheckpointGeneration(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	clientSet, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(clientSet, "checkpoint-generation", t)
	checkpoints := clientSet.NodeV1alpha1().PodCheckpoints(ns.Name)
	createCheckpointSource(tCtx, t, clientSet, ns.Name, "source-pod")

	for _, tc := range []struct {
		name       string
		generation int64
	}{
		{name: "omitted"},
		{name: "client-supplied", generation: 42},
	} {
		t.Run(tc.name, func(t *testing.T) {
			checkpoint := newPodCheckpoint(ns.Name, tc.name, "source-pod")
			checkpoint.Generation = tc.generation
			created, err := checkpoints.Create(tCtx, checkpoint, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("create PodCheckpoint: %v", err)
			}
			if created.Generation != 1 {
				t.Errorf("created PodCheckpoint generation = %d, want 1", created.Generation)
			}
			stored, err := checkpoints.Get(tCtx, created.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get PodCheckpoint: %v", err)
			}
			if stored.Generation != 1 {
				t.Errorf("stored PodCheckpoint generation = %d, want 1", stored.Generation)
			}
		})
	}
}

// TestPodCheckpointAPINotServedWhenDisabled verifies that with the
// PodLevelCheckpointRestore feature gate disabled, PodCheckpoint is not served
// even when enabled in runtime configuration.
func TestPodCheckpointAPINotServedWhenDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)

	tCtx := ktesting.Init(t)
	clientSet, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "podcheckpoint-disabled", t)
	checkpoints := clientSet.NodeV1alpha1().PodCheckpoints(ns.Name)

	if _, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "cp-1", "src-pod"), metav1.CreateOptions{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected PodCheckpoint API to be unavailable when the feature gate is disabled, got %v", err)
	}
}
