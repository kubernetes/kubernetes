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

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	nodeapply "k8s.io/client-go/applyconfigurations/node/v1alpha1"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestPodCheckpointStatusValidation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-status", t)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	source := createCheckpointSource(tCtx, t, client, ns.Name, "source-pod")
	checkpoint, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "checkpoint", "source-pod"), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create checkpoint: %v", err)
	}
	for name, status := range map[string]nodev1alpha1.PodCheckpointStatus{
		"unknown backend":      {CheckpointLocation: &nodev1alpha1.CheckpointSource{Type: "Unknown"}},
		"missing member":       {CheckpointLocation: &nodev1alpha1.CheckpointSource{Type: nodev1alpha1.CheckpointSourceTypeNodeLocal}},
		"empty path":           {CheckpointLocation: &nodev1alpha1.CheckpointSource{Type: nodev1alpha1.CheckpointSourceTypeNodeLocal, NodeLocal: &nodev1alpha1.NodeLocalCheckpointSource{}}},
		"empty container name": {CheckpointedContainers: []nodev1alpha1.PodCheckpointContainerStatus{{}}},
		"duplicate containers": {CheckpointedContainers: []nodev1alpha1.PodCheckpointContainerStatus{{Name: "app"}, {Name: "app"}}},
	} {
		t.Run(name, func(t *testing.T) {
			update := checkpoint.DeepCopy()
			update.Status = status
			if _, err := checkpoints.UpdateStatus(tCtx, update, metav1.UpdateOptions{}); !apierrors.IsInvalid(err) {
				t.Fatalf("expected invalid checkpoint status, got %v", err)
			}
		})
	}
	checkpoint.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(source)
	checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
	checkpoint.Status.NodeName = new("node-1")
	if _, err := checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("record checkpoint node: %v", err)
	}
}

func TestPodCheckpointTemplateSetOnce(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-template", t)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	createCheckpointSource(tCtx, t, client, ns.Name, "source-pod")
	checkpoint, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "checkpoint", "source-pod"), metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	template := &v1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": "source"}},
		Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "image:v1"}}},
	}
	checkpoint.Status.CheckpointedPodTemplate = template
	checkpoint, err = checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("capture initial template: %v", err)
	}
	if checkpoint.Status.CheckpointedPodTemplate == nil {
		t.Fatal("initial template was not persisted")
	}
	expectedTemplate := checkpoint.Status.CheckpointedPodTemplate.DeepCopy()
	checkpoint.Status.NodeName = new("node-1")
	checkpoint, err = checkpoints.UpdateStatus(tCtx, checkpoint, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("update other status fields with an unchanged template: %v", err)
	}
	for name, mutate := range map[string]func(*nodev1alpha1.PodCheckpoint){
		"changed spec": func(pc *nodev1alpha1.PodCheckpoint) {
			pc.Status.CheckpointedPodTemplate.Spec.Containers[0].Image = "image:v2"
		},
		"changed metadata": func(pc *nodev1alpha1.PodCheckpoint) {
			pc.Status.CheckpointedPodTemplate.Labels["app"] = "other"
		},
		"cleared template": func(pc *nodev1alpha1.PodCheckpoint) {
			pc.Status.CheckpointedPodTemplate = nil
		},
		"empty template": func(pc *nodev1alpha1.PodCheckpoint) {
			pc.Status.CheckpointedPodTemplate = &v1.PodTemplateSpec{}
		},
	} {
		t.Run(name, func(t *testing.T) {
			update := checkpoint.DeepCopy()
			mutate(update)
			if _, err := checkpoints.UpdateStatus(tCtx, update, metav1.UpdateOptions{}); !apierrors.IsInvalid(err) {
				t.Fatalf("expected invalid template update, got %v", err)
			}
		})
	}
	if _, err := checkpoints.Patch(tCtx, checkpoint.Name, types.MergePatchType, []byte(`{"status":{"checkpointedPodTemplate":null}}`), metav1.PatchOptions{}, "status"); !apierrors.IsInvalid(err) {
		t.Fatalf("expected explicit template deletion to be invalid, got %v", err)
	}
	stored, err := checkpoints.Get(tCtx, checkpoint.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !equality.Semantic.DeepEqual(expectedTemplate, stored.Status.CheckpointedPodTemplate) {
		t.Fatal("persisted template changed after rejected updates")
	}
}

func TestPodCheckpointApplyStatus(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	tCtx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "checkpoint-apply-status", t)
	source := createCheckpointSource(tCtx, t, client, ns.Name, "source")
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	apply := nodeapply.PodCheckpoint("checkpoint", ns.Name).
		WithFinalizers("example.com/status").
		WithSpec(nodeapply.PodCheckpointSpec().WithSourcePod(nodeapply.PodReference().WithName(source.Name))).
		WithStatus(nodeapply.PodCheckpointStatus().WithNodeName("node-1"))
	options := metav1.ApplyOptions{FieldManager: "status-test", Force: true}

	t.Run("missing checkpoint", func(t *testing.T) {
		_, err := checkpoints.ApplyStatus(tCtx, apply, options)
		require.True(t, apierrors.IsNotFound(err), "expected NotFound, got %v", err)
		_, err = checkpoints.Get(tCtx, "checkpoint", metav1.GetOptions{})
		require.True(t, apierrors.IsNotFound(err), "checkpoint must not be created, got %v", err)
	})

	t.Run("existing checkpoint", func(t *testing.T) {
		created, err := checkpoints.Create(tCtx, newPodCheckpoint(ns.Name, "existing", source.Name), metav1.CreateOptions{})
		require.NoError(t, err)
		apply.WithName(created.Name)
		apply.Spec.WithTimeoutSeconds(123)
		updated, err := checkpoints.ApplyStatus(tCtx, apply, options)
		require.NoError(t, err)
		require.Equal(t, "node-1", ptr.Deref(updated.Status.NodeName, ""))
		require.Equal(t, created.Spec, updated.Spec)
		require.Equal(t, created.Finalizers, updated.Finalizers)
		stored, err := checkpoints.Get(tCtx, created.Name, metav1.GetOptions{})
		require.NoError(t, err)
		require.Equal(t, updated.Status, stored.Status)
		require.Equal(t, created.Spec, stored.Spec)
		require.Equal(t, created.Finalizers, stored.Finalizers)
	})
}
