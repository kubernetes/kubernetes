/*
Copyright 2025 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	checkpointv1alpha1 "k8s.io/api/checkpoint/v1alpha1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var podCheckpointGVR = schema.GroupVersionResource{
	Group:    "checkpoint.k8s.io",
	Version:  "v1alpha1",
	Resource: "podcheckpoints",
}

// runtimeUnsupportedFragments identify a CheckpointFailed message caused by the
// container runtime not (yet) supporting the CheckpointPod CRI RPC or missing
// its CRIU dependency; the test skips instead of failing in those environments.
var runtimeUnsupportedFragments = []string{
	"Unimplemented",
	"checkpoint/restore support not available",
	"CRIU binary not found or too old",
}

// Pod-level checkpoint is driven declaratively (KEP-5823): the test creates a
// PodCheckpoint object and the kubelet that runs the source Pod observes it,
// performs the checkpoint through the CRI, and finalizes the object's status.
// There is no imperative kubelet endpoint to call.
var _ = SIGDescribe("Checkpoint Pod", feature.PodLevelCheckpointRestore, func() {
	f := framework.NewDefaultFramework("checkpoint-pod-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.It("will checkpoint a pod", func(ctx context.Context) {
		ginkgo.By("creating a target pod")
		podClient := e2epod.NewPodClient(f)
		pod := podClient.CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "checkpoint-pod-test",
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "test-container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sleep"},
						Args:    []string{"10000"},
					},
				},
			},
		})

		p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		if !isReady {
			framework.Failf("pod %q should be ready", p.Name)
		}

		ginkgo.By("creating a PodCheckpoint for the pod")
		framework.Logf("About to checkpoint pod %q on %q", p.Name, p.Spec.NodeName)
		checkpointName := "checkpoint-pod-test-cp"
		pc := &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": checkpointv1alpha1.SchemeGroupVersion.String(),
			"kind":       "PodCheckpoint",
			"metadata": map[string]interface{}{
				"name":      checkpointName,
				"namespace": f.Namespace.Name,
			},
			"spec": map[string]interface{}{
				"sourcePod": map[string]interface{}{
					"name": p.Name,
					// Pin the checkpoint to this Pod instance so a same-name
					// replacement is never checkpointed by mistake.
					"uid": string(p.UID),
				},
			},
		}}
		_, err = f.DynamicClient.Resource(podCheckpointGVR).Namespace(f.Namespace.Name).Create(ctx, pc, metav1.CreateOptions{})
		if apierrors.IsNotFound(err) {
			// The checkpoint.k8s.io/v1alpha1 group is only served when the
			// PodLevelCheckpointRestore feature gate is enabled.
			ginkgo.Skip("Feature 'PodLevelCheckpointRestore' is not enabled and not available")
			return
		}
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			_ = f.DynamicClient.Resource(podCheckpointGVR).Namespace(f.Namespace.Name).Delete(ctx, checkpointName, metav1.DeleteOptions{})
		})

		ginkgo.By("waiting for the PodCheckpoint Ready condition to reach a terminal state")
		var got checkpointv1alpha1.PodCheckpoint
		err = wait.PollUntilContextTimeout(ctx, time.Second, 3*time.Minute, true, func(ctx context.Context) (bool, error) {
			obj, err := f.DynamicClient.Resource(podCheckpointGVR).Namespace(f.Namespace.Name).Get(ctx, checkpointName, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			got = checkpointv1alpha1.PodCheckpoint{}
			if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &got); err != nil {
				return false, err
			}
			cond := apimeta.FindStatusCondition(got.Status.Conditions, checkpointv1alpha1.PodCheckpointReady)
			if cond == nil {
				return false, nil
			}
			switch cond.Reason {
			case checkpointv1alpha1.PodCheckpointReasonCompleted,
				checkpointv1alpha1.PodCheckpointReasonFailed,
				checkpointv1alpha1.PodCheckpointReasonSourcePodReplaced:
				return true, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "PodCheckpoint did not reach a terminal Ready condition")

		cond := apimeta.FindStatusCondition(got.Status.Conditions, checkpointv1alpha1.PodCheckpointReady)
		if cond.Reason == checkpointv1alpha1.PodCheckpointReasonFailed {
			// Tolerate container runtimes that have not adopted the
			// CheckpointPod CRI RPC (or lack CRIU) by skipping.
			for _, fragment := range runtimeUnsupportedFragments {
				if strings.Contains(cond.Message, fragment) {
					ginkgo.Skip("Container engine does not support 'CheckpointPod': " + cond.Message)
					return
				}
			}
			framework.Failf("checkpoint failed: %s", cond.Message)
		}
		if cond.Reason != checkpointv1alpha1.PodCheckpointReasonCompleted || cond.Status != metav1.ConditionTrue {
			framework.Failf("unexpected terminal Ready condition: status=%s reason=%s message=%s", cond.Status, cond.Reason, cond.Message)
		}

		ginkgo.By("verifying the PodCheckpoint status")
		if got.Status.NodeName == nil || *got.Status.NodeName != p.Spec.NodeName {
			framework.Failf("status.nodeName = %v, want the source pod's node %q", got.Status.NodeName, p.Spec.NodeName)
		}
		if got.Status.SourcePodUID == nil || *got.Status.SourcePodUID != p.UID {
			framework.Failf("status.sourcePodUID = %v, want the source pod's UID %q", got.Status.SourcePodUID, p.UID)
		}
		if got.Status.CompletionTime == nil {
			framework.Failf("status.completionTime is not set on a completed checkpoint")
		}
		loc := got.Status.CheckpointLocation
		if loc == nil || loc.Type != checkpointv1alpha1.CheckpointSourceTypeNodeLocal || loc.NodeLocal == nil {
			framework.Failf("status.checkpointLocation = %+v, want type NodeLocal with a nodeLocal member", loc)
		}
		// The location is relative to the kubelet's checkpoint root, never an
		// absolute host path, and must not escape the root.
		if loc.NodeLocal.Path == "" || filepath.IsAbs(loc.NodeLocal.Path) || strings.HasPrefix(filepath.Clean(loc.NodeLocal.Path), "..") {
			framework.Failf("status.checkpointLocation.nodeLocal.path = %q, want a non-empty path relative to the kubelet checkpoint root", loc.NodeLocal.Path)
		}
		if got.Status.CheckpointedPodTemplate == nil {
			framework.Failf("status.checkpointedPodTemplate is not populated on a completed checkpoint")
		}
		foundContainer := false
		for _, c := range got.Status.CheckpointedContainers {
			if c.Name == "test-container-1" {
				foundContainer = true
			}
		}
		if !foundContainer {
			framework.Failf("status.checkpointedContainers = %+v, want an entry for %q", got.Status.CheckpointedContainers, "test-container-1")
		}
	})
})
