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

package scheduler

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestPatchPodStatusUIDPrecondition verifies against a real API server that a status
// patch built for one Pod cannot be applied to a different Pod that took over its name.
//
// A fake clientset cannot cover this: its tracker applies a strategic merge patch
// without any immutability validation, so it happily rewrites metadata.uid.
func TestPatchPodStatusUIDPrecondition(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "patch-pod-status", nil)

	ctx := testCtx.Ctx
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	newStatus := v1.PodStatus{
		Conditions: []v1.PodCondition{{
			Type:    v1.PodScheduled,
			Status:  v1.ConditionFalse,
			Reason:  v1.PodReasonUnschedulable,
			Message: "patched by the test",
		}},
	}

	createPod := func() *v1.Pod {
		t.Helper()
		pod, err := cs.CoreV1().Pods(ns).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: ns},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{Name: "container", Image: "image"}},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod: %v", err)
		}
		return pod
	}

	original := createPod()
	originalStatus := original.Status.DeepCopy()

	// A patch aimed at the Pod that is actually there still works.
	if err := schedutil.PatchPodStatus(ctx, cs, original.Name, ns, original.UID, originalStatus, &newStatus); err != nil {
		t.Fatalf("Failed to patch the status of the live pod: %v", err)
	}

	if err := cs.CoreV1().Pods(ns).Delete(ctx, original.Name, *metav1.NewDeleteOptions(0)); err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := cs.CoreV1().Pods(ns).Get(ctx, original.Name, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("Pod was not deleted: %v", err)
	}

	// A deleted Pod keeps reporting not-found: the precondition only ever fires against
	// an object that is actually there.
	err := schedutil.PatchPodStatus(ctx, cs, original.Name, ns, original.UID, originalStatus, &newStatus)
	if !apierrors.IsNotFound(err) {
		t.Errorf("Expected a not-found error when patching a deleted pod, got %v", err)
	}

	// Recreate under the same name. This is what the informer can report to the
	// scheduler as a plain update of the original Pod.
	recreated := createPod()
	if recreated.UID == original.UID {
		t.Fatalf("Recreated pod unexpectedly reuses uid %v", recreated.UID)
	}

	err = schedutil.PatchPodStatus(ctx, cs, original.Name, ns, original.UID, originalStatus, &newStatus)
	if !apierrors.IsInvalid(err) {
		t.Errorf("Expected an invalid error when patching a recreated pod with the previous uid, got %v", err)
	}

	live, err := cs.CoreV1().Pods(ns).Get(ctx, original.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get the recreated pod: %v", err)
	}
	if live.UID != recreated.UID {
		t.Errorf("Recreated pod uid changed: got %v, want %v", live.UID, recreated.UID)
	}
	for _, condition := range live.Status.Conditions {
		if condition.Message == "patched by the test" {
			t.Errorf("Stale patch was applied to the recreated pod: %v", condition)
		}
	}
}
