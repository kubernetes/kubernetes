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

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func createTestPod(t *testing.T, testCtx *testutils.TestContext, name string) *v1.Pod {
	t.Helper()
	pod, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testCtx.NS.Name},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "container", Image: "image"}},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}
	return pod
}

func deleteTestPod(t *testing.T, testCtx *testutils.TestContext, name string) {
	t.Helper()
	if err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, name, *metav1.NewDeleteOptions(0)); err != nil {
		t.Fatalf("Failed to delete pod: %v", err)
	}
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 10*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := testCtx.ClientSet.CoreV1().Pods(testCtx.NS.Name).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	}); err != nil {
		t.Fatalf("Pod was not deleted: %v", err)
	}
}

// TestPodUIDPrecondition verifies against a real API server that status patches
// and deletes built for one Pod cannot mutate or delete a different Pod that
// took over its name.
//
// A fake clientset cannot cover this: its tracker applies a strategic merge patch
// without any immutability validation and does not enforce delete preconditions.
func TestPodUIDPrecondition(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "pod-uid-precondition", nil)

	ctx := testCtx.Ctx
	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	t.Run("PatchPodStatus", func(t *testing.T) {
		patchedCondition := v1.PodCondition{
			Type:    v1.PodScheduled,
			Status:  v1.ConditionFalse,
			Reason:  v1.PodReasonUnschedulable,
			Message: "patched by the test",
		}
		newStatus := v1.PodStatus{
			Conditions: []v1.PodCondition{patchedCondition},
		}

		tests := []struct {
			name           string
			podName        string
			beforePatch    func(t *testing.T, original *v1.Pod) *v1.Pod
			validateErr    func(error) bool
			wantConditions []v1.PodCondition
		}{
			{
				name:    "live pod with matching uid is patched",
				podName: "patch-pod-live",
				beforePatch: func(t *testing.T, original *v1.Pod) *v1.Pod {
					return original
				},
				wantConditions: []v1.PodCondition{patchedCondition},
			},
			{
				name:    "deleted pod returns not-found",
				podName: "patch-pod-deleted",
				beforePatch: func(t *testing.T, original *v1.Pod) *v1.Pod {
					deleteTestPod(t, testCtx, original.Name)
					return nil
				},
				validateErr: apierrors.IsNotFound,
			},
			{
				name:    "recreated pod with different uid returns invalid and is not patched",
				podName: "patch-pod-recreated",
				beforePatch: func(t *testing.T, original *v1.Pod) *v1.Pod {
					deleteTestPod(t, testCtx, original.Name)
					recreated := createTestPod(t, testCtx, original.Name)
					if recreated.UID == original.UID {
						t.Fatalf("Recreated pod unexpectedly reuses uid %v", recreated.UID)
					}
					return recreated
				},
				validateErr: schedutil.IsPodUIDPreconditionError,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				original := createTestPod(t, testCtx, tc.podName)
				originalStatus := original.Status.DeepCopy()

				wantLivePod := tc.beforePatch(t, original)

				err := schedutil.PatchPodStatus(ctx, cs, original.Name, ns, original.UID, originalStatus, &newStatus)
				if tc.validateErr != nil {
					if !tc.validateErr(err) {
						t.Fatalf("Unexpected error from PatchPodStatus: %v", err)
					}
				} else if err != nil {
					t.Fatalf("Failed to patch pod status: %v", err)
				}

				if wantLivePod == nil {
					return
				}

				live, err := cs.CoreV1().Pods(ns).Get(ctx, original.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get pod: %v", err)
				}
				if live.UID != wantLivePod.UID {
					t.Errorf("Pod uid changed: got %v, want %v", live.UID, wantLivePod.UID)
				}
				if diff := cmp.Diff(tc.wantConditions, live.Status.Conditions); diff != "" {
					t.Errorf("Unexpected pod conditions (-want,+got):\n%s", diff)
				}
			})
		}
	})

	t.Run("DeletePod", func(t *testing.T) {
		tests := []struct {
			name         string
			podName      string
			beforeDelete func(t *testing.T, original *v1.Pod) *v1.Pod
			validateErr  func(error) bool
		}{
			{
				name:    "live pod with matching uid is deleted",
				podName: "delete-pod-live",
				beforeDelete: func(t *testing.T, original *v1.Pod) *v1.Pod {
					return nil
				},
			},
			{
				name:    "deleted pod returns not-found",
				podName: "delete-pod-deleted",
				beforeDelete: func(t *testing.T, original *v1.Pod) *v1.Pod {
					deleteTestPod(t, testCtx, original.Name)
					return nil
				},
				validateErr: apierrors.IsNotFound,
			},
			{
				name:    "recreated pod with different uid returns conflict and is not deleted",
				podName: "delete-pod-recreated",
				beforeDelete: func(t *testing.T, original *v1.Pod) *v1.Pod {
					deleteTestPod(t, testCtx, original.Name)
					recreated := createTestPod(t, testCtx, original.Name)
					if recreated.UID == original.UID {
						t.Fatalf("Recreated pod unexpectedly reuses uid %v", recreated.UID)
					}
					return recreated
				},
				validateErr: apierrors.IsConflict,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				original := createTestPod(t, testCtx, tc.podName)
				wantLivePod := tc.beforeDelete(t, original)

				err := schedutil.DeletePod(ctx, cs, original)
				if tc.validateErr != nil {
					if !tc.validateErr(err) {
						t.Fatalf("Unexpected error from DeletePod: %v", err)
					}
				} else if err != nil {
					t.Fatalf("Failed to delete pod: %v", err)
				}

				if wantLivePod == nil {
					return
				}

				live, err := cs.CoreV1().Pods(ns).Get(ctx, original.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get pod: %v", err)
				}
				if live.UID != wantLivePod.UID || live.DeletionTimestamp != nil {
					t.Errorf("Expected recreated pod (uid=%v) to remain live, got uid=%v deletionTimestamp=%v", wantLivePod.UID, live.UID, live.DeletionTimestamp)
				}
			})
		}
	})
}
