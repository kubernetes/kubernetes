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

package job

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/util/feature"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	typedv1 "k8s.io/client-go/kubernetes/typed/batch/v1"
	"k8s.io/client-go/metadata"
	"k8s.io/client-go/metadata/metadatainformer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/controller-manager/pkg/informerfactory"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	jobcontroller "k8s.io/kubernetes/pkg/controller/job"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

const waitInterval = time.Second

// TestNonParallelJob tests that a Job that only executes one Pod. The test
// recreates the Job controller at some points to make sure a new controller
// is able to pickup.
func TestNonParallelJob(t *testing.T) {
	for _, wFinalizers := range []bool{false, true} {
		t.Run(fmt.Sprintf("finalizers=%t", wFinalizers), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, wFinalizers)()

			closeFn, restConfig, clientSet, ns := setup(t, "simple")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer func() {
				cancel()
			}()

			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			if got := hasJobTrackingAnnotation(jobObj); got != wFinalizers {
				t.Errorf("apiserver created job with tracking annotation: %t, want %t", got, wFinalizers)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 1,
				Ready:  pointer.Int32(0),
			}, wFinalizers)

			// Restarting controller.
			cancel()
			ctx, cancel = startJobControllerAndWaitForCaches(restConfig)

			// Failed Pod is replaced.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 1); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodFailed, err)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 1,
				Failed: 1,
				Ready:  pointer.Int32(0),
			}, wFinalizers)

			// Restarting controller.
			cancel()
			ctx, cancel = startJobControllerAndWaitForCaches(restConfig)

			// No more Pods are created after the Pod succeeds.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 1); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodSucceeded, err)
			}
			validateJobSucceeded(ctx, t, clientSet, jobObj)
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Failed:    1,
				Succeeded: 1,
				Ready:     pointer.Int32(0),
			}, false)
			validateFinishedPodsNoFinalizer(ctx, t, clientSet, jobObj)
		})
	}
}

func TestParallelJob(t *testing.T) {
	cases := map[string]struct {
		trackWithFinalizers bool
		enableReadyPods     bool
	}{
		"none": {},
		"with finalizers": {
			trackWithFinalizers: true,
		},
		"ready pods": {
			enableReadyPods: true,
		},
		"all": {
			trackWithFinalizers: true,
			enableReadyPods:     true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.trackWithFinalizers)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobReadyPods, tc.enableReadyPods)()

			closeFn, restConfig, clientSet, ns := setup(t, "parallel")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer cancel()

			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: pointer.Int32Ptr(5),
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			want := podsByStatus{Active: 5}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32Ptr(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)

			// Tracks ready pods, if enabled.
			if err := setJobPodsReady(ctx, clientSet, jobObj, 2); err != nil {
				t.Fatalf("Failed Marking Pods as ready: %v", err)
			}
			if tc.enableReadyPods {
				*want.Ready = 2
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)

			// Failed Pods are replaced.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
			}
			want = podsByStatus{
				Active: 5,
				Failed: 2,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)
			// Once one Pod succeeds, no more Pods are created, even if some fail.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 1); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodSucceeded, err)
			}
			want = podsByStatus{
				Failed:    2,
				Succeeded: 1,
				Active:    4,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
			}
			want = podsByStatus{
				Failed:    4,
				Succeeded: 1,
				Active:    2,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)
			// No more Pods are created after remaining Pods succeed.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 2); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodSucceeded, err)
			}
			validateJobSucceeded(ctx, t, clientSet, jobObj)
			want = podsByStatus{
				Failed:    4,
				Succeeded: 3,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, false)
			validateFinishedPodsNoFinalizer(ctx, t, clientSet, jobObj)
		})
	}
}

func TestParallelJobParallelism(t *testing.T) {
	for _, wFinalizers := range []bool{false, true} {
		t.Run(fmt.Sprintf("finalizers=%t", wFinalizers), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, wFinalizers)()

			closeFn, restConfig, clientSet, ns := setup(t, "parallel")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer cancel()

			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
				Spec: batchv1.JobSpec{
					BackoffLimit: pointer.Int32(2),
					Parallelism:  pointer.Int32Ptr(5),
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 5,
				Ready:  pointer.Int32(0),
			}, wFinalizers)

			// Reduce parallelism by a number greater than backoffLimit.
			patch := []byte(`{"spec":{"parallelism":2}}`)
			jobObj, err = clientSet.BatchV1().Jobs(ns.Name).Patch(ctx, jobObj.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			if err != nil {
				t.Fatalf("Updating Job: %v", err)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 2,
				Ready:  pointer.Int32(0),
			}, wFinalizers)

			// Increase parallelism again.
			patch = []byte(`{"spec":{"parallelism":4}}`)
			jobObj, err = clientSet.BatchV1().Jobs(ns.Name).Patch(ctx, jobObj.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			if err != nil {
				t.Fatalf("Updating Job: %v", err)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 4,
				Ready:  pointer.Int32(0),
			}, wFinalizers)

			// Succeed Job
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 4); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
			}
			validateJobSucceeded(ctx, t, clientSet, jobObj)
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Succeeded: 4,
				Ready:     pointer.Int32(0),
			}, false)
			validateFinishedPodsNoFinalizer(ctx, t, clientSet, jobObj)
		})
	}
}

func TestParallelJobWithCompletions(t *testing.T) {
	// Lower limits for a job sync so that we can test partial updates with a low
	// number of pods.
	t.Cleanup(setDuringTest(&jobcontroller.MaxUncountedPods, 10))
	t.Cleanup(setDuringTest(&jobcontroller.MaxPodCreateDeletePerSync, 10))
	cases := map[string]struct {
		trackWithFinalizers bool
		enableReadyPods     bool
	}{
		"none": {},
		"with finalizers": {
			trackWithFinalizers: true,
		},
		"ready pods": {
			enableReadyPods: true,
		},
		"all": {
			trackWithFinalizers: true,
			enableReadyPods:     true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.trackWithFinalizers)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobReadyPods, tc.enableReadyPods)()
			closeFn, restConfig, clientSet, ns := setup(t, "completions")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer cancel()

			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: pointer.Int32Ptr(54),
					Completions: pointer.Int32Ptr(56),
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			if got := hasJobTrackingAnnotation(jobObj); got != tc.trackWithFinalizers {
				t.Errorf("apiserver created job with tracking annotation: %t, want %t", got, tc.trackWithFinalizers)
			}
			want := podsByStatus{Active: 54}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32Ptr(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)

			// Tracks ready pods, if enabled.
			if err := setJobPodsReady(ctx, clientSet, jobObj, 52); err != nil {
				t.Fatalf("Failed Marking Pods as ready: %v", err)
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(52)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)

			// Failed Pods are replaced.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodFailed, err)
			}
			want = podsByStatus{
				Active: 54,
				Failed: 2,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(50)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)
			// Pods are created until the number of succeeded Pods equals completions.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 53); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodSucceeded, err)
			}
			want = podsByStatus{
				Failed:    2,
				Succeeded: 53,
				Active:    3,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, tc.trackWithFinalizers)
			// No more Pods are created after the Job completes.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 3); err != nil {
				t.Fatalf("Failed setting phase %s on Job Pods: %v", v1.PodSucceeded, err)
			}
			validateJobSucceeded(ctx, t, clientSet, jobObj)
			want = podsByStatus{
				Failed:    2,
				Succeeded: 56,
			}
			if tc.enableReadyPods {
				want.Ready = pointer.Int32(0)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, want, false)
			validateFinishedPodsNoFinalizer(ctx, t, clientSet, jobObj)
		})
	}
}

func TestIndexedJob(t *testing.T) {
	for _, wFinalizers := range []bool{false, true} {
		t.Run(fmt.Sprintf("finalizers=%t", wFinalizers), func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, wFinalizers)()

			closeFn, restConfig, clientSet, ns := setup(t, "indexed")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer func() {
				cancel()
			}()

			mode := batchv1.IndexedCompletion
			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism:    pointer.Int32Ptr(3),
					Completions:    pointer.Int32Ptr(4),
					CompletionMode: &mode,
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			if got := hasJobTrackingAnnotation(jobObj); got != wFinalizers {
				t.Errorf("apiserver created job with tracking annotation: %t, want %t", got, wFinalizers)
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 3,
				Ready:  pointer.Int32(0),
			}, wFinalizers)
			validateIndexedJobPods(ctx, t, clientSet, jobObj, sets.NewInt(0, 1, 2), "")

			// One Pod succeeds.
			if err := setJobPhaseForIndex(ctx, clientSet, jobObj, v1.PodSucceeded, 1); err != nil {
				t.Fatal("Failed trying to succeed pod with index 1")
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active:    3,
				Succeeded: 1,
				Ready:     pointer.Int32(0),
			}, wFinalizers)
			validateIndexedJobPods(ctx, t, clientSet, jobObj, sets.NewInt(0, 2, 3), "1")

			// One Pod fails, which should be recreated.
			if err := setJobPhaseForIndex(ctx, clientSet, jobObj, v1.PodFailed, 2); err != nil {
				t.Fatal("Failed trying to succeed pod with index 2")
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active:    3,
				Failed:    1,
				Succeeded: 1,
				Ready:     pointer.Int32(0),
			}, wFinalizers)
			validateIndexedJobPods(ctx, t, clientSet, jobObj, sets.NewInt(0, 2, 3), "1")

			// Remaining Pods succeed.
			if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodSucceeded, 3); err != nil {
				t.Fatal("Failed trying to succeed remaining pods")
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active:    0,
				Failed:    1,
				Succeeded: 4,
				Ready:     pointer.Int32(0),
			}, false)
			validateIndexedJobPods(ctx, t, clientSet, jobObj, nil, "0-3")
			validateJobSucceeded(ctx, t, clientSet, jobObj)
			validateFinishedPodsNoFinalizer(ctx, t, clientSet, jobObj)
		})
	}
}

// TestDisableJobTrackingWithFinalizers ensures that when the
// JobTrackingWithFinalizers feature is disabled, tracking finalizers are
// removed from all pods, but Job continues to be tracked.
// This test can be removed once the feature graduates to GA.
func TestDisableJobTrackingWithFinalizers(t *testing.T) {
	// Step 1: job created while feature is enabled.
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, true)()

	closeFn, restConfig, clientSet, ns := setup(t, "simple")
	defer closeFn()
	ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
	defer func() {
		cancel()
	}()

	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
		Spec: batchv1.JobSpec{
			Parallelism: pointer.Int32Ptr(2),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	if !hasJobTrackingAnnotation(jobObj) {
		t.Error("apiserver didn't add the tracking annotation")
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 2,
		Ready:  pointer.Int32(0),
	}, true)

	// Step 2: Disable tracking with finalizers.
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, false)()
	cancel()

	// Fail a pod while Job controller is stopped.
	if err := setJobPodsPhase(context.Background(), clientSet, jobObj, v1.PodFailed, 1); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodFailed, err)
	}

	// Restart controller.
	ctx, cancel = startJobControllerAndWaitForCaches(restConfig)

	// Ensure Job continues to be tracked and finalizers are removed.
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 2,
		Failed: 1,
		Ready:  pointer.Int32(0),
	}, false)

	jobObj, err = clientSet.BatchV1().Jobs(jobObj.Namespace).Get(ctx, jobObj.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Obtaining updated Job object: %v", err)
	}
	if hasJobTrackingAnnotation(jobObj) {
		t.Error("controller didn't remove the tracking annotation")
	}

	// Step 3: Reenable tracking with finalizers.
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, true)()
	cancel()

	// Succeed a pod while Job controller is stopped.
	if err := setJobPodsPhase(context.Background(), clientSet, jobObj, v1.PodSucceeded, 1); err != nil {
		t.Fatalf("Failed setting phase %s on Job Pod: %v", v1.PodFailed, err)
	}

	// Restart controller.
	ctx, cancel = startJobControllerAndWaitForCaches(restConfig)

	// Ensure Job continues to be tracked and finalizers are removed.
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active:    1,
		Failed:    1,
		Succeeded: 1,
		Ready:     pointer.Int32(0),
	}, false)
}

func TestOrphanPodsFinalizersClearedWithGC(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, true)()
	for _, policy := range []metav1.DeletionPropagation{metav1.DeletePropagationOrphan, metav1.DeletePropagationBackground, metav1.DeletePropagationForeground} {
		t.Run(string(policy), func(t *testing.T) {
			closeFn, restConfig, clientSet, ns := setup(t, "simple")
			defer closeFn()
			informerSet := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "controller-informers")), 0)
			// Make the job controller significantly slower to trigger race condition.
			restConfig.QPS = 1
			restConfig.Burst = 1
			jc, ctx, cancel := createJobControllerWithSharedInformers(restConfig, informerSet)
			defer cancel()
			restConfig.QPS = 200
			restConfig.Burst = 200
			runGC := createGC(ctx, t, restConfig, informerSet)
			informerSet.Start(ctx.Done())
			go jc.Run(ctx, 1)
			runGC()

			jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: pointer.Int32Ptr(2),
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			if !hasJobTrackingAnnotation(jobObj) {
				t.Error("apiserver didn't add the tracking annotation")
			}
			validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
				Active: 2,
				Ready:  pointer.Int32(0),
			}, true)

			// Delete Job. The GC should delete the pods in cascade.
			err = clientSet.BatchV1().Jobs(jobObj.Namespace).Delete(ctx, jobObj.Name, metav1.DeleteOptions{
				PropagationPolicy: &policy,
			})
			if err != nil {
				t.Fatalf("Failed to delete job: %v", err)
			}
			validateNoOrphanPodsWithFinalizers(ctx, t, clientSet, jobObj)
		})
	}
}

func TestFinalizersClearedWhenBackoffLimitExceeded(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, true)()

	closeFn, restConfig, clientSet, ns := setup(t, "simple")
	defer closeFn()
	ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
	defer cancel()

	// Job tracking with finalizers requires less calls in Indexed mode,
	// so it's more likely to process all finalizers before all the pods
	// are visible.
	mode := batchv1.IndexedCompletion
	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
		Spec: batchv1.JobSpec{
			CompletionMode: &mode,
			Completions:    pointer.Int32(500),
			Parallelism:    pointer.Int32(500),
			BackoffLimit:   pointer.Int32(0),
		},
	})
	if err != nil {
		t.Fatalf("Could not create job: %v", err)
	}

	// Fail a pod ASAP.
	err = wait.PollImmediate(time.Millisecond, wait.ForeverTestTimeout, func() (done bool, err error) {
		if err := setJobPodsPhase(ctx, clientSet, jobObj, v1.PodFailed, 1); err != nil {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Could not fail pod: %v", err)
	}

	validateJobFailed(ctx, t, clientSet, jobObj)

	validateNoOrphanPodsWithFinalizers(ctx, t, clientSet, jobObj)
}

func validateNoOrphanPodsWithFinalizers(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job) {
	t.Helper()
	orphanPods := 0
	if err := wait.Poll(waitInterval, wait.ForeverTestTimeout, func() (done bool, err error) {
		pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{
			LabelSelector: metav1.FormatLabelSelector(jobObj.Spec.Selector),
		})
		if err != nil {
			return false, err
		}
		orphanPods = 0
		for _, pod := range pods.Items {
			if hasJobTrackingFinalizer(&pod) {
				orphanPods++
			}
		}
		return orphanPods == 0, nil
	}); err != nil {
		t.Errorf("Failed waiting for pods to be freed from finalizer: %v", err)
		t.Logf("Last saw %d orphan pods", orphanPods)
	}
}

func TestOrphanPodsFinalizersClearedWithFeatureDisabled(t *testing.T) {
	// Step 0: job created while feature is enabled.
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, true)()

	closeFn, restConfig, clientSet, ns := setup(t, "simple")
	defer closeFn()
	ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
	defer func() {
		cancel()
	}()

	jobObj, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
		Spec: batchv1.JobSpec{
			Parallelism: pointer.Int32Ptr(1),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	if !hasJobTrackingAnnotation(jobObj) {
		t.Error("apiserver didn't add the tracking annotation")
	}
	validateJobPodsStatus(ctx, t, clientSet, jobObj, podsByStatus{
		Active: 1,
		Ready:  pointer.Int32(0),
	}, true)

	// Step 2: Disable tracking with finalizers.
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, false)()
	cancel()

	// Delete the Job while controller is stopped.
	err = clientSet.BatchV1().Jobs(jobObj.Namespace).Delete(context.Background(), jobObj.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete job: %v", err)
	}

	// Restart controller.
	ctx, cancel = startJobControllerAndWaitForCaches(restConfig)
	if err := wait.Poll(waitInterval, wait.ForeverTestTimeout, func() (done bool, err error) {
		pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			t.Fatalf("Failed to list Job Pods: %v", err)
		}
		sawPods := false
		for _, pod := range pods.Items {
			if metav1.IsControlledBy(&pod, jobObj) {
				if hasJobTrackingFinalizer(&pod) {
					return false, nil
				}
				sawPods = true
			}
		}
		return sawPods, nil
	}); err != nil {
		t.Errorf("Waiting for finalizers to be removed: %v", err)
	}
}

func TestSuspendJob(t *testing.T) {
	type step struct {
		flag       bool
		wantActive int
		wantStatus v1.ConditionStatus
		wantReason string
	}
	testCases := []struct {
		featureGate bool
		create      step
		update      step
	}{
		// Exhaustively test all combinations other than trivial true->true and
		// false->false cases.
		{
			create: step{flag: false, wantActive: 2},
			update: step{flag: true, wantActive: 0, wantStatus: v1.ConditionTrue, wantReason: "Suspended"},
		},
		{
			create: step{flag: true, wantActive: 0, wantStatus: v1.ConditionTrue, wantReason: "Suspended"},
			update: step{flag: false, wantActive: 2, wantStatus: v1.ConditionFalse, wantReason: "Resumed"},
		},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("feature=%v,create=%v,update=%v", tc.featureGate, tc.create.flag, tc.update.flag)
		t.Run(name, func(t *testing.T) {
			closeFn, restConfig, clientSet, ns := setup(t, "suspend")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer cancel()
			events, err := clientSet.EventsV1().Events(ns.Name).Watch(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer events.Stop()

			parallelism := int32(2)
			job, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
				Spec: batchv1.JobSpec{
					Parallelism: pointer.Int32Ptr(parallelism),
					Completions: pointer.Int32Ptr(4),
					Suspend:     pointer.BoolPtr(tc.create.flag),
				},
			})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}

			validate := func(s string, active int, status v1.ConditionStatus, reason string) {
				validateJobPodsStatus(ctx, t, clientSet, job, podsByStatus{
					Active: active,
					Ready:  pointer.Int32(0),
				}, feature.DefaultFeatureGate.Enabled(features.JobTrackingWithFinalizers))
				job, err = clientSet.BatchV1().Jobs(ns.Name).Get(ctx, job.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to get Job after %s: %v", s, err)
				}
				if got, want := getJobConditionStatus(ctx, job, batchv1.JobSuspended), status; got != want {
					t.Errorf("Unexpected Job condition %q status after %s: got %q, want %q", batchv1.JobSuspended, s, got, want)
				}
				if err := waitForEvent(events, job.UID, reason); err != nil {
					t.Errorf("Waiting for event with reason %q after %s: %v", reason, s, err)
				}
			}
			validate("create", tc.create.wantActive, tc.create.wantStatus, tc.create.wantReason)

			job.Spec.Suspend = pointer.BoolPtr(tc.update.flag)
			job, err = clientSet.BatchV1().Jobs(ns.Name).Update(ctx, job, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to update Job: %v", err)
			}
			validate("update", tc.update.wantActive, tc.update.wantStatus, tc.update.wantReason)
		})
	}
}

func TestSuspendJobControllerRestart(t *testing.T) {
	closeFn, restConfig, clientSet, ns := setup(t, "suspend")
	defer closeFn()
	ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
	defer cancel()

	job, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{
		Spec: batchv1.JobSpec{
			Parallelism: pointer.Int32Ptr(2),
			Completions: pointer.Int32Ptr(4),
			Suspend:     pointer.BoolPtr(true),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create Job: %v", err)
	}
	validateJobPodsStatus(ctx, t, clientSet, job, podsByStatus{
		Active: 0,
		Ready:  pointer.Int32(0),
	}, true)
}

func TestNodeSelectorUpdate(t *testing.T) {
	for name, featureGate := range map[string]bool{
		"feature gate disabled": false,
		"feature gate enabled":  true,
	} {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobMutableNodeSchedulingDirectives, featureGate)()

			closeFn, restConfig, clientSet, ns := setup(t, "suspend")
			defer closeFn()
			ctx, cancel := startJobControllerAndWaitForCaches(restConfig)
			defer cancel()

			job, err := createJobWithDefaults(ctx, clientSet, ns.Name, &batchv1.Job{Spec: batchv1.JobSpec{
				Parallelism: pointer.Int32Ptr(1),
				Suspend:     pointer.BoolPtr(true),
			}})
			if err != nil {
				t.Fatalf("Failed to create Job: %v", err)
			}
			jobName := job.Name
			jobNamespace := job.Namespace
			jobClient := clientSet.BatchV1().Jobs(jobNamespace)

			// (1) Unsuspend and set node selector in the same update.
			nodeSelector := map[string]string{"foo": "bar"}
			_, err = updateJob(ctx, jobClient, jobName, func(j *batchv1.Job) {
				j.Spec.Template.Spec.NodeSelector = nodeSelector
				j.Spec.Suspend = pointer.BoolPtr(false)
			})
			if !featureGate {
				if err == nil || !strings.Contains(err.Error(), "spec.template: Invalid value") {
					t.Errorf("Expected \"spec.template: Invalid value\" error, got: %v", err)
				}
			} else if featureGate && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			// (2) Check that the pod was created using the expected node selector.
			if featureGate {
				var pod *v1.Pod
				if err := wait.PollImmediate(waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
					pods, err := clientSet.CoreV1().Pods(jobNamespace).List(ctx, metav1.ListOptions{})
					if err != nil {
						t.Fatalf("Failed to list Job Pods: %v", err)
					}
					if len(pods.Items) == 0 {
						return false, nil
					}
					pod = &pods.Items[0]
					return true, nil
				}); err != nil || pod == nil {
					t.Fatalf("pod not found: %v", err)
				}

				// if the feature gate is enabled, then the job should now be unsuspended and
				// the pod has the node selector.
				if diff := cmp.Diff(nodeSelector, pod.Spec.NodeSelector); diff != "" {
					t.Errorf("Unexpected nodeSelector (-want,+got):\n%s", diff)
				}
			}

			// (3) Update node selector again. It should fail since the job is unsuspended.
			_, err = updateJob(ctx, jobClient, jobName, func(j *batchv1.Job) {
				j.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "baz"}
			})

			if err == nil || !strings.Contains(err.Error(), "spec.template: Invalid value") {
				t.Errorf("Expected \"spec.template: Invalid value\" error, got: %v", err)
			}

		})
	}
}

type podsByStatus struct {
	Active    int
	Ready     *int32
	Failed    int
	Succeeded int
}

func validateJobPodsStatus(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job, desired podsByStatus, wFinalizer bool) {
	t.Helper()
	var actualCounts podsByStatus
	if err := wait.Poll(waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		updatedJob, err := clientSet.BatchV1().Jobs(jobObj.Namespace).Get(ctx, jobObj.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get updated Job: %v", err)
		}
		actualCounts = podsByStatus{
			Active:    int(updatedJob.Status.Active),
			Ready:     updatedJob.Status.Ready,
			Succeeded: int(updatedJob.Status.Succeeded),
			Failed:    int(updatedJob.Status.Failed),
		}
		return cmp.Equal(actualCounts, desired), nil
	}); err != nil {
		diff := cmp.Diff(desired, actualCounts)
		t.Errorf("Waiting for Job Status: %v\nPods (-want,+got):\n%s", err, diff)
	}
	var active []*v1.Pod
	if err := wait.PollImmediate(waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			t.Fatalf("Failed to list Job Pods: %v", err)
		}
		active = nil
		for _, pod := range pods.Items {
			phase := pod.Status.Phase
			if metav1.IsControlledBy(&pod, jobObj) && (phase == v1.PodPending || phase == v1.PodRunning) {
				p := pod
				active = append(active, &p)
			}
		}
		return len(active) == desired.Active, nil
	}); err != nil {
		if len(active) != desired.Active {
			t.Errorf("Found %d active Pods, want %d", len(active), desired.Active)
		}
	}
	for _, p := range active {
		if got := hasJobTrackingFinalizer(p); got != wFinalizer {
			t.Errorf("Pod %s has tracking finalizer %t, want %t", p.Name, got, wFinalizer)
		}
	}
}

func validateFinishedPodsNoFinalizer(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job) {
	t.Helper()
	pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list Job Pods: %v", err)
	}
	for _, pod := range pods.Items {
		phase := pod.Status.Phase
		if metav1.IsControlledBy(&pod, jobObj) && (phase == v1.PodPending || phase == v1.PodRunning) && hasJobTrackingFinalizer(&pod) {
			t.Errorf("Finished pod %s still has a tracking finalizer", pod.Name)
		}
	}
}

// validateIndexedJobPods validates indexes and hostname of
// active and completed Pods of an Indexed Job.
// Call after validateJobPodsStatus
func validateIndexedJobPods(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job, wantActive sets.Int, gotCompleted string) {
	t.Helper()
	updatedJob, err := clientSet.BatchV1().Jobs(jobObj.Namespace).Get(ctx, jobObj.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get updated Job: %v", err)
	}
	if updatedJob.Status.CompletedIndexes != gotCompleted {
		t.Errorf("Got completed indexes %q, want %q", updatedJob.Status.CompletedIndexes, gotCompleted)
	}
	pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list Job Pods: %v", err)
	}
	gotActive := sets.NewInt()
	for _, pod := range pods.Items {
		if metav1.IsControlledBy(&pod, jobObj) {
			if pod.Status.Phase == v1.PodPending || pod.Status.Phase == v1.PodRunning {
				ix, err := getCompletionIndex(&pod)
				if err != nil {
					t.Errorf("Failed getting completion index for pod %s: %v", pod.Name, err)
				} else {
					gotActive.Insert(ix)
				}
				expectedName := fmt.Sprintf("%s-%d", jobObj.Name, ix)
				if diff := cmp.Equal(expectedName, pod.Spec.Hostname); !diff {
					t.Errorf("Got pod hostname %s, want %s", pod.Spec.Hostname, expectedName)
				}
			}
		}
	}
	if wantActive == nil {
		wantActive = sets.NewInt()
	}
	if diff := cmp.Diff(wantActive.List(), gotActive.List()); diff != "" {
		t.Errorf("Unexpected active indexes (-want,+got):\n%s", diff)
	}
}

func waitForEvent(events watch.Interface, uid types.UID, reason string) error {
	if reason == "" {
		return nil
	}
	return wait.Poll(waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		for {
			var ev watch.Event
			select {
			case ev = <-events.ResultChan():
			default:
				return false, nil
			}
			e, ok := ev.Object.(*eventsv1.Event)
			if !ok {
				continue
			}
			ctrl := "job-controller"
			if (e.ReportingController == ctrl || e.DeprecatedSource.Component == ctrl) && e.Reason == reason && e.Regarding.UID == uid {
				return true, nil
			}
		}
	})
}

func getJobConditionStatus(ctx context.Context, job *batchv1.Job, cType batchv1.JobConditionType) v1.ConditionStatus {
	for _, cond := range job.Status.Conditions {
		if cond.Type == cType {
			return cond.Status
		}
	}
	return ""
}

func validateJobFailed(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job) {
	t.Helper()
	validateJobCondition(ctx, t, clientSet, jobObj, batchv1.JobFailed)
}

func validateJobSucceeded(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job) {
	t.Helper()
	validateJobCondition(ctx, t, clientSet, jobObj, batchv1.JobComplete)
}

func validateJobCondition(ctx context.Context, t *testing.T, clientSet clientset.Interface, jobObj *batchv1.Job, cond batchv1.JobConditionType) {
	t.Helper()
	if err := wait.Poll(waitInterval, wait.ForeverTestTimeout, func() (bool, error) {
		j, err := clientSet.BatchV1().Jobs(jobObj.Namespace).Get(ctx, jobObj.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to obtain updated Job: %v", err)
		}
		return getJobConditionStatus(ctx, j, cond) == v1.ConditionTrue, nil
	}); err != nil {
		t.Errorf("Waiting for Job to have condition %s: %v", cond, err)
	}
}

func setJobPodsPhase(ctx context.Context, clientSet clientset.Interface, jobObj *batchv1.Job, phase v1.PodPhase, cnt int) error {
	op := func(p *v1.Pod) bool {
		p.Status.Phase = phase
		return true
	}
	return updateJobPodsStatus(ctx, clientSet, jobObj, op, cnt)
}

func setJobPodsReady(ctx context.Context, clientSet clientset.Interface, jobObj *batchv1.Job, cnt int) error {
	op := func(p *v1.Pod) bool {
		if podutil.IsPodReady(p) {
			return false
		}
		p.Status.Conditions = append(p.Status.Conditions, v1.PodCondition{
			Type:   v1.PodReady,
			Status: v1.ConditionTrue,
		})
		return true
	}
	return updateJobPodsStatus(ctx, clientSet, jobObj, op, cnt)
}

func updateJobPodsStatus(ctx context.Context, clientSet clientset.Interface, jobObj *batchv1.Job, op func(*v1.Pod) bool, cnt int) error {
	pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("listing Job Pods: %w", err)
	}
	updates := make([]v1.Pod, 0, cnt)
	for _, pod := range pods.Items {
		if len(updates) == cnt {
			break
		}
		if p := pod.Status.Phase; metav1.IsControlledBy(&pod, jobObj) && p != v1.PodFailed && p != v1.PodSucceeded {
			if !op(&pod) {
				continue
			}
			updates = append(updates, pod)
		}
	}
	if len(updates) != cnt {
		return fmt.Errorf("couldn't set phase on %d Job Pods", cnt)
	}
	return updatePodStatuses(ctx, clientSet, updates)
}

func updatePodStatuses(ctx context.Context, clientSet clientset.Interface, updates []v1.Pod) error {
	wg := sync.WaitGroup{}
	wg.Add(len(updates))
	errCh := make(chan error, len(updates))

	for _, pod := range updates {
		pod := pod
		go func() {
			_, err := clientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(ctx, &pod, metav1.UpdateOptions{})
			if err != nil {
				errCh <- err
			}
			wg.Done()
		}()
	}
	wg.Wait()

	select {
	case err := <-errCh:
		return fmt.Errorf("updating Pod status: %w", err)
	default:
	}
	return nil
}

func setJobPhaseForIndex(ctx context.Context, clientSet clientset.Interface, jobObj *batchv1.Job, phase v1.PodPhase, ix int) error {
	pods, err := clientSet.CoreV1().Pods(jobObj.Namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("listing Job Pods: %w", err)
	}
	for _, pod := range pods.Items {
		if p := pod.Status.Phase; !metav1.IsControlledBy(&pod, jobObj) || p == v1.PodFailed || p == v1.PodSucceeded {
			continue
		}
		if pix, err := getCompletionIndex(&pod); err == nil && pix == ix {
			pod.Status.Phase = phase
			_, err := clientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(ctx, &pod, metav1.UpdateOptions{})
			if err != nil {
				return fmt.Errorf("updating pod %s status: %w", pod.Name, err)
			}
			return nil
		}
	}
	return errors.New("no pod matching index found")
}

func getCompletionIndex(p *v1.Pod) (int, error) {
	if p.Annotations == nil {
		return 0, errors.New("no annotations found")
	}
	v, ok := p.Annotations[batchv1.JobCompletionIndexAnnotation]
	if !ok {
		return 0, fmt.Errorf("annotation %s not found", batchv1.JobCompletionIndexAnnotation)
	}
	return strconv.Atoi(v)
}

func createJobWithDefaults(ctx context.Context, clientSet clientset.Interface, ns string, jobObj *batchv1.Job) (*batchv1.Job, error) {
	if jobObj.Name == "" {
		jobObj.Name = "test-job"
	}
	if len(jobObj.Spec.Template.Spec.Containers) == 0 {
		jobObj.Spec.Template.Spec.Containers = []v1.Container{
			{Name: "foo", Image: "bar"},
		}
	}
	if jobObj.Spec.Template.Spec.RestartPolicy == "" {
		jobObj.Spec.Template.Spec.RestartPolicy = v1.RestartPolicyNever
	}
	return clientSet.BatchV1().Jobs(ns).Create(ctx, jobObj, metav1.CreateOptions{})
}

func setup(t *testing.T, nsBaseName string) (framework.CloseFunc, *restclient.Config, clientset.Interface, *v1.Namespace) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	config.QPS = 200
	config.Burst = 200
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(clientSet, nsBaseName, t)
	closeFn := func() {
		framework.DeleteNamespaceOrDie(clientSet, ns, t)
		server.TearDownFn()
	}
	return closeFn, config, clientSet, ns
}

func startJobControllerAndWaitForCaches(restConfig *restclient.Config) (context.Context, context.CancelFunc) {
	informerSet := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "job-informers")), 0)
	jc, ctx, cancel := createJobControllerWithSharedInformers(restConfig, informerSet)
	informerSet.Start(ctx.Done())
	go jc.Run(ctx, 1)

	// since this method starts the controller in a separate goroutine
	// and the tests don't check /readyz there is no way
	// the tests can tell it is safe to call the server and requests won't be rejected
	// thus we wait until caches have synced
	informerSet.WaitForCacheSync(ctx.Done())
	return ctx, cancel
}

func createJobControllerWithSharedInformers(restConfig *restclient.Config, informerSet informers.SharedInformerFactory) (*jobcontroller.Controller, context.Context, context.CancelFunc) {
	clientSet := clientset.NewForConfigOrDie(restclient.AddUserAgent(restConfig, "job-controller"))
	ctx, cancel := context.WithCancel(context.Background())
	jc := jobcontroller.NewController(informerSet.Core().V1().Pods(), informerSet.Batch().V1().Jobs(), clientSet)
	return jc, ctx, cancel
}

func createGC(ctx context.Context, t *testing.T, restConfig *restclient.Config, informerSet informers.SharedInformerFactory) func() {
	restConfig = restclient.AddUserAgent(restConfig, "gc-controller")
	clientSet := clientset.NewForConfigOrDie(restConfig)
	metadataClient, err := metadata.NewForConfig(restConfig)
	if err != nil {
		t.Fatalf("Failed to create metadataClient: %v", err)
	}
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cacheddiscovery.NewMemCacheClient(clientSet.Discovery()))
	restMapper.Reset()
	metadataInformers := metadatainformer.NewSharedInformerFactory(metadataClient, 0)
	alwaysStarted := make(chan struct{})
	close(alwaysStarted)
	gc, err := garbagecollector.NewGarbageCollector(
		clientSet,
		metadataClient,
		restMapper,
		garbagecollector.DefaultIgnoredResources(),
		informerfactory.NewInformerFactory(informerSet, metadataInformers),
		alwaysStarted,
	)
	if err != nil {
		t.Fatalf("Failed creating garbage collector")
	}
	startGC := func() {
		syncPeriod := 5 * time.Second
		go wait.Until(func() {
			restMapper.Reset()
		}, syncPeriod, ctx.Done())
		go gc.Run(ctx, 1)
		go gc.Sync(clientSet.Discovery(), syncPeriod, ctx.Done())
	}
	return startGC
}

func hasJobTrackingFinalizer(obj metav1.Object) bool {
	for _, fin := range obj.GetFinalizers() {
		if fin == batchv1.JobTrackingFinalizer {
			return true
		}
	}
	return false
}

func hasJobTrackingAnnotation(job *batchv1.Job) bool {
	if job.Annotations == nil {
		return false
	}
	_, ok := job.Annotations[batchv1.JobTrackingFinalizer]
	return ok
}

func setDuringTest(val *int, newVal int) func() {
	origVal := *val
	*val = newVal
	return func() {
		*val = origVal
	}
}

func updateJob(ctx context.Context, jobClient typedv1.JobInterface, jobName string, updateFunc func(*batchv1.Job)) (*batchv1.Job, error) {
	var job *batchv1.Job
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newJob, err := jobClient.Get(ctx, jobName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newJob)
		job, err = jobClient.Update(ctx, newJob, metav1.UpdateOptions{})
		return err
	})
	return job, err
}
