/*
Copyright 2015 The Kubernetes Authors.

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
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/job/metrics"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

var alwaysReady = func() bool { return true }

func newJobWithName(name string, parallelism, completions, backoffLimit int32, completionMode batch.CompletionMode) *batch.Job {
	j := &batch.Job{
		TypeMeta: metav1.TypeMeta{Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			UID:       uuid.NewUUID(),
			Namespace: metav1.NamespaceDefault,
		},
		Spec: batch.JobSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Image: "foo/bar"},
					},
				},
			},
		},
	}
	if completionMode != "" {
		j.Spec.CompletionMode = &completionMode
	}
	// Special case: -1 for either completions or parallelism means leave nil (negative is not allowed
	// in practice by validation.
	if completions >= 0 {
		j.Spec.Completions = &completions
	} else {
		j.Spec.Completions = nil
	}
	if parallelism >= 0 {
		j.Spec.Parallelism = &parallelism
	} else {
		j.Spec.Parallelism = nil
	}
	j.Spec.BackoffLimit = &backoffLimit

	return j
}

func newJob(parallelism, completions, backoffLimit int32, completionMode batch.CompletionMode) *batch.Job {
	return newJobWithName("foobar", parallelism, completions, backoffLimit, completionMode)
}

func newControllerFromClient(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) (*Controller, informers.SharedInformerFactory) {
	sharedInformers := informers.NewSharedInformerFactory(kubeClient, resyncPeriod())
	jm := NewController(sharedInformers.Core().V1().Pods(), sharedInformers.Batch().V1().Jobs(), kubeClient)
	jm.podControl = &controller.FakePodControl{}

	return jm, sharedInformers
}

func newPod(name string, job *batch.Job) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			UID:             types.UID(name),
			Labels:          job.Spec.Selector.MatchLabels,
			Namespace:       job.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, controllerKind)},
		},
	}
}

// create count pods with the given phase for the given job
func newPodList(count int, status v1.PodPhase, job *batch.Job) []*v1.Pod {
	var pods []*v1.Pod
	for i := 0; i < count; i++ {
		newPod := newPod(fmt.Sprintf("pod-%v", rand.String(10)), job)
		newPod.Status = v1.PodStatus{Phase: status}
		if trackingUncountedPods(job) {
			newPod.Finalizers = append(newPod.Finalizers, batch.JobTrackingFinalizer)
		}
		pods = append(pods, newPod)
	}
	return pods
}

func setPodsStatuses(podIndexer cache.Indexer, job *batch.Job, pendingPods, activePods, succeededPods, failedPods, readyPods int) {
	for _, pod := range newPodList(pendingPods, v1.PodPending, job) {
		podIndexer.Add(pod)
	}
	running := newPodList(activePods, v1.PodRunning, job)
	for i, p := range running {
		if i >= readyPods {
			break
		}
		p.Status.Conditions = append(p.Status.Conditions, v1.PodCondition{
			Type:   v1.PodReady,
			Status: v1.ConditionTrue,
		})
	}
	for _, pod := range running {
		podIndexer.Add(pod)
	}
	for _, pod := range newPodList(succeededPods, v1.PodSucceeded, job) {
		podIndexer.Add(pod)
	}
	for _, pod := range newPodList(failedPods, v1.PodFailed, job) {
		podIndexer.Add(pod)
	}
}

func setPodsStatusesWithIndexes(podIndexer cache.Indexer, job *batch.Job, status []indexPhase) {
	for _, s := range status {
		p := newPod(fmt.Sprintf("pod-%s", rand.String(10)), job)
		p.Status = v1.PodStatus{Phase: s.Phase}
		if s.Index != noIndex {
			p.Annotations = map[string]string{
				batch.JobCompletionIndexAnnotation: s.Index,
			}
			p.Spec.Hostname = fmt.Sprintf("%s-%s", job.Name, s.Index)
		}
		if trackingUncountedPods(job) {
			p.Finalizers = append(p.Finalizers, batch.JobTrackingFinalizer)
		}
		podIndexer.Add(p)
	}
}

func TestControllerSyncJob(t *testing.T) {
	jobConditionComplete := batch.JobComplete
	jobConditionFailed := batch.JobFailed
	jobConditionSuspended := batch.JobSuspended

	testCases := map[string]struct {
		// job setup
		parallelism    int32
		completions    int32
		backoffLimit   int32
		deleting       bool
		podLimit       int
		completionMode batch.CompletionMode
		wasSuspended   bool
		suspend        bool

		// If set, it means that the case is exclusive to tracking with/without finalizers.
		wFinalizersExclusive *bool

		// pod setup
		podControllerError        error
		jobKeyForget              bool
		pendingPods               int
		activePods                int
		readyPods                 int
		succeededPods             int
		failedPods                int
		podsWithIndexes           []indexPhase
		fakeExpectationAtCreation int32 // negative: ExpectDeletions, positive: ExpectCreations

		// expectations
		expectedCreations       int32
		expectedDeletions       int32
		expectedActive          int32
		expectedReady           *int32
		expectedSucceeded       int32
		expectedCompletedIdxs   string
		expectedFailed          int32
		expectedCondition       *batch.JobConditionType
		expectedConditionStatus v1.ConditionStatus
		expectedConditionReason string
		expectedCreatedIndexes  sets.Int

		// only applicable to tracking with finalizers
		expectedPodPatches int

		// features
		jobReadyPodsEnabled bool
	}{
		"job start": {
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			jobKeyForget:      true,
			expectedCreations: 2,
			expectedActive:    2,
		},
		"WQ job start": {
			parallelism:       2,
			completions:       -1,
			backoffLimit:      6,
			jobKeyForget:      true,
			expectedCreations: 2,
			expectedActive:    2,
		},
		"pending pods": {
			parallelism:    2,
			completions:    5,
			backoffLimit:   6,
			jobKeyForget:   true,
			pendingPods:    2,
			expectedActive: 2,
		},
		"correct # of pods": {
			parallelism:    3,
			completions:    5,
			backoffLimit:   6,
			jobKeyForget:   true,
			activePods:     3,
			readyPods:      2,
			expectedActive: 3,
		},
		"correct # of pods, ready enabled": {
			parallelism:         3,
			completions:         5,
			backoffLimit:        6,
			jobKeyForget:        true,
			activePods:          3,
			readyPods:           2,
			expectedActive:      3,
			expectedReady:       pointer.Int32(2),
			jobReadyPodsEnabled: true,
		},
		"WQ job: correct # of pods": {
			parallelism:    2,
			completions:    -1,
			backoffLimit:   6,
			jobKeyForget:   true,
			activePods:     2,
			expectedActive: 2,
		},
		"too few active pods": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			jobKeyForget:       true,
			activePods:         1,
			succeededPods:      1,
			expectedCreations:  1,
			expectedActive:     2,
			expectedSucceeded:  1,
			expectedPodPatches: 1,
		},
		"too few active pods with a dynamic job": {
			parallelism:       2,
			completions:       -1,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        1,
			expectedCreations: 1,
			expectedActive:    2,
		},
		"too few active pods, with controller error": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			podControllerError: fmt.Errorf("fake error"),
			jobKeyForget:       true,
			activePods:         1,
			succeededPods:      1,
			expectedCreations:  1,
			expectedActive:     1,
			expectedSucceeded:  1,
		},
		"too many active pods": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			jobKeyForget:       true,
			activePods:         3,
			expectedDeletions:  1,
			expectedActive:     2,
			expectedPodPatches: 1,
		},
		"too many active pods, with controller error": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			podControllerError: fmt.Errorf("fake error"),
			jobKeyForget:       true,
			activePods:         3,
			expectedDeletions:  1,
			expectedActive:     3,
		},
		"failed + succeed pods: reset backoff delay": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			podControllerError: fmt.Errorf("fake error"),
			jobKeyForget:       true,
			activePods:         1,
			succeededPods:      1,
			failedPods:         1,
			expectedCreations:  1,
			expectedActive:     1,
			expectedSucceeded:  1,
			expectedFailed:     1,
		},
		"new failed pod": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			activePods:         1,
			failedPods:         1,
			expectedCreations:  1,
			expectedActive:     2,
			expectedFailed:     1,
			expectedPodPatches: 1,
		},
		"only new failed pod with controller error": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			podControllerError: fmt.Errorf("fake error"),
			activePods:         1,
			failedPods:         1,
			expectedCreations:  1,
			expectedActive:     1,
			expectedFailed:     1,
		},
		"job finish": {
			parallelism:             2,
			completions:             5,
			backoffLimit:            6,
			jobKeyForget:            true,
			succeededPods:           5,
			expectedSucceeded:       5,
			expectedCondition:       &jobConditionComplete,
			expectedConditionStatus: v1.ConditionTrue,
			expectedPodPatches:      5,
		},
		"WQ job finishing": {
			parallelism:        2,
			completions:        -1,
			backoffLimit:       6,
			jobKeyForget:       true,
			activePods:         1,
			succeededPods:      1,
			expectedActive:     1,
			expectedSucceeded:  1,
			expectedPodPatches: 1,
		},
		"WQ job all finished": {
			parallelism:             2,
			completions:             -1,
			backoffLimit:            6,
			jobKeyForget:            true,
			succeededPods:           2,
			expectedSucceeded:       2,
			expectedCondition:       &jobConditionComplete,
			expectedConditionStatus: v1.ConditionTrue,
			expectedPodPatches:      2,
		},
		"WQ job all finished despite one failure": {
			parallelism:             2,
			completions:             -1,
			backoffLimit:            6,
			jobKeyForget:            true,
			succeededPods:           1,
			failedPods:              1,
			expectedSucceeded:       1,
			expectedFailed:          1,
			expectedCondition:       &jobConditionComplete,
			expectedConditionStatus: v1.ConditionTrue,
			expectedPodPatches:      2,
		},
		"more active pods than parallelism": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			jobKeyForget:       true,
			activePods:         10,
			expectedDeletions:  8,
			expectedActive:     2,
			expectedPodPatches: 8,
		},
		"more active pods than remaining completions": {
			parallelism:        3,
			completions:        4,
			backoffLimit:       6,
			jobKeyForget:       true,
			activePods:         3,
			succeededPods:      2,
			expectedDeletions:  1,
			expectedActive:     2,
			expectedSucceeded:  2,
			expectedPodPatches: 3,
		},
		"status change": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			jobKeyForget:       true,
			activePods:         2,
			succeededPods:      2,
			expectedActive:     2,
			expectedSucceeded:  2,
			expectedPodPatches: 2,
		},
		"deleting job": {
			parallelism:        2,
			completions:        5,
			backoffLimit:       6,
			deleting:           true,
			jobKeyForget:       true,
			pendingPods:        1,
			activePods:         1,
			succeededPods:      1,
			expectedActive:     2,
			expectedSucceeded:  1,
			expectedPodPatches: 3,
		},
		"limited pods": {
			parallelism:       100,
			completions:       200,
			backoffLimit:      6,
			podLimit:          10,
			jobKeyForget:      true,
			expectedCreations: 10,
			expectedActive:    10,
		},
		"too many job failures": {
			parallelism:             2,
			completions:             5,
			deleting:                true,
			jobKeyForget:            true,
			failedPods:              1,
			expectedFailed:          1,
			expectedCondition:       &jobConditionFailed,
			expectedConditionStatus: v1.ConditionTrue,
			expectedConditionReason: "BackoffLimitExceeded",
			expectedPodPatches:      1,
		},
		"job failures, unsatisfied expectations": {
			wFinalizersExclusive:      pointer.Bool(true),
			parallelism:               2,
			completions:               5,
			deleting:                  true,
			failedPods:                1,
			fakeExpectationAtCreation: 1,
			expectedFailed:            1,
			expectedPodPatches:        1,
		},
		"indexed job start": {
			parallelism:            2,
			completions:            5,
			backoffLimit:           6,
			completionMode:         batch.IndexedCompletion,
			jobKeyForget:           true,
			expectedCreations:      2,
			expectedActive:         2,
			expectedCreatedIndexes: sets.NewInt(0, 1),
		},
		"indexed job completed": {
			parallelism:    2,
			completions:    3,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			jobKeyForget:   true,
			podsWithIndexes: []indexPhase{
				{"0", v1.PodSucceeded},
				{"1", v1.PodFailed},
				{"1", v1.PodSucceeded},
				{"2", v1.PodSucceeded},
			},
			expectedSucceeded:       3,
			expectedFailed:          1,
			expectedCompletedIdxs:   "0-2",
			expectedCondition:       &jobConditionComplete,
			expectedConditionStatus: v1.ConditionTrue,
			expectedPodPatches:      4,
		},
		"indexed job repeated completed index": {
			parallelism:    2,
			completions:    3,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			jobKeyForget:   true,
			podsWithIndexes: []indexPhase{
				{"0", v1.PodSucceeded},
				{"1", v1.PodSucceeded},
				{"1", v1.PodSucceeded},
			},
			expectedCreations:      1,
			expectedActive:         1,
			expectedSucceeded:      2,
			expectedCompletedIdxs:  "0,1",
			expectedCreatedIndexes: sets.NewInt(2),
			expectedPodPatches:     3,
		},
		"indexed job some running and completed pods": {
			parallelism:    8,
			completions:    20,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			podsWithIndexes: []indexPhase{
				{"0", v1.PodRunning},
				{"2", v1.PodSucceeded},
				{"3", v1.PodPending},
				{"4", v1.PodSucceeded},
				{"5", v1.PodSucceeded},
				{"7", v1.PodSucceeded},
				{"8", v1.PodSucceeded},
				{"9", v1.PodSucceeded},
			},
			jobKeyForget:           true,
			expectedCreations:      6,
			expectedActive:         8,
			expectedSucceeded:      6,
			expectedCompletedIdxs:  "2,4,5,7-9",
			expectedCreatedIndexes: sets.NewInt(1, 6, 10, 11, 12, 13),
			expectedPodPatches:     6,
		},
		"indexed job some failed pods": {
			parallelism:    3,
			completions:    4,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			podsWithIndexes: []indexPhase{
				{"0", v1.PodFailed},
				{"1", v1.PodPending},
				{"2", v1.PodFailed},
			},
			expectedCreations:      2,
			expectedActive:         3,
			expectedFailed:         2,
			expectedCreatedIndexes: sets.NewInt(0, 2),
			expectedPodPatches:     2,
		},
		"indexed job some pods without index": {
			parallelism:    2,
			completions:    5,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			activePods:     1,
			succeededPods:  1,
			failedPods:     1,
			podsWithIndexes: []indexPhase{
				{"invalid", v1.PodRunning},
				{"invalid", v1.PodSucceeded},
				{"invalid", v1.PodFailed},
				{"invalid", v1.PodPending},
				{"0", v1.PodSucceeded},
				{"1", v1.PodRunning},
				{"2", v1.PodRunning},
			},
			jobKeyForget:          true,
			expectedDeletions:     3,
			expectedActive:        2,
			expectedSucceeded:     1,
			expectedFailed:        0,
			expectedCompletedIdxs: "0",
			expectedPodPatches:    8,
		},
		"indexed job repeated indexes": {
			parallelism:    5,
			completions:    5,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			succeededPods:  1,
			failedPods:     1,
			podsWithIndexes: []indexPhase{
				{"invalid", v1.PodRunning},
				{"0", v1.PodSucceeded},
				{"1", v1.PodRunning},
				{"2", v1.PodRunning},
				{"2", v1.PodPending},
			},
			jobKeyForget:          true,
			expectedCreations:     0,
			expectedDeletions:     2,
			expectedActive:        2,
			expectedSucceeded:     1,
			expectedCompletedIdxs: "0",
			expectedPodPatches:    5,
		},
		"indexed job with indexes outside of range": {
			parallelism:    2,
			completions:    5,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			podsWithIndexes: []indexPhase{
				{"0", v1.PodSucceeded},
				{"5", v1.PodRunning},
				{"6", v1.PodSucceeded},
				{"7", v1.PodPending},
				{"8", v1.PodFailed},
			},
			jobKeyForget:          true,
			expectedCreations:     0, // only one of creations and deletions can happen in a sync
			expectedSucceeded:     1,
			expectedDeletions:     2,
			expectedCompletedIdxs: "0",
			expectedActive:        0,
			expectedFailed:        0,
			expectedPodPatches:    5,
		},
		"suspending a job with satisfied expectations": {
			// Suspended Job should delete active pods when expectations are
			// satisfied.
			suspend:                 true,
			parallelism:             2,
			activePods:              2, // parallelism == active, expectations satisfied
			completions:             4,
			backoffLimit:            6,
			jobKeyForget:            true,
			expectedCreations:       0,
			expectedDeletions:       2,
			expectedActive:          0,
			expectedCondition:       &jobConditionSuspended,
			expectedConditionStatus: v1.ConditionTrue,
			expectedConditionReason: "JobSuspended",
			expectedPodPatches:      2,
		},
		"suspending a job with unsatisfied expectations": {
			// Unlike the previous test, we expect the controller to NOT suspend the
			// Job in the syncJob call because the controller will wait for
			// expectations to be satisfied first. The next syncJob call (not tested
			// here) will be the same as the previous test.
			suspend:                   true,
			parallelism:               2,
			activePods:                3,  // active > parallelism, expectations unsatisfied
			fakeExpectationAtCreation: -1, // the controller is expecting a deletion
			completions:               4,
			backoffLimit:              6,
			jobKeyForget:              true,
			expectedCreations:         0,
			expectedDeletions:         0,
			expectedActive:            3,
		},
		"resuming a suspended job": {
			wasSuspended:            true,
			suspend:                 false,
			parallelism:             2,
			completions:             4,
			backoffLimit:            6,
			jobKeyForget:            true,
			expectedCreations:       2,
			expectedDeletions:       0,
			expectedActive:          2,
			expectedCondition:       &jobConditionSuspended,
			expectedConditionStatus: v1.ConditionFalse,
			expectedConditionReason: "JobResumed",
		},
		"suspending a deleted job": {
			// We would normally expect the active pods to be deleted (see a few test
			// cases above), but since this job is being deleted, we don't expect
			// anything changed here from before the job was suspended. The
			// JobSuspended condition is also missing.
			suspend:            true,
			deleting:           true,
			parallelism:        2,
			activePods:         2, // parallelism == active, expectations satisfied
			completions:        4,
			backoffLimit:       6,
			jobKeyForget:       true,
			expectedCreations:  0,
			expectedDeletions:  0,
			expectedActive:     2,
			expectedPodPatches: 2,
		},
	}

	for name, tc := range testCases {
		for _, wFinalizers := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s, finalizers=%t", name, wFinalizers), func(t *testing.T) {
				if wFinalizers && tc.podControllerError != nil {
					t.Skip("Can't track status if finalizers can't be removed")
				}
				if tc.wFinalizersExclusive != nil && *tc.wFinalizersExclusive != wFinalizers {
					t.Skipf("Test is exclusive for wFinalizers=%t", *tc.wFinalizersExclusive)
				}
				defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobReadyPods, tc.jobReadyPodsEnabled)()
				defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, wFinalizers)()

				// job manager setup
				clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
				manager, sharedInformerFactory := newControllerFromClient(clientSet, controller.NoResyncPeriodFunc)
				fakePodControl := controller.FakePodControl{Err: tc.podControllerError, CreateLimit: tc.podLimit}
				manager.podControl = &fakePodControl
				manager.podStoreSynced = alwaysReady
				manager.jobStoreSynced = alwaysReady

				// job & pods setup
				job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, tc.completionMode)
				job.Spec.Suspend = pointer.BoolPtr(tc.suspend)
				key, err := controller.KeyFunc(job)
				if err != nil {
					t.Errorf("Unexpected error getting job key: %v", err)
				}
				if tc.fakeExpectationAtCreation < 0 {
					manager.expectations.ExpectDeletions(key, int(-tc.fakeExpectationAtCreation))
				} else if tc.fakeExpectationAtCreation > 0 {
					manager.expectations.ExpectCreations(key, int(tc.fakeExpectationAtCreation))
				}
				if tc.wasSuspended {
					job.Status.Conditions = append(job.Status.Conditions, *newCondition(batch.JobSuspended, v1.ConditionTrue, "JobSuspended", "Job suspended"))
				}
				if wFinalizers {
					job.Annotations = map[string]string{
						batch.JobTrackingFinalizer: "",
					}
				}
				if tc.deleting {
					now := metav1.Now()
					job.DeletionTimestamp = &now
				}
				sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
				podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
				setPodsStatuses(podIndexer, job, tc.pendingPods, tc.activePods, tc.succeededPods, tc.failedPods, tc.readyPods)
				setPodsStatusesWithIndexes(podIndexer, job, tc.podsWithIndexes)

				actual := job
				manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
					actual = job
					return job, nil
				}

				// run
				forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))

				// We need requeue syncJob task if podController error
				if tc.podControllerError != nil {
					if err == nil {
						t.Error("Syncing jobs expected to return error on podControl exception")
					}
				} else if tc.expectedCondition == nil && (hasValidFailingPods(tc.podsWithIndexes, int(tc.completions)) || (tc.completionMode != batch.IndexedCompletion && tc.failedPods > 0)) {
					if err == nil {
						t.Error("Syncing jobs expected to return error when there are new failed pods and Job didn't finish")
					}
				} else if tc.podLimit != 0 && fakePodControl.CreateCallCount > tc.podLimit {
					if err == nil {
						t.Error("Syncing jobs expected to return error when reached the podControl limit")
					}
				} else if err != nil {
					t.Errorf("Unexpected error when syncing jobs: %v", err)
				}
				if forget != tc.jobKeyForget {
					t.Errorf("Unexpected forget value. Expected %v, saw %v\n", tc.jobKeyForget, forget)
				}
				// validate created/deleted pods
				if int32(len(fakePodControl.Templates)) != tc.expectedCreations {
					t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", tc.expectedCreations, len(fakePodControl.Templates))
				}
				if tc.completionMode == batch.IndexedCompletion {
					checkIndexedJobPods(t, &fakePodControl, tc.expectedCreatedIndexes, job.Name)
				} else {
					for _, p := range fakePodControl.Templates {
						// Fake pod control doesn't add generate name from the owner reference.
						if p.GenerateName != "" {
							t.Errorf("Got pod generate name %s, want %s", p.GenerateName, "")
						}
						if p.Spec.Hostname != "" {
							t.Errorf("Got pod hostname %q, want none", p.Spec.Hostname)
						}
					}
				}
				if int32(len(fakePodControl.DeletePodName)) != tc.expectedDeletions {
					t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", tc.expectedDeletions, len(fakePodControl.DeletePodName))
				}
				// Each create should have an accompanying ControllerRef.
				if len(fakePodControl.ControllerRefs) != int(tc.expectedCreations) {
					t.Errorf("Unexpected number of ControllerRefs.  Expected %d, saw %d\n", tc.expectedCreations, len(fakePodControl.ControllerRefs))
				}
				// Make sure the ControllerRefs are correct.
				for _, controllerRef := range fakePodControl.ControllerRefs {
					if got, want := controllerRef.APIVersion, "batch/v1"; got != want {
						t.Errorf("controllerRef.APIVersion = %q, want %q", got, want)
					}
					if got, want := controllerRef.Kind, "Job"; got != want {
						t.Errorf("controllerRef.Kind = %q, want %q", got, want)
					}
					if got, want := controllerRef.Name, job.Name; got != want {
						t.Errorf("controllerRef.Name = %q, want %q", got, want)
					}
					if got, want := controllerRef.UID, job.UID; got != want {
						t.Errorf("controllerRef.UID = %q, want %q", got, want)
					}
					if controllerRef.Controller == nil || *controllerRef.Controller != true {
						t.Errorf("controllerRef.Controller is not set to true")
					}
				}
				// validate status
				if actual.Status.Active != tc.expectedActive {
					t.Errorf("Unexpected number of active pods.  Expected %d, saw %d\n", tc.expectedActive, actual.Status.Active)
				}
				if diff := cmp.Diff(tc.expectedReady, actual.Status.Ready); diff != "" {
					t.Errorf("Unexpected number of ready pods (-want,+got): %s", diff)
				}
				if actual.Status.Succeeded != tc.expectedSucceeded {
					t.Errorf("Unexpected number of succeeded pods.  Expected %d, saw %d\n", tc.expectedSucceeded, actual.Status.Succeeded)
				}
				if diff := cmp.Diff(tc.expectedCompletedIdxs, actual.Status.CompletedIndexes); diff != "" {
					t.Errorf("Unexpected completed indexes (-want,+got):\n%s", diff)
				}
				if actual.Status.Failed != tc.expectedFailed {
					t.Errorf("Unexpected number of failed pods.  Expected %d, saw %d\n", tc.expectedFailed, actual.Status.Failed)
				}
				if actual.Status.StartTime != nil && tc.suspend {
					t.Error("Unexpected .status.startTime not nil when suspend is true")
				}
				if actual.Status.StartTime == nil && !tc.suspend {
					t.Error("Missing .status.startTime")
				}
				// validate conditions
				if tc.expectedCondition != nil {
					if !getCondition(actual, *tc.expectedCondition, tc.expectedConditionStatus, tc.expectedConditionReason) {
						t.Errorf("Expected completion condition.  Got %#v", actual.Status.Conditions)
					}
				} else {
					if cond := hasTrueCondition(actual); cond != nil {
						t.Errorf("Got condition %s, want none", *cond)
					}
				}
				if tc.expectedCondition == nil && tc.suspend && len(actual.Status.Conditions) != 0 {
					t.Errorf("Unexpected conditions %v", actual.Status.Conditions)
				}
				// validate slow start
				expectedLimit := 0
				for pass := uint8(0); expectedLimit <= tc.podLimit; pass++ {
					expectedLimit += controller.SlowStartInitialBatchSize << pass
				}
				if tc.podLimit > 0 && fakePodControl.CreateCallCount > expectedLimit {
					t.Errorf("Unexpected number of create calls.  Expected <= %d, saw %d\n", fakePodControl.CreateLimit*2, fakePodControl.CreateCallCount)
				}
				wantPodPatches := 0
				if wFinalizers {
					wantPodPatches = tc.expectedPodPatches
				}
				if p := len(fakePodControl.Patches); p != wantPodPatches {
					t.Errorf("Got %d pod patches, want %d", p, wantPodPatches)
				}
			})
		}
	}
}

func checkIndexedJobPods(t *testing.T, control *controller.FakePodControl, wantIndexes sets.Int, jobName string) {
	t.Helper()
	gotIndexes := sets.NewInt()
	for _, p := range control.Templates {
		checkJobCompletionEnvVariable(t, &p.Spec)
		ix := getCompletionIndex(p.Annotations)
		if ix == -1 {
			t.Errorf("Created pod %s didn't have completion index", p.Name)
		} else {
			gotIndexes.Insert(ix)
		}
		expectedName := fmt.Sprintf("%s-%d", jobName, ix)
		if expectedName != p.Spec.Hostname {
			t.Errorf("Got pod hostname %s, want %s", p.Spec.Hostname, expectedName)
		}
		expectedName += "-"
		if expectedName != p.GenerateName {
			t.Errorf("Got pod generate name %s, want %s", p.GenerateName, expectedName)
		}
	}
	if diff := cmp.Diff(wantIndexes.List(), gotIndexes.List()); diff != "" {
		t.Errorf("Unexpected created completion indexes (-want,+got):\n%s", diff)
	}
}

// TestSyncJobLegacyTracking makes sure that a Job is only tracked with
// finalizers only when the feature is enabled and the job has the finalizer.
func TestSyncJobLegacyTracking(t *testing.T) {
	cases := map[string]struct {
		job                           batch.Job
		trackingWithFinalizersEnabled bool
		wantUncounted                 bool
		wantPatches                   int
	}{
		"no annotation": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "ns",
				},
				Spec: batch.JobSpec{
					Parallelism: pointer.Int32Ptr(1),
				},
			},
		},
		"no annotation, feature enabled": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "ns",
				},
				Spec: batch.JobSpec{
					Parallelism: pointer.Int32Ptr(1),
				},
			},
			trackingWithFinalizersEnabled: true,
		},
		"tracking annotation, feature disabled": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "ns",
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
					},
				},
				Spec: batch.JobSpec{
					Parallelism: pointer.Int32Ptr(1),
				},
			},
			// Finalizer removed.
			wantPatches: 1,
		},
		"tracking annotation, feature enabled": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "ns",
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
					},
				},
				Spec: batch.JobSpec{
					Parallelism: pointer.Int32Ptr(1),
				},
			},
			trackingWithFinalizersEnabled: true,
			wantUncounted:                 true,
		},
		"different annotation, feature enabled": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "ns",
					Annotations: map[string]string{
						"foo": "bar",
					},
				},
				Spec: batch.JobSpec{
					Parallelism: pointer.Int32Ptr(1),
				},
			},
			trackingWithFinalizersEnabled: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.trackingWithFinalizersEnabled)()

			// Job manager setup.
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(clientSet, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			jobPatches := 0
			manager.patchJobHandler = func(context.Context, *batch.Job, []byte) error {
				jobPatches++
				return nil
			}
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(&tc.job)

			var actual *batch.Job
			manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
				actual = job
				return job, nil
			}

			// Run.
			_, err := manager.syncJob(context.TODO(), testutil.GetKey(&tc.job, t))
			if err != nil {
				t.Fatalf("Syncing job: %v", err)
			}

			// Checks.
			if got := actual.Status.UncountedTerminatedPods != nil; got != tc.wantUncounted {
				t.Errorf("Job got uncounted pods %t, want %t", got, tc.wantUncounted)
			}
			if jobPatches != tc.wantPatches {
				t.Errorf("Sync did %d patches, want %d", jobPatches, tc.wantPatches)
			}
		})
	}
}

func TestGetStatus(t *testing.T) {
	cases := map[string]struct {
		job                  batch.Job
		pods                 []*v1.Pod
		expectedRmFinalizers sets.String
		wantSucceeded        int32
		wantFailed           int32
	}{
		"without finalizers": {
			job: batch.Job{
				Status: batch.JobStatus{
					Succeeded: 1,
					Failed:    2,
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).Pod,
				buildPod().uid("b").phase(v1.PodSucceeded).Pod,
				buildPod().uid("c").phase(v1.PodFailed).Pod,
				buildPod().uid("d").phase(v1.PodFailed).Pod,
				buildPod().uid("e").phase(v1.PodFailed).Pod,
				buildPod().uid("f").phase(v1.PodRunning).Pod,
			},
			wantSucceeded: 2,
			wantFailed:    3,
		},
		"some counted": {
			job: batch.Job{
				Status: batch.JobStatus{
					Succeeded:               2,
					Failed:                  1,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).Pod,
				buildPod().uid("b").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("d").phase(v1.PodFailed).Pod,
				buildPod().uid("e").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("f").phase(v1.PodRunning).Pod,
			},
			wantSucceeded: 4,
			wantFailed:    2,
		},
		"some uncounted": {
			job: batch.Job{
				Status: batch.JobStatus{
					Succeeded: 1,
					Failed:    1,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "c"},
						Failed:    []types.UID{"e", "f"},
					},
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).Pod,
				buildPod().uid("b").phase(v1.PodSucceeded).Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("d").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("e").phase(v1.PodFailed).Pod,
				buildPod().uid("f").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("g").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			wantSucceeded: 4,
			wantFailed:    4,
		},
		"with expected removed finalizers": {
			job: batch.Job{
				Status: batch.JobStatus{
					Succeeded: 2,
					Failed:    2,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"d"},
					},
				},
			},
			expectedRmFinalizers: sets.NewString("b", "f"),
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).Pod,
				buildPod().uid("b").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("d").phase(v1.PodFailed).Pod,
				buildPod().uid("e").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("f").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("g").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			wantSucceeded: 4,
			wantFailed:    5,
		},
		"deleted pods": {
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).deletionTimestamp().Pod,
				buildPod().uid("b").phase(v1.PodFailed).deletionTimestamp().Pod,
				buildPod().uid("c").phase(v1.PodRunning).deletionTimestamp().Pod,
				buildPod().uid("d").phase(v1.PodPending).deletionTimestamp().Pod,
			},
			wantSucceeded: 1,
			wantFailed:    1,
		},
		"deleted pods, tracking with finalizers": {
			job: batch.Job{
				Status: batch.JobStatus{
					Succeeded:               1,
					Failed:                  1,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().uid("c").phase(v1.PodRunning).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().uid("d").phase(v1.PodPending).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().uid("e").phase(v1.PodRunning).deletionTimestamp().Pod,
				buildPod().uid("f").phase(v1.PodPending).deletionTimestamp().Pod,
			},
			wantSucceeded: 2,
			wantFailed:    4,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var uncounted *uncountedTerminatedPods
			if tc.job.Status.UncountedTerminatedPods != nil {
				uncounted = newUncountedTerminatedPods(*tc.job.Status.UncountedTerminatedPods)
			}
			succeeded, failed := getStatus(&tc.job, tc.pods, uncounted, tc.expectedRmFinalizers)
			if succeeded != tc.wantSucceeded {
				t.Errorf("getStatus reports %d succeeded pods, want %d", succeeded, tc.wantSucceeded)
			}
			if failed != tc.wantFailed {
				t.Errorf("getStatus reports %d succeeded pods, want %d", failed, tc.wantFailed)
			}
		})
	}
}

func TestTrackJobStatusAndRemoveFinalizers(t *testing.T) {
	succeededCond := newCondition(batch.JobComplete, v1.ConditionTrue, "", "")
	failedCond := newCondition(batch.JobFailed, v1.ConditionTrue, "", "")
	indexedCompletion := batch.IndexedCompletion
	mockErr := errors.New("mock error")
	cases := map[string]struct {
		job                  batch.Job
		pods                 []*v1.Pod
		finishedCond         *batch.JobCondition
		expectedRmFinalizers sets.String
		needsFlush           bool
		statusUpdateErr      error
		podControlErr        error
		wantErr              error
		wantRmFinalizers     int
		wantStatusUpdates    []batch.JobStatus
	}{
		"no updates": {},
		"new active": {
			job: batch.Job{
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			needsFlush: true,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Active:                  1,
				},
			},
		},
		"track finished pods": {
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().uid("d").phase(v1.PodFailed).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().uid("e").phase(v1.PodPending).trackingFinalizer().deletionTimestamp().Pod,
				buildPod().phase(v1.PodPending).trackingFinalizer().Pod,
				buildPod().phase(v1.PodRunning).trackingFinalizer().Pod,
			},
			wantRmFinalizers: 5,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "c"},
						Failed:    []types.UID{"b", "d", "e"},
					},
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Succeeded:               2,
					Failed:                  3,
				},
			},
		},
		"past and new finished pods": {
			job: batch.Job{
				Status: batch.JobStatus{
					Active:    1,
					Succeeded: 2,
					Failed:    3,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "e"},
						Failed:    []types.UID{"b", "f"},
					},
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("e").phase(v1.PodSucceeded).Pod,
				buildPod().phase(v1.PodFailed).Pod,
				buildPod().phase(v1.PodPending).Pod,
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("d").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			wantRmFinalizers: 4,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "c"},
						Failed:    []types.UID{"b", "d"},
					},
					Active:    1,
					Succeeded: 3,
					Failed:    4,
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Active:                  1,
					Succeeded:               5,
					Failed:                  6,
				},
			},
		},
		"expecting removed finalizers": {
			job: batch.Job{
				Status: batch.JobStatus{
					Succeeded: 2,
					Failed:    3,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "g"},
						Failed:    []types.UID{"b", "h"},
					},
				},
			},
			expectedRmFinalizers: sets.NewString("c", "d", "g", "h"),
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("d").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("e").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("f").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("g").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("h").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			wantRmFinalizers: 4,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a", "e"},
						Failed:    []types.UID{"b", "f"},
					},
					Succeeded: 3,
					Failed:    4,
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Succeeded:               5,
					Failed:                  6,
				},
			},
		},
		"succeeding job": {
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			finishedCond:     succeededCond,
			wantRmFinalizers: 2,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"b"},
					},
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Succeeded:               1,
					Failed:                  1,
					Conditions:              []batch.JobCondition{*succeededCond},
					CompletionTime:          &succeededCond.LastTransitionTime,
				},
			},
		},
		"failing job": {
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().uid("c").phase(v1.PodRunning).trackingFinalizer().Pod,
			},
			finishedCond: failedCond,
			// Running pod counts as failed.
			wantRmFinalizers: 3,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"b", "c"},
					},
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Succeeded:               1,
					Failed:                  2,
					Conditions:              []batch.JobCondition{*failedCond},
				},
			},
		},
		"deleted job": {
			job: batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &metav1.Time{},
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
				buildPod().phase(v1.PodRunning).trackingFinalizer().Pod,
			},
			// Removing finalizer from Running pod, but doesn't count as failed.
			wantRmFinalizers: 3,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"b"},
					},
					Active: 1,
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					Active:                  1,
					Succeeded:               1,
					Failed:                  1,
				},
			},
		},
		"status update error": {
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			statusUpdateErr: mockErr,
			wantErr:         mockErr,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"b"},
					},
				},
			},
		},
		"pod patch errors": {
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			podControlErr:    mockErr,
			wantErr:          mockErr,
			wantRmFinalizers: 2,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"b"},
					},
				},
			},
		},
		"pod patch errors with partial success": {
			job: batch.Job{
				Status: batch.JobStatus{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
						Failed:    []types.UID{"b"},
					},
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodSucceeded).Pod,
				buildPod().uid("c").phase(v1.PodSucceeded).trackingFinalizer().Pod,
				buildPod().uid("d").phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			podControlErr:    mockErr,
			wantErr:          mockErr,
			wantRmFinalizers: 2,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"c"},
						Failed:    []types.UID{"d"},
					},
					Succeeded: 1,
					Failed:    1,
				},
			},
		},
		"indexed job new successful pods": {
			job: batch.Job{
				Spec: batch.JobSpec{
					CompletionMode: &indexedCompletion,
					Completions:    pointer.Int32Ptr(6),
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			pods: []*v1.Pod{
				buildPod().phase(v1.PodSucceeded).trackingFinalizer().index("1").Pod,
				buildPod().phase(v1.PodSucceeded).trackingFinalizer().index("3").Pod,
				buildPod().phase(v1.PodSucceeded).trackingFinalizer().index("3").Pod,
				buildPod().phase(v1.PodRunning).trackingFinalizer().index("5").Pod,
				buildPod().phase(v1.PodSucceeded).trackingFinalizer().Pod,
			},
			wantRmFinalizers: 4,
			wantStatusUpdates: []batch.JobStatus{
				{
					Active:                  1,
					Succeeded:               2,
					CompletedIndexes:        "1,3",
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
				},
			},
		},
		"indexed job new failed pods": {
			job: batch.Job{
				Spec: batch.JobSpec{
					CompletionMode: &indexedCompletion,
					Completions:    pointer.Int32Ptr(6),
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			pods: []*v1.Pod{
				buildPod().uid("a").phase(v1.PodFailed).trackingFinalizer().index("1").Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().index("3").Pod,
				buildPod().uid("c").phase(v1.PodFailed).trackingFinalizer().index("3").Pod,
				buildPod().uid("d").phase(v1.PodRunning).trackingFinalizer().index("5").Pod,
				buildPod().phase(v1.PodFailed).trackingFinalizer().Pod,
			},
			wantRmFinalizers: 4,
			wantStatusUpdates: []batch.JobStatus{
				{
					Active: 1,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a", "b", "c"},
					},
				},
				{
					Active:                  1,
					Failed:                  3,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
				},
			},
		},
		"indexed job past and new pods": {
			job: batch.Job{
				Spec: batch.JobSpec{
					CompletionMode: &indexedCompletion,
					Completions:    pointer.Int32Ptr(7),
				},
				Status: batch.JobStatus{
					Failed:           2,
					Succeeded:        5,
					CompletedIndexes: "0-2,4,6,7",
				},
			},
			pods: []*v1.Pod{
				buildPod().phase(v1.PodSucceeded).index("0").Pod,
				buildPod().phase(v1.PodFailed).index("1").Pod,
				buildPod().phase(v1.PodSucceeded).trackingFinalizer().index("1").Pod,
				buildPod().phase(v1.PodSucceeded).trackingFinalizer().index("3").Pod,
				buildPod().uid("a").phase(v1.PodFailed).trackingFinalizer().index("2").Pod,
				buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().index("5").Pod,
			},
			wantRmFinalizers: 4,
			wantStatusUpdates: []batch.JobStatus{
				{
					Succeeded:        6,
					Failed:           2,
					CompletedIndexes: "0-4,6",
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a", "b"},
					},
				},
				{
					Succeeded:               6,
					Failed:                  4,
					CompletedIndexes:        "0-4,6",
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
				},
			},
		},
		"too many finished": {
			job: batch.Job{
				Status: batch.JobStatus{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a", "b"},
					},
				},
			},
			pods: func() []*v1.Pod {
				pods := make([]*v1.Pod, 500)
				for i := range pods {
					pods[i] = buildPod().uid(strconv.Itoa(i)).phase(v1.PodSucceeded).trackingFinalizer().Pod
				}
				pods = append(pods, buildPod().uid("b").phase(v1.PodFailed).trackingFinalizer().Pod)
				return pods
			}(),
			wantRmFinalizers: 499,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: func() []types.UID {
							uids := make([]types.UID, 499)
							for i := range uids {
								uids[i] = types.UID(strconv.Itoa(i))
							}
							return uids
						}(),
						Failed: []types.UID{"b"},
					},
					Failed: 1,
				},
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"b"},
					},
					Succeeded: 499,
					Failed:    1,
				},
			},
		},
		"too many indexed finished": {
			job: batch.Job{
				Spec: batch.JobSpec{
					CompletionMode: &indexedCompletion,
					Completions:    pointer.Int32Ptr(501),
				},
			},
			pods: func() []*v1.Pod {
				pods := make([]*v1.Pod, 501)
				for i := range pods {
					pods[i] = buildPod().uid(strconv.Itoa(i)).index(strconv.Itoa(i)).phase(v1.PodSucceeded).trackingFinalizer().Pod
				}
				return pods
			}(),
			wantRmFinalizers: 500,
			wantStatusUpdates: []batch.JobStatus{
				{
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{},
					CompletedIndexes:        "0-499",
					Succeeded:               500,
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, _ := newControllerFromClient(clientSet, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{Err: tc.podControlErr}
			metrics.JobPodsFinished.Reset()
			manager.podControl = &fakePodControl
			var statusUpdates []batch.JobStatus
			manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
				statusUpdates = append(statusUpdates, *job.Status.DeepCopy())
				return job, tc.statusUpdateErr
			}
			job := tc.job.DeepCopy()
			if job.Status.UncountedTerminatedPods == nil {
				job.Status.UncountedTerminatedPods = &batch.UncountedTerminatedPods{}
			}
			uncounted := newUncountedTerminatedPods(*job.Status.UncountedTerminatedPods)
			succeededIndexes := succeededIndexesFromJob(job)
			err := manager.trackJobStatusAndRemoveFinalizers(context.TODO(), job, tc.pods, succeededIndexes, *uncounted, tc.expectedRmFinalizers, tc.finishedCond, tc.needsFlush)
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Got error %v, want %v", err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.wantStatusUpdates, statusUpdates, cmpopts.IgnoreFields(batch.JobCondition{}, "LastProbeTime", "LastTransitionTime")); diff != "" {
				t.Errorf("Unexpected status updates (-want,+got):\n%s", diff)
			}
			rmFinalizers := len(fakePodControl.Patches)
			if rmFinalizers != tc.wantRmFinalizers {
				t.Errorf("Removed %d finalizers, want %d", rmFinalizers, tc.wantRmFinalizers)
			}
			if tc.wantErr == nil {
				completionMode := completionModeStr(job)
				v, err := metricstestutil.GetCounterMetricValue(metrics.JobPodsFinished.WithLabelValues(completionMode, metrics.Succeeded))
				if err != nil {
					t.Fatalf("Obtaining succeeded job_pods_finished_total: %v", err)
				}
				newSucceeded := job.Status.Succeeded - tc.job.Status.Succeeded
				if float64(newSucceeded) != v {
					t.Errorf("Metric reports %.0f succeeded pods, want %d", v, newSucceeded)
				}
				v, err = metricstestutil.GetCounterMetricValue(metrics.JobPodsFinished.WithLabelValues(completionMode, metrics.Failed))
				if err != nil {
					t.Fatalf("Obtaining failed job_pods_finished_total: %v", err)
				}
				newFailed := job.Status.Failed - tc.job.Status.Failed
				if float64(newFailed) != v {
					t.Errorf("Metric reports %.0f failed pods, want %d", v, newFailed)
				}
			}
		})
	}
}

func TestSyncJobPastDeadline(t *testing.T) {
	testCases := map[string]struct {
		// job setup
		parallelism           int32
		completions           int32
		activeDeadlineSeconds int64
		startTime             int64
		backoffLimit          int32
		suspend               bool

		// pod setup
		activePods    int
		succeededPods int
		failedPods    int

		// expectations
		expectedForGetKey       bool
		expectedDeletions       int32
		expectedActive          int32
		expectedSucceeded       int32
		expectedFailed          int32
		expectedCondition       batch.JobConditionType
		expectedConditionReason string
	}{
		"activeDeadlineSeconds less than single pod execution": {
			parallelism:             1,
			completions:             1,
			activeDeadlineSeconds:   10,
			startTime:               15,
			backoffLimit:            6,
			activePods:              1,
			expectedForGetKey:       true,
			expectedDeletions:       1,
			expectedFailed:          1,
			expectedCondition:       batch.JobFailed,
			expectedConditionReason: "DeadlineExceeded",
		},
		"activeDeadlineSeconds bigger than single pod execution": {
			parallelism:             1,
			completions:             2,
			activeDeadlineSeconds:   10,
			startTime:               15,
			backoffLimit:            6,
			activePods:              1,
			succeededPods:           1,
			expectedForGetKey:       true,
			expectedDeletions:       1,
			expectedSucceeded:       1,
			expectedFailed:          1,
			expectedCondition:       batch.JobFailed,
			expectedConditionReason: "DeadlineExceeded",
		},
		"activeDeadlineSeconds times-out before any pod starts": {
			parallelism:             1,
			completions:             1,
			activeDeadlineSeconds:   10,
			startTime:               10,
			backoffLimit:            6,
			expectedForGetKey:       true,
			expectedCondition:       batch.JobFailed,
			expectedConditionReason: "DeadlineExceeded",
		},
		"activeDeadlineSeconds with backofflimit reach": {
			parallelism:             1,
			completions:             1,
			activeDeadlineSeconds:   1,
			startTime:               10,
			failedPods:              1,
			expectedForGetKey:       true,
			expectedFailed:          1,
			expectedCondition:       batch.JobFailed,
			expectedConditionReason: "BackoffLimitExceeded",
		},
		"activeDeadlineSeconds is not triggered when Job is suspended": {
			suspend:                 true,
			parallelism:             1,
			completions:             2,
			activeDeadlineSeconds:   10,
			startTime:               15,
			backoffLimit:            6,
			expectedForGetKey:       true,
			expectedCondition:       batch.JobSuspended,
			expectedConditionReason: "JobSuspended",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			// job manager setup
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(clientSet, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			var actual *batch.Job
			manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
				actual = job
				return job, nil
			}

			// job & pods setup
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
			job.Spec.ActiveDeadlineSeconds = &tc.activeDeadlineSeconds
			job.Spec.Suspend = pointer.BoolPtr(tc.suspend)
			start := metav1.Unix(metav1.Now().Time.Unix()-tc.startTime, 0)
			job.Status.StartTime = &start
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			setPodsStatuses(podIndexer, job, 0, tc.activePods, tc.succeededPods, tc.failedPods, 0)

			// run
			forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))
			if err != nil {
				t.Errorf("Unexpected error when syncing jobs %v", err)
			}
			if forget != tc.expectedForGetKey {
				t.Errorf("Unexpected forget value. Expected %v, saw %v\n", tc.expectedForGetKey, forget)
			}
			// validate created/deleted pods
			if int32(len(fakePodControl.Templates)) != 0 {
				t.Errorf("Unexpected number of creates.  Expected 0, saw %d\n", len(fakePodControl.Templates))
			}
			if int32(len(fakePodControl.DeletePodName)) != tc.expectedDeletions {
				t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", tc.expectedDeletions, len(fakePodControl.DeletePodName))
			}
			// validate status
			if actual.Status.Active != tc.expectedActive {
				t.Errorf("Unexpected number of active pods.  Expected %d, saw %d\n", tc.expectedActive, actual.Status.Active)
			}
			if actual.Status.Succeeded != tc.expectedSucceeded {
				t.Errorf("Unexpected number of succeeded pods.  Expected %d, saw %d\n", tc.expectedSucceeded, actual.Status.Succeeded)
			}
			if actual.Status.Failed != tc.expectedFailed {
				t.Errorf("Unexpected number of failed pods.  Expected %d, saw %d\n", tc.expectedFailed, actual.Status.Failed)
			}
			if actual.Status.StartTime == nil {
				t.Error("Missing .status.startTime")
			}
			// validate conditions
			if !getCondition(actual, tc.expectedCondition, v1.ConditionTrue, tc.expectedConditionReason) {
				t.Errorf("Expected fail condition.  Got %#v", actual.Status.Conditions)
			}
		})
	}
}

func getCondition(job *batch.Job, condition batch.JobConditionType, status v1.ConditionStatus, reason string) bool {
	for _, v := range job.Status.Conditions {
		if v.Type == condition && v.Status == status && v.Reason == reason {
			return true
		}
	}
	return false
}

func hasTrueCondition(job *batch.Job) *batch.JobConditionType {
	for _, v := range job.Status.Conditions {
		if v.Status == v1.ConditionTrue {
			return &v.Type
		}
	}
	return nil
}

func TestSyncPastDeadlineJobFinished(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sharedInformerFactory.Start(ctx.Done())
	sharedInformerFactory.WaitForCacheSync(ctx.Done())

	go manager.Run(ctx, 1)

	tests := []struct {
		name         string
		setStartTime bool
		jobName      string
	}{
		{
			name:         "New job created without start time being set",
			setStartTime: false,
			jobName:      "job1",
		},
		{
			name:         "New job created with start time being set",
			setStartTime: true,
			jobName:      "job2",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			job := newJobWithName(tc.jobName, 1, 1, 6, batch.NonIndexedCompletion)
			activeDeadlineSeconds := int64(1)
			job.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
			if tc.setStartTime {
				start := metav1.Unix(metav1.Now().Time.Unix()-1, 0)
				job.Status.StartTime = &start
			}

			_, err := clientset.BatchV1().Jobs(job.GetNamespace()).Create(ctx, job, metav1.CreateOptions{})
			if err != nil {
				t.Errorf("Could not create Job: %v", err)
			}

			if err := sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job); err != nil {
				t.Fatalf("Failed to insert job in index: %v", err)
			}
			var j *batch.Job
			err = wait.Poll(200*time.Millisecond, 3*time.Second, func() (done bool, err error) {
				j, err = clientset.BatchV1().Jobs(metav1.NamespaceDefault).Get(ctx, job.GetName(), metav1.GetOptions{})
				if err != nil {
					return false, nil
				}
				if len(j.Status.Conditions) == 1 && j.Status.Conditions[0].Reason == "DeadlineExceeded" {
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Errorf("Job failed to enforce activeDeadlineSeconds configuration. Expected condition with Reason 'DeadlineExceeded' was not found in %v", j.Status)
			}
		})
	}
}

func TestSingleJobFailedCondition(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	var actual *batch.Job
	manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
		actual = job
		return job, nil
	}

	job := newJob(1, 1, 6, batch.NonIndexedCompletion)
	activeDeadlineSeconds := int64(10)
	job.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	start := metav1.Unix(metav1.Now().Time.Unix()-15, 0)
	job.Status.StartTime = &start
	job.Status.Conditions = append(job.Status.Conditions, *newCondition(batch.JobFailed, v1.ConditionFalse, "DeadlineExceeded", "Job was active longer than specified deadline"))
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))
	if err != nil {
		t.Errorf("Unexpected error when syncing jobs %v", err)
	}
	if !forget {
		t.Errorf("Unexpected forget value. Expected %v, saw %v\n", true, forget)
	}
	if len(fakePodControl.DeletePodName) != 0 {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", 0, len(fakePodControl.DeletePodName))
	}
	if actual == nil {
		t.Error("Expected job modification\n")
	}
	failedConditions := getConditionsByType(actual.Status.Conditions, batch.JobFailed)
	if len(failedConditions) != 1 {
		t.Error("Unexpected number of failed conditions\n")
	}
	if failedConditions[0].Status != v1.ConditionTrue {
		t.Errorf("Unexpected status for the failed condition. Expected: %v, saw %v\n", v1.ConditionTrue, failedConditions[0].Status)
	}

}

func TestSyncJobComplete(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady

	job := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job.Status.Conditions = append(job.Status.Conditions, *newCondition(batch.JobComplete, v1.ConditionTrue, "", ""))
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))
	if err != nil {
		t.Fatalf("Unexpected error when syncing jobs %v", err)
	}
	if !forget {
		t.Errorf("Unexpected forget value. Expected %v, saw %v\n", true, forget)
	}
	actual, err := manager.jobLister.Jobs(job.Namespace).Get(job.Name)
	if err != nil {
		t.Fatalf("Unexpected error when trying to get job from the store: %v", err)
	}
	// Verify that after syncing a complete job, the conditions are the same.
	if got, expected := len(actual.Status.Conditions), 1; got != expected {
		t.Fatalf("Unexpected job status conditions amount; expected %d, got %d", expected, got)
	}
}

func TestSyncJobDeleted(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, _ := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
		return job, nil
	}
	job := newJob(2, 2, 6, batch.NonIndexedCompletion)
	forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))
	if err != nil {
		t.Errorf("Unexpected error when syncing jobs %v", err)
	}
	if !forget {
		t.Errorf("Unexpected forget value. Expected %v, saw %v\n", true, forget)
	}
	if len(fakePodControl.Templates) != 0 {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", 0, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != 0 {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", 0, len(fakePodControl.DeletePodName))
	}
}

func TestSyncJobUpdateRequeue(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	DefaultJobBackOff = time.Duration(0) // overwrite the default value for testing
	cases := map[string]struct {
		updateErr      error
		wantRequeue    bool
		withFinalizers bool
	}{
		"no error": {},
		"generic error": {
			updateErr:   fmt.Errorf("update error"),
			wantRequeue: true,
		},
		"conflict error": {
			updateErr:   apierrors.NewConflict(schema.GroupResource{}, "", nil),
			wantRequeue: true,
		},
		"conflict error, with finalizers": {
			withFinalizers: true,
			updateErr:      apierrors.NewConflict(schema.GroupResource{}, "", nil),
			wantRequeue:    true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, tc.withFinalizers)()
			manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
				return job, tc.updateErr
			}
			job := newJob(2, 2, 6, batch.NonIndexedCompletion)
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			manager.queue.Add(testutil.GetKey(job, t))
			manager.processNextWorkItem(context.TODO())
			// With DefaultJobBackOff=0, the queueing is synchronous.
			requeued := manager.queue.Len() > 0
			if requeued != tc.wantRequeue {
				t.Errorf("Unexpected requeue, got %t, want %t", requeued, tc.wantRequeue)
			}
			if requeued {
				key, _ := manager.queue.Get()
				expectedKey := testutil.GetKey(job, t)
				if key != expectedKey {
					t.Errorf("Expected requeue of job with key %s got %s", expectedKey, key)
				}
			}
		})
	}
}

func TestJobPodLookup(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	testCases := []struct {
		job *batch.Job
		pod *v1.Pod

		expectedName string
	}{
		// pods without labels don't match any job
		{
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "basic"},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo1", Namespace: metav1.NamespaceAll},
			},
			expectedName: "",
		},
		// matching labels, different namespace
		{
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"foo": "bar"},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo2",
					Namespace: "ns",
					Labels:    map[string]string{"foo": "bar"},
				},
			},
			expectedName: "",
		},
		// matching ns and labels returns
		{
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "ns"},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpIn,
								Values:   []string{"bar"},
							},
						},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo3",
					Namespace: "ns",
					Labels:    map[string]string{"foo": "bar"},
				},
			},
			expectedName: "bar",
		},
	}
	for _, tc := range testCases {
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(tc.job)
		if jobs := manager.getPodJobs(tc.pod); len(jobs) > 0 {
			if got, want := len(jobs), 1; got != want {
				t.Errorf("len(jobs) = %v, want %v", got, want)
			}
			job := jobs[0]
			if tc.expectedName != job.Name {
				t.Errorf("Got job %+v expected %+v", job.Name, tc.expectedName)
			}
		} else if tc.expectedName != "" {
			t.Errorf("Expected a job %v pod %v, found none", tc.expectedName, tc.pod.Name)
		}
	}
}

func TestGetPodsForJob(t *testing.T) {
	job := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job.Name = "test_job"
	otherJob := newJob(1, 1, 6, batch.NonIndexedCompletion)
	otherJob.Name = "other_job"
	cases := map[string]struct {
		jobDeleted        bool
		jobDeletedInCache bool
		pods              []*v1.Pod
		wantPods          []string
		// only applicable to tracking with finalizers
		wantPodsFinalizer []string
	}{
		"only matching": {
			pods: []*v1.Pod{
				buildPod().name("pod1").job(job).trackingFinalizer().Pod,
				buildPod().name("pod2").job(otherJob).Pod,
				buildPod().name("pod3").ns(job.Namespace).Pod,
				buildPod().name("pod4").job(job).Pod,
			},
			wantPods:          []string{"pod1", "pod4"},
			wantPodsFinalizer: []string{"pod1"},
		},
		"adopt": {
			pods: []*v1.Pod{
				buildPod().name("pod1").job(job).Pod,
				buildPod().name("pod2").job(job).clearOwner().Pod,
				buildPod().name("pod3").job(otherJob).Pod,
			},
			wantPods:          []string{"pod1", "pod2"},
			wantPodsFinalizer: []string{"pod2"},
		},
		"no adopt when deleting": {
			jobDeleted:        true,
			jobDeletedInCache: true,
			pods: []*v1.Pod{
				buildPod().name("pod1").job(job).Pod,
				buildPod().name("pod2").job(job).clearOwner().Pod,
			},
			wantPods: []string{"pod1"},
		},
		"no adopt when deleting race": {
			jobDeleted: true,
			pods: []*v1.Pod{
				buildPod().name("pod1").job(job).Pod,
				buildPod().name("pod2").job(job).clearOwner().Pod,
			},
			wantPods: []string{"pod1"},
		},
		"release": {
			pods: []*v1.Pod{
				buildPod().name("pod1").job(job).Pod,
				buildPod().name("pod2").job(job).clearLabels().Pod,
			},
			wantPods: []string{"pod1"},
		},
	}
	for name, tc := range cases {
		for _, wFinalizers := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s, finalizers=%t", name, wFinalizers), func(t *testing.T) {
				job := job.DeepCopy()
				if tc.jobDeleted {
					job.DeletionTimestamp = &metav1.Time{}
				}
				clientSet := fake.NewSimpleClientset(job, otherJob)
				jm, informer := newControllerFromClient(clientSet, controller.NoResyncPeriodFunc)
				jm.podStoreSynced = alwaysReady
				jm.jobStoreSynced = alwaysReady
				cachedJob := job.DeepCopy()
				if tc.jobDeletedInCache {
					cachedJob.DeletionTimestamp = &metav1.Time{}
				}
				informer.Batch().V1().Jobs().Informer().GetIndexer().Add(cachedJob)
				informer.Batch().V1().Jobs().Informer().GetIndexer().Add(otherJob)
				for _, p := range tc.pods {
					informer.Core().V1().Pods().Informer().GetIndexer().Add(p)
				}

				pods, err := jm.getPodsForJob(context.TODO(), job, wFinalizers)
				if err != nil {
					t.Fatalf("getPodsForJob() error: %v", err)
				}
				got := make([]string, len(pods))
				var gotFinalizer []string
				for i, p := range pods {
					got[i] = p.Name
					if hasJobTrackingFinalizer(p) {
						gotFinalizer = append(gotFinalizer, p.Name)
					}
				}
				sort.Strings(got)
				if diff := cmp.Diff(tc.wantPods, got); diff != "" {
					t.Errorf("getPodsForJob() returned (-want,+got):\n%s", diff)
				}
				if wFinalizers {
					sort.Strings(gotFinalizer)
					if diff := cmp.Diff(tc.wantPodsFinalizer, gotFinalizer); diff != "" {
						t.Errorf("Pods with finalizers (-want,+got):\n%s", diff)
					}
				}
			})
		}
	}
}

func TestAddPod(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job2)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	jm.addPod(pod1)
	if got, want := jm.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := jm.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod1.Name)
	}
	expectedKey, _ := controller.KeyFunc(job1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	jm.addPod(pod2)
	if got, want := jm.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = jm.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod2.Name)
	}
	expectedKey, _ = controller.KeyFunc(job2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestAddPodOrphan(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	job3 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job3.Name = "job3"
	job3.Spec.Selector.MatchLabels = map[string]string{"other": "labels"}
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job3)

	pod1 := newPod("pod1", job1)
	// Make pod an orphan. Expect all matching controllers to be queued.
	pod1.OwnerReferences = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	jm.addPod(pod1)
	if got, want := jm.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdatePod(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job2)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	prev := *pod1
	bumpResourceVersion(pod1)
	jm.updatePod(&prev, pod1)
	if got, want := jm.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := jm.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod1.Name)
	}
	expectedKey, _ := controller.KeyFunc(job1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	prev = *pod2
	bumpResourceVersion(pod2)
	jm.updatePod(&prev, pod2)
	if got, want := jm.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = jm.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod2.Name)
	}
	expectedKey, _ = controller.KeyFunc(job2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestUpdatePodOrphanWithNewLabels(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	pod1.OwnerReferences = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	// Labels changed on orphan. Expect newly matching controllers to queue.
	prev := *pod1
	prev.Labels = map[string]string{"foo2": "bar2"}
	bumpResourceVersion(pod1)
	jm.updatePod(&prev, pod1)
	if got, want := jm.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdatePodChangeControllerRef(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	// Changed ControllerRef. Expect both old and new to queue.
	prev := *pod1
	prev.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(job2, controllerKind)}
	bumpResourceVersion(pod1)
	jm.updatePod(&prev, pod1)
	if got, want := jm.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestUpdatePodRelease(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	// Remove ControllerRef. Expect all matching to queue for adoption.
	prev := *pod1
	pod1.OwnerReferences = nil
	bumpResourceVersion(pod1)
	jm.updatePod(&prev, pod1)
	if got, want := jm.queue.Len(), 2; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

func TestDeletePod(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job2)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	jm.deletePod(pod1, true)
	if got, want := jm.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done := jm.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod1.Name)
	}
	expectedKey, _ := controller.KeyFunc(job1)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}

	jm.deletePod(pod2, true)
	if got, want := jm.queue.Len(), 1; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
	key, done = jm.queue.Get()
	if key == nil || done {
		t.Fatalf("failed to enqueue controller for pod %v", pod2.Name)
	}
	expectedKey, _ = controller.KeyFunc(job2)
	if got, want := key.(string), expectedKey; got != want {
		t.Errorf("queue.Get() = %v, want %v", got, want)
	}
}

func TestDeletePodOrphan(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady
	// Disable batching of pod updates.
	jm.podUpdateBatchPeriod = 0

	job1 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job2.Name = "job2"
	job3 := newJob(1, 1, 6, batch.NonIndexedCompletion)
	job3.Name = "job3"
	job3.Spec.Selector.MatchLabels = map[string]string{"other": "labels"}
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job3)

	pod1 := newPod("pod1", job1)
	pod1.OwnerReferences = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)

	jm.deletePod(pod1, true)
	if got, want := jm.queue.Len(), 0; got != want {
		t.Fatalf("queue.Len() = %v, want %v", got, want)
	}
}

type FakeJobExpectations struct {
	*controller.ControllerExpectations
	satisfied    bool
	expSatisfied func()
}

func (fe FakeJobExpectations) SatisfiedExpectations(controllerKey string) bool {
	fe.expSatisfied()
	return fe.satisfied
}

// TestSyncJobExpectations tests that a pod cannot sneak in between counting active pods
// and checking expectations.
func TestSyncJobExpectations(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
		return job, nil
	}

	job := newJob(2, 2, 6, batch.NonIndexedCompletion)
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	pods := newPodList(2, v1.PodPending, job)
	podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
	podIndexer.Add(pods[0])

	manager.expectations = FakeJobExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectations, the job
			// will create a new replica because it doesn't see this pod, but
			// has fulfilled its expectations.
			podIndexer.Add(pods[1])
		},
	}
	manager.syncJob(context.TODO(), testutil.GetKey(job, t))
	if len(fakePodControl.Templates) != 0 {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", 0, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != 0 {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", 0, len(fakePodControl.DeletePodName))
	}
}

func TestWatchJobs(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	clientset.PrependWatchReactor("jobs", core.DefaultWatchReactor(fakeWatch, nil))
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady

	var testJob batch.Job
	received := make(chan struct{})

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler.
	manager.syncHandler = func(ctx context.Context, key string) (bool, error) {
		defer close(received)
		ns, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			t.Errorf("Error getting namespace/name from key %v: %v", key, err)
		}
		job, err := manager.jobLister.Jobs(ns).Get(name)
		if err != nil || job == nil {
			t.Errorf("Expected to find job under key %v: %v", key, err)
			return true, nil
		}
		if !apiequality.Semantic.DeepDerivative(*job, testJob) {
			t.Errorf("Expected %#v, but got %#v", testJob, *job)
		}
		return true, nil
	}
	// Start only the job watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	stopCh := make(chan struct{})
	defer close(stopCh)
	sharedInformerFactory.Start(stopCh)
	go manager.Run(context.TODO(), 1)

	// We're sending new job to see if it reaches syncHandler.
	testJob.Namespace = "bar"
	testJob.Name = "foo"
	fakeWatch.Add(&testJob)
	t.Log("Waiting for job to reach syncHandler")
	<-received
}

func TestWatchPods(t *testing.T) {
	testJob := newJob(2, 2, 6, batch.NonIndexedCompletion)
	clientset := fake.NewSimpleClientset(testJob)
	fakeWatch := watch.NewFake()
	clientset.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady

	// Put one job and one pod into the store
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(testJob)
	received := make(chan struct{})
	// The pod update sent through the fakeWatcher should figure out the managing job and
	// send it into the syncHandler.
	manager.syncHandler = func(ctx context.Context, key string) (bool, error) {
		ns, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			t.Errorf("Error getting namespace/name from key %v: %v", key, err)
		}
		job, err := manager.jobLister.Jobs(ns).Get(name)
		if err != nil {
			t.Errorf("Expected to find job under key %v: %v", key, err)
		}
		if !apiequality.Semantic.DeepDerivative(job, testJob) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testJob, job)
			close(received)
			return true, nil
		}
		close(received)
		return true, nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right job.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go sharedInformerFactory.Core().V1().Pods().Informer().Run(stopCh)
	go manager.Run(context.TODO(), 1)

	pods := newPodList(1, v1.PodRunning, testJob)
	testPod := pods[0]
	testPod.Status.Phase = v1.PodFailed
	fakeWatch.Add(testPod)

	t.Log("Waiting for pod to reach syncHandler")
	<-received
}

func TestWatchOrphanPods(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	sharedInformers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	manager := NewController(sharedInformers.Core().V1().Pods(), sharedInformers.Batch().V1().Jobs(), clientset)
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady

	stopCh := make(chan struct{})
	defer close(stopCh)
	podInformer := sharedInformers.Core().V1().Pods().Informer()
	go podInformer.Run(stopCh)
	cache.WaitForCacheSync(stopCh, podInformer.HasSynced)
	go manager.Run(context.TODO(), 1)

	// Create job but don't add it to the store.
	cases := map[string]struct {
		job     *batch.Job
		inCache bool
	}{
		"job_does_not_exist": {
			job: newJob(2, 2, 6, batch.NonIndexedCompletion),
		},
		"orphan": {},
		"job_finished": {
			job: func() *batch.Job {
				j := newJob(2, 2, 6, batch.NonIndexedCompletion)
				j.Status.Conditions = append(j.Status.Conditions, batch.JobCondition{
					Type:   batch.JobComplete,
					Status: v1.ConditionTrue,
				})
				return j
			}(),
			inCache: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if tc.inCache {
				if err := sharedInformers.Batch().V1().Jobs().Informer().GetIndexer().Add(tc.job); err != nil {
					t.Fatalf("Failed to insert job in index: %v", err)
				}
				t.Cleanup(func() {
					sharedInformers.Batch().V1().Jobs().Informer().GetIndexer().Delete(tc.job)
				})
			}

			podBuilder := buildPod().name(name).deletionTimestamp().trackingFinalizer()
			if tc.job != nil {
				podBuilder = podBuilder.job(tc.job)
			}
			orphanPod := podBuilder.Pod
			orphanPod, err := clientset.CoreV1().Pods("default").Create(context.Background(), orphanPod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Creating orphan pod: %v", err)
			}

			if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
				p, err := clientset.CoreV1().Pods(orphanPod.Namespace).Get(context.Background(), orphanPod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return !hasJobTrackingFinalizer(p), nil
			}); err != nil {
				t.Errorf("Waiting for Pod to get the finalizer removed: %v", err)
			}
		})
	}
}

func bumpResourceVersion(obj metav1.Object) {
	ver, _ := strconv.ParseInt(obj.GetResourceVersion(), 10, 32)
	obj.SetResourceVersion(strconv.FormatInt(ver+1, 10))
}

type pods struct {
	pending int
	active  int
	succeed int
	failed  int
}

func TestJobBackoffReset(t *testing.T) {
	testCases := map[string]struct {
		// job setup
		parallelism  int32
		completions  int32
		backoffLimit int32

		// pod setup - each row is additive!
		pods []pods
	}{
		"parallelism=1": {
			1, 2, 1,
			[]pods{
				{0, 1, 0, 1},
				{0, 0, 1, 0},
			},
		},
		"parallelism=2 (just failure)": {
			2, 2, 1,
			[]pods{
				{0, 2, 0, 1},
				{0, 0, 1, 0},
			},
		},
	}

	for name, tc := range testCases {
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
		DefaultJobBackOff = time.Duration(0) // overwrite the default value for testing
		manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
		fakePodControl := controller.FakePodControl{}
		manager.podControl = &fakePodControl
		manager.podStoreSynced = alwaysReady
		manager.jobStoreSynced = alwaysReady
		var actual *batch.Job
		manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
			actual = job
			return job, nil
		}

		// job & pods setup
		job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
		key := testutil.GetKey(job, t)
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
		podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()

		setPodsStatuses(podIndexer, job, tc.pods[0].pending, tc.pods[0].active, tc.pods[0].succeed, tc.pods[0].failed, 0)
		manager.queue.Add(key)
		manager.processNextWorkItem(context.TODO())
		retries := manager.queue.NumRequeues(key)
		if retries != 1 {
			t.Errorf("%s: expected exactly 1 retry, got %d", name, retries)
		}

		job = actual
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Replace([]interface{}{actual}, actual.ResourceVersion)
		setPodsStatuses(podIndexer, job, tc.pods[1].pending, tc.pods[1].active, tc.pods[1].succeed, tc.pods[1].failed, 0)
		manager.processNextWorkItem(context.TODO())
		retries = manager.queue.NumRequeues(key)
		if retries != 0 {
			t.Errorf("%s: expected exactly 0 retries, got %d", name, retries)
		}
		if getCondition(actual, batch.JobFailed, v1.ConditionTrue, "BackoffLimitExceeded") {
			t.Errorf("%s: unexpected job failure", name)
		}
	}
}

var _ workqueue.RateLimitingInterface = &fakeRateLimitingQueue{}

type fakeRateLimitingQueue struct {
	workqueue.Interface
	requeues int
	item     interface{}
	duration time.Duration
}

func (f *fakeRateLimitingQueue) AddRateLimited(item interface{}) {}
func (f *fakeRateLimitingQueue) Forget(item interface{}) {
	f.requeues = 0
}
func (f *fakeRateLimitingQueue) NumRequeues(item interface{}) int {
	return f.requeues
}
func (f *fakeRateLimitingQueue) AddAfter(item interface{}, duration time.Duration) {
	f.item = item
	f.duration = duration
}

func TestJobBackoff(t *testing.T) {
	job := newJob(1, 1, 1, batch.NonIndexedCompletion)
	oldPod := newPod(fmt.Sprintf("pod-%v", rand.String(10)), job)
	oldPod.Status.Phase = v1.PodRunning
	oldPod.ResourceVersion = "1"
	newPod := oldPod.DeepCopy()
	newPod.ResourceVersion = "2"

	testCases := map[string]struct {
		requeues            int
		phase               v1.PodPhase
		jobReadyPodsEnabled bool
		wantBackoff         time.Duration
	}{
		"1st failure": {
			requeues:    0,
			phase:       v1.PodFailed,
			wantBackoff: 0,
		},
		"2nd failure": {
			requeues:    1,
			phase:       v1.PodFailed,
			wantBackoff: DefaultJobBackOff,
		},
		"3rd failure": {
			requeues:    2,
			phase:       v1.PodFailed,
			wantBackoff: 2 * DefaultJobBackOff,
		},
		"1st success": {
			requeues:    0,
			phase:       v1.PodSucceeded,
			wantBackoff: 0,
		},
		"2nd success": {
			requeues:    1,
			phase:       v1.PodSucceeded,
			wantBackoff: 0,
		},
		"1st running": {
			requeues:    0,
			phase:       v1.PodSucceeded,
			wantBackoff: 0,
		},
		"2nd running": {
			requeues:    1,
			phase:       v1.PodSucceeded,
			wantBackoff: 0,
		},
		"1st failure with pod updates batching": {
			requeues:    0,
			phase:       v1.PodFailed,
			wantBackoff: podUpdateBatchPeriod,
		},
		"2nd failure with pod updates batching": {
			requeues:    1,
			phase:       v1.PodFailed,
			wantBackoff: DefaultJobBackOff,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobReadyPods, tc.jobReadyPodsEnabled)()
			clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			queue := &fakeRateLimitingQueue{}
			manager.queue = queue
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)

			queue.requeues = tc.requeues
			newPod.Status.Phase = tc.phase
			manager.updatePod(oldPod, newPod)

			if queue.duration.Nanoseconds() != int64(tc.wantBackoff)*DefaultJobBackOff.Nanoseconds() {
				t.Errorf("unexpected backoff %v", queue.duration)
			}
		})
	}
}

func TestJobBackoffForOnFailure(t *testing.T) {
	jobConditionFailed := batch.JobFailed

	testCases := map[string]struct {
		// job setup
		parallelism  int32
		completions  int32
		backoffLimit int32

		// pod setup
		jobKeyForget  bool
		restartCounts []int32
		podPhase      v1.PodPhase

		// expectations
		expectedActive          int32
		expectedSucceeded       int32
		expectedFailed          int32
		expectedCondition       *batch.JobConditionType
		expectedConditionReason string
	}{
		"backoffLimit 0 should have 1 pod active": {
			1, 1, 0,
			true, []int32{0}, v1.PodRunning,
			1, 0, 0, nil, "",
		},
		"backoffLimit 1 with restartCount 0 should have 1 pod active": {
			1, 1, 1,
			true, []int32{0}, v1.PodRunning,
			1, 0, 0, nil, "",
		},
		"backoffLimit 1 with restartCount 1 and podRunning should have 0 pod active": {
			1, 1, 1,
			true, []int32{1}, v1.PodRunning,
			0, 0, 1, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"backoffLimit 1 with restartCount 1 and podPending should have 0 pod active": {
			1, 1, 1,
			true, []int32{1}, v1.PodPending,
			0, 0, 1, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"too many job failures with podRunning - single pod": {
			1, 5, 2,
			true, []int32{2}, v1.PodRunning,
			0, 0, 1, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"too many job failures with podPending - single pod": {
			1, 5, 2,
			true, []int32{2}, v1.PodPending,
			0, 0, 1, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"too many job failures with podRunning - multiple pods": {
			2, 5, 2,
			true, []int32{1, 1}, v1.PodRunning,
			0, 0, 2, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"too many job failures with podPending - multiple pods": {
			2, 5, 2,
			true, []int32{1, 1}, v1.PodPending,
			0, 0, 2, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"not enough failures": {
			2, 5, 3,
			true, []int32{1, 1}, v1.PodRunning,
			2, 0, 0, nil, "",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			// job manager setup
			clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			var actual *batch.Job
			manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
				actual = job
				return job, nil
			}

			// job & pods setup
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
			job.Spec.Template.Spec.RestartPolicy = v1.RestartPolicyOnFailure
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			for i, pod := range newPodList(len(tc.restartCounts), tc.podPhase, job) {
				pod.Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: tc.restartCounts[i]}}
				podIndexer.Add(pod)
			}

			// run
			forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))

			if err != nil {
				t.Errorf("unexpected error syncing job.  Got %#v", err)
			}
			if forget != tc.jobKeyForget {
				t.Errorf("unexpected forget value. Expected %v, saw %v\n", tc.jobKeyForget, forget)
			}
			// validate status
			if actual.Status.Active != tc.expectedActive {
				t.Errorf("unexpected number of active pods.  Expected %d, saw %d\n", tc.expectedActive, actual.Status.Active)
			}
			if actual.Status.Succeeded != tc.expectedSucceeded {
				t.Errorf("unexpected number of succeeded pods.  Expected %d, saw %d\n", tc.expectedSucceeded, actual.Status.Succeeded)
			}
			if actual.Status.Failed != tc.expectedFailed {
				t.Errorf("unexpected number of failed pods.  Expected %d, saw %d\n", tc.expectedFailed, actual.Status.Failed)
			}
			// validate conditions
			if tc.expectedCondition != nil && !getCondition(actual, *tc.expectedCondition, v1.ConditionTrue, tc.expectedConditionReason) {
				t.Errorf("expected completion condition.  Got %#v", actual.Status.Conditions)
			}
		})
	}
}

func TestJobBackoffOnRestartPolicyNever(t *testing.T) {
	jobConditionFailed := batch.JobFailed

	testCases := map[string]struct {
		// job setup
		parallelism  int32
		completions  int32
		backoffLimit int32

		// pod setup
		activePodsPhase v1.PodPhase
		activePods      int
		failedPods      int

		// expectations
		isExpectingAnError      bool
		jobKeyForget            bool
		expectedActive          int32
		expectedSucceeded       int32
		expectedFailed          int32
		expectedCondition       *batch.JobConditionType
		expectedConditionReason string
	}{
		"not enough failures with backoffLimit 0 - single pod": {
			1, 1, 0,
			v1.PodRunning, 1, 0,
			false, true, 1, 0, 0, nil, "",
		},
		"not enough failures with backoffLimit 1 - single pod": {
			1, 1, 1,
			"", 0, 1,
			true, false, 1, 0, 1, nil, "",
		},
		"too many failures with backoffLimit 1 - single pod": {
			1, 1, 1,
			"", 0, 2,
			false, true, 0, 0, 2, &jobConditionFailed, "BackoffLimitExceeded",
		},
		"not enough failures with backoffLimit 6 - multiple pods": {
			2, 2, 6,
			v1.PodRunning, 1, 6,
			true, false, 2, 0, 6, nil, "",
		},
		"too many failures with backoffLimit 6 - multiple pods": {
			2, 2, 6,
			"", 0, 7,
			false, true, 0, 0, 7, &jobConditionFailed, "BackoffLimitExceeded",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			// job manager setup
			clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			var actual *batch.Job
			manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
				actual = job
				return job, nil
			}

			// job & pods setup
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
			job.Spec.Template.Spec.RestartPolicy = v1.RestartPolicyNever
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			for _, pod := range newPodList(tc.failedPods, v1.PodFailed, job) {
				podIndexer.Add(pod)
			}
			for _, pod := range newPodList(tc.activePods, tc.activePodsPhase, job) {
				podIndexer.Add(pod)
			}

			// run
			forget, err := manager.syncJob(context.TODO(), testutil.GetKey(job, t))

			if (err != nil) != tc.isExpectingAnError {
				t.Errorf("unexpected error syncing job. Got %#v, isExpectingAnError: %v\n", err, tc.isExpectingAnError)
			}
			if forget != tc.jobKeyForget {
				t.Errorf("unexpected forget value. Expected %v, saw %v\n", tc.jobKeyForget, forget)
			}
			// validate status
			if actual.Status.Active != tc.expectedActive {
				t.Errorf("unexpected number of active pods. Expected %d, saw %d\n", tc.expectedActive, actual.Status.Active)
			}
			if actual.Status.Succeeded != tc.expectedSucceeded {
				t.Errorf("unexpected number of succeeded pods. Expected %d, saw %d\n", tc.expectedSucceeded, actual.Status.Succeeded)
			}
			if actual.Status.Failed != tc.expectedFailed {
				t.Errorf("unexpected number of failed pods. Expected %d, saw %d\n", tc.expectedFailed, actual.Status.Failed)
			}
			// validate conditions
			if tc.expectedCondition != nil && !getCondition(actual, *tc.expectedCondition, v1.ConditionTrue, tc.expectedConditionReason) {
				t.Errorf("expected completion condition. Got %#v", actual.Status.Conditions)
			}
		})
	}
}

func TestEnsureJobConditions(t *testing.T) {
	testCases := []struct {
		name         string
		haveList     []batch.JobCondition
		wantType     batch.JobConditionType
		wantStatus   v1.ConditionStatus
		wantReason   string
		expectList   []batch.JobCondition
		expectUpdate bool
	}{
		{
			name:         "append true condition",
			haveList:     []batch.JobCondition{},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			expectUpdate: true,
		},
		{
			name:         "append false condition",
			haveList:     []batch.JobCondition{},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionFalse,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{},
			expectUpdate: false,
		},
		{
			name:         "update true condition reason",
			haveList:     []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "bar",
			expectList:   []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "bar", "")},
			expectUpdate: true,
		},
		{
			name:         "update true condition status",
			haveList:     []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionFalse,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionFalse, "foo", "")},
			expectUpdate: true,
		},
		{
			name:         "update false condition status",
			haveList:     []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionFalse, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			expectUpdate: true,
		},
		{
			name:         "condition already exists",
			haveList:     []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{*newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			expectUpdate: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gotList, isUpdated := ensureJobConditionStatus(tc.haveList, tc.wantType, tc.wantStatus, tc.wantReason, "")
			if isUpdated != tc.expectUpdate {
				t.Errorf("Got isUpdated=%v, want %v", isUpdated, tc.expectUpdate)
			}
			if len(gotList) != len(tc.expectList) {
				t.Errorf("got a list of length %d, want %d", len(gotList), len(tc.expectList))
			}
			if diff := cmp.Diff(tc.expectList, gotList, cmpopts.IgnoreFields(batch.JobCondition{}, "LastProbeTime", "LastTransitionTime")); diff != "" {
				t.Errorf("Unexpected JobCondition list: (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestFinalizersRemovedExpectations(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.JobTrackingWithFinalizers, true)()
	clientset := fake.NewSimpleClientset()
	sharedInformers := informers.NewSharedInformerFactory(clientset, controller.NoResyncPeriodFunc())
	manager := NewController(sharedInformers.Core().V1().Pods(), sharedInformers.Batch().V1().Jobs(), clientset)
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	manager.podControl = &controller.FakePodControl{Err: errors.New("fake pod controller error")}
	manager.updateStatusHandler = func(ctx context.Context, job *batch.Job) (*batch.Job, error) {
		return job, nil
	}

	job := newJob(2, 2, 6, batch.NonIndexedCompletion)
	job.Annotations = map[string]string{
		batch.JobTrackingFinalizer: "",
	}
	sharedInformers.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	pods := append(newPodList(2, v1.PodSucceeded, job), newPodList(2, v1.PodFailed, job)...)
	podInformer := sharedInformers.Core().V1().Pods().Informer()
	podIndexer := podInformer.GetIndexer()
	uids := sets.NewString()
	for i := range pods {
		clientset.Tracker().Add(pods[i])
		podIndexer.Add(pods[i])
		uids.Insert(string(pods[i].UID))
	}
	jobKey := testutil.GetKey(job, t)

	manager.syncJob(context.TODO(), jobKey)
	gotExpectedUIDs := manager.finalizerExpectations.getExpectedUIDs(jobKey)
	if len(gotExpectedUIDs) != 0 {
		t.Errorf("Got unwanted expectations for removed finalizers after first syncJob with client failures:\n%s", gotExpectedUIDs.List())
	}

	// Remove failures and re-sync.
	manager.podControl.(*controller.FakePodControl).Err = nil
	manager.syncJob(context.TODO(), jobKey)
	gotExpectedUIDs = manager.finalizerExpectations.getExpectedUIDs(jobKey)
	if diff := cmp.Diff(uids, gotExpectedUIDs); diff != "" {
		t.Errorf("Different expectations for removed finalizers after syncJob (-want,+got):\n%s", diff)
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	go sharedInformers.Core().V1().Pods().Informer().Run(stopCh)
	cache.WaitForCacheSync(stopCh, podInformer.HasSynced)

	// Make sure the first syncJob sets the expectations, even after the caches synced.
	gotExpectedUIDs = manager.finalizerExpectations.getExpectedUIDs(jobKey)
	if diff := cmp.Diff(uids, gotExpectedUIDs); diff != "" {
		t.Errorf("Different expectations for removed finalizers after syncJob and cacheSync (-want,+got):\n%s", diff)
	}

	// Change pods in different ways.

	podsResource := schema.GroupVersionResource{Version: "v1", Resource: "pods"}

	update := pods[0].DeepCopy()
	update.Finalizers = nil
	update.ResourceVersion = "1"
	err := clientset.Tracker().Update(podsResource, update, update.Namespace)
	if err != nil {
		t.Errorf("Removing finalizer: %v", err)
	}

	update = pods[1].DeepCopy()
	update.Finalizers = nil
	update.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	update.ResourceVersion = "1"
	err = clientset.Tracker().Update(podsResource, update, update.Namespace)
	if err != nil {
		t.Errorf("Removing finalizer and setting deletion timestamp: %v", err)
	}

	// Preserve the finalizer.
	update = pods[2].DeepCopy()
	update.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	update.ResourceVersion = "1"
	err = clientset.Tracker().Update(podsResource, update, update.Namespace)
	if err != nil {
		t.Errorf("Setting deletion timestamp: %v", err)
	}

	err = clientset.Tracker().Delete(podsResource, pods[3].Namespace, pods[3].Name)
	if err != nil {
		t.Errorf("Deleting pod that had finalizer: %v", err)
	}

	uids = sets.NewString(string(pods[2].UID))
	var diff string
	if err := wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		gotExpectedUIDs = manager.finalizerExpectations.getExpectedUIDs(jobKey)
		diff = cmp.Diff(uids, gotExpectedUIDs)
		return diff == "", nil
	}); err != nil {
		t.Errorf("Timeout waiting for expectations (-want, +got):\n%s", diff)
	}
}

func checkJobCompletionEnvVariable(t *testing.T, spec *v1.PodSpec) {
	t.Helper()
	want := []v1.EnvVar{
		{
			Name: "JOB_COMPLETION_INDEX",
			ValueFrom: &v1.EnvVarSource{
				FieldRef: &v1.ObjectFieldSelector{
					FieldPath: fmt.Sprintf("metadata.annotations['%s']", batch.JobCompletionIndexAnnotation),
				},
			},
		},
	}
	for _, c := range spec.InitContainers {
		if diff := cmp.Diff(want, c.Env); diff != "" {
			t.Errorf("Unexpected Env in container %s (-want,+got):\n%s", c.Name, diff)
		}
	}
	for _, c := range spec.Containers {
		if diff := cmp.Diff(want, c.Env); diff != "" {
			t.Errorf("Unexpected Env in container %s (-want,+got):\n%s", c.Name, diff)
		}
	}
}

// hasValidFailingPods checks if there exists failed pods with valid index.
func hasValidFailingPods(status []indexPhase, completions int) bool {
	for _, s := range status {
		ix, err := strconv.Atoi(s.Index)
		if err != nil {
			continue
		}
		if ix < 0 || ix >= completions {
			continue
		}
		if s.Phase == v1.PodFailed {
			return true
		}
	}
	return false
}

type podBuilder struct {
	*v1.Pod
}

func buildPod() podBuilder {
	return podBuilder{Pod: &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(rand.String(5)),
		},
	}}
}

func getConditionsByType(list []batch.JobCondition, cType batch.JobConditionType) []*batch.JobCondition {
	var result []*batch.JobCondition
	for i := range list {
		if list[i].Type == cType {
			result = append(result, &list[i])
		}
	}
	return result
}

func (pb podBuilder) name(n string) podBuilder {
	pb.Name = n
	return pb
}

func (pb podBuilder) ns(n string) podBuilder {
	pb.Namespace = n
	return pb
}

func (pb podBuilder) uid(u string) podBuilder {
	pb.UID = types.UID(u)
	return pb
}

func (pb podBuilder) job(j *batch.Job) podBuilder {
	pb.Labels = j.Spec.Selector.MatchLabels
	pb.Namespace = j.Namespace
	pb.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(j, controllerKind)}
	return pb
}

func (pb podBuilder) clearOwner() podBuilder {
	pb.OwnerReferences = nil
	return pb
}

func (pb podBuilder) clearLabels() podBuilder {
	pb.Labels = nil
	return pb
}

func (pb podBuilder) index(ix string) podBuilder {
	if pb.Annotations == nil {
		pb.Annotations = make(map[string]string)
	}
	pb.Annotations[batch.JobCompletionIndexAnnotation] = ix
	return pb
}

func (pb podBuilder) phase(p v1.PodPhase) podBuilder {
	pb.Status.Phase = p
	return pb
}

func (pb podBuilder) trackingFinalizer() podBuilder {
	for _, f := range pb.Finalizers {
		if f == batch.JobTrackingFinalizer {
			return pb
		}
	}
	pb.Finalizers = append(pb.Finalizers, batch.JobTrackingFinalizer)
	return pb
}

func (pb podBuilder) deletionTimestamp() podBuilder {
	pb.DeletionTimestamp = &metav1.Time{}
	return pb
}
