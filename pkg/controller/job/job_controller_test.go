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
	"fmt"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

var alwaysReady = func() bool { return true }

func newJob(parallelism, completions, backoffLimit int32, completionMode batch.CompletionMode) *batch.Job {
	j := &batch.Job{
		TypeMeta: metav1.TypeMeta{Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foobar",
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
			Labels:          job.Spec.Selector.MatchLabels,
			Namespace:       job.Namespace,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(job, controllerKind)},
		},
	}
}

// create count pods with the given phase for the given job
func newPodList(count int32, status v1.PodPhase, job *batch.Job) []v1.Pod {
	pods := []v1.Pod{}
	for i := int32(0); i < count; i++ {
		newPod := newPod(fmt.Sprintf("pod-%v", rand.String(10)), job)
		newPod.Status = v1.PodStatus{Phase: status}
		pods = append(pods, *newPod)
	}
	return pods
}

func setPodsStatuses(podIndexer cache.Indexer, job *batch.Job, pendingPods, activePods, succeededPods, failedPods int32) {
	for _, pod := range newPodList(pendingPods, v1.PodPending, job) {
		podIndexer.Add(&pod)
	}
	for _, pod := range newPodList(activePods, v1.PodRunning, job) {
		podIndexer.Add(&pod)
	}
	for _, pod := range newPodList(succeededPods, v1.PodSucceeded, job) {
		podIndexer.Add(&pod)
	}
	for _, pod := range newPodList(failedPods, v1.PodFailed, job) {
		podIndexer.Add(&pod)
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

		// pod setup
		podControllerError        error
		jobKeyForget              bool
		pendingPods               int32
		activePods                int32
		succeededPods             int32
		failedPods                int32
		podsWithIndexes           []indexPhase
		fakeExpectationAtCreation int32 // negative: ExpectDeletions, positive: ExpectCreations

		// expectations
		expectedCreations       int32
		expectedDeletions       int32
		expectedActive          int32
		expectedSucceeded       int32
		expectedCompletedIdxs   string
		expectedFailed          int32
		expectedCondition       *batch.JobConditionType
		expectedConditionStatus v1.ConditionStatus
		expectedConditionReason string
		expectedCreatedIndexes  sets.Int

		// features
		indexedJobEnabled bool
		suspendJobEnabled bool
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
			parallelism:    2,
			completions:    5,
			backoffLimit:   6,
			jobKeyForget:   true,
			activePods:     2,
			expectedActive: 2,
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
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        1,
			succeededPods:     1,
			expectedCreations: 1,
			expectedActive:    2,
			expectedSucceeded: 1,
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
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        3,
			expectedDeletions: 1,
			expectedActive:    2,
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
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			activePods:        1,
			failedPods:        1,
			expectedCreations: 1,
			expectedActive:    2,
			expectedFailed:    1,
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
		},
		"WQ job finishing": {
			parallelism:       2,
			completions:       -1,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        1,
			succeededPods:     1,
			expectedActive:    1,
			expectedSucceeded: 1,
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
		},
		"more active pods than parallelism": {
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        10,
			expectedDeletions: 8,
			expectedActive:    2,
		},
		"more active pods than remaining completions": {
			parallelism:       3,
			completions:       4,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        3,
			succeededPods:     2,
			expectedDeletions: 1,
			expectedActive:    2,
			expectedSucceeded: 2,
		},
		"status change": {
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			jobKeyForget:      true,
			activePods:        2,
			succeededPods:     2,
			expectedActive:    2,
			expectedSucceeded: 2,
		},
		"deleting job": {
			parallelism:       2,
			completions:       5,
			backoffLimit:      6,
			deleting:          true,
			jobKeyForget:      true,
			pendingPods:       1,
			activePods:        1,
			succeededPods:     1,
			expectedActive:    2,
			expectedSucceeded: 1,
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
			indexedJobEnabled:      true,
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
			indexedJobEnabled:       true,
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
			indexedJobEnabled:      true,
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
			indexedJobEnabled:      true,
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
			indexedJobEnabled:      true,
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
			indexedJobEnabled:     true,
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
			indexedJobEnabled:     true,
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
			indexedJobEnabled:     true,
		},
		"indexed job feature disabled": {
			parallelism:    2,
			completions:    3,
			backoffLimit:   6,
			completionMode: batch.IndexedCompletion,
			podsWithIndexes: []indexPhase{
				{"0", v1.PodRunning},
				{"1", v1.PodSucceeded},
			},
			// No status updates.
			indexedJobEnabled: false,
		},
		"suspending a job with satisfied expectations": {
			// Suspended Job should delete active pods when expectations are
			// satisfied.
			suspendJobEnabled:       true,
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
		},
		"suspending a job with unsatisfied expectations": {
			// Unlike the previous test, we expect the controller to NOT suspend the
			// Job in the syncJob call because the controller will wait for
			// expectations to be satisfied first. The next syncJob call (not tested
			// here) will be the same as the previous test.
			suspendJobEnabled:         true,
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
			suspendJobEnabled:       true,
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
			suspendJobEnabled: true,
			suspend:           true,
			deleting:          true,
			parallelism:       2,
			activePods:        2, // parallelism == active, expectations satisfied
			completions:       4,
			backoffLimit:      6,
			jobKeyForget:      true,
			expectedCreations: 0,
			expectedDeletions: 0,
			expectedActive:    2,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.IndexedJob, tc.indexedJobEnabled)()
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.SuspendJob, tc.suspendJobEnabled)()

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
				job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobSuspended, v1.ConditionTrue, "JobSuspended", "Job suspended"))
			}
			if tc.deleting {
				now := metav1.Now()
				job.DeletionTimestamp = &now
			}
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			setPodsStatuses(podIndexer, job, tc.pendingPods, tc.activePods, tc.succeededPods, tc.failedPods)
			setPodsStatusesWithIndexes(podIndexer, job, tc.podsWithIndexes)

			actual := job
			manager.updateHandler = func(job *batch.Job) error {
				actual = job
				return nil
			}

			// run
			forget, err := manager.syncJob(testutil.GetKey(job, t))

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
			if actual.Status.StartTime == nil && tc.indexedJobEnabled && !tc.suspend {
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
		})
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
		if diff := cmp.Equal(expectedName, p.Spec.Hostname); !diff {
			t.Errorf("Got pod hostname %s, want %s", p.Spec.Hostname, expectedName)
		}
	}
	if diff := cmp.Diff(wantIndexes.List(), gotIndexes.List()); diff != "" {
		t.Errorf("Unexpected created completion indexes (-want,+got):\n%s", diff)
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
		activePods    int32
		succeededPods int32
		failedPods    int32

		// expectations
		expectedForGetKey       bool
		expectedDeletions       int32
		expectedActive          int32
		expectedSucceeded       int32
		expectedFailed          int32
		expectedCondition       batch.JobConditionType
		expectedConditionReason string

		// features
		suspendJobEnabled bool
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
			suspendJobEnabled:       true,
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
			defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.SuspendJob, tc.suspendJobEnabled)()

			// job manager setup
			clientSet := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
			manager, sharedInformerFactory := newControllerFromClient(clientSet, controller.NoResyncPeriodFunc)
			fakePodControl := controller.FakePodControl{}
			manager.podControl = &fakePodControl
			manager.podStoreSynced = alwaysReady
			manager.jobStoreSynced = alwaysReady
			var actual *batch.Job
			manager.updateHandler = func(job *batch.Job) error {
				actual = job
				return nil
			}

			// job & pods setup
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
			job.Spec.ActiveDeadlineSeconds = &tc.activeDeadlineSeconds
			job.Spec.Suspend = pointer.BoolPtr(tc.suspend)
			start := metav1.Unix(metav1.Now().Time.Unix()-tc.startTime, 0)
			job.Status.StartTime = &start
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			setPodsStatuses(podIndexer, job, 0, tc.activePods, tc.succeededPods, tc.failedPods)

			// run
			forget, err := manager.syncJob(testutil.GetKey(job, t))
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
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	var actual *batch.Job
	manager.updateHandler = func(job *batch.Job) error {
		actual = job
		return nil
	}

	job := newJob(1, 1, 6, batch.NonIndexedCompletion)
	activeDeadlineSeconds := int64(10)
	job.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	start := metav1.Unix(metav1.Now().Time.Unix()-15, 0)
	job.Status.StartTime = &start
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobFailed, v1.ConditionTrue, "DeadlineExceeded", "Job was active longer than specified deadline"))
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	forget, err := manager.syncJob(testutil.GetKey(job, t))
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
	if actual != nil {
		t.Error("Unexpected job modification")
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
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobComplete, v1.ConditionTrue, "", ""))
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	forget, err := manager.syncJob(testutil.GetKey(job, t))
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
	manager.updateHandler = func(job *batch.Job) error { return nil }
	job := newJob(2, 2, 6, batch.NonIndexedCompletion)
	forget, err := manager.syncJob(testutil.GetKey(job, t))
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
	manager, sharedInformerFactory := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	updateError := fmt.Errorf("update error")
	manager.updateHandler = func(job *batch.Job) error {
		manager.queue.AddRateLimited(testutil.GetKey(job, t))
		return updateError
	}
	job := newJob(2, 2, 6, batch.NonIndexedCompletion)
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	forget, err := manager.syncJob(testutil.GetKey(job, t))
	if err == nil || err != updateError {
		t.Errorf("Expected error %v when syncing jobs, got %v", updateError, err)
	}
	if forget != false {
		t.Errorf("Unexpected forget value. Expected %v, saw %v\n", false, forget)
	}
	t.Log("Waiting for a job in the queue")
	key, _ := manager.queue.Get()
	expectedKey := testutil.GetKey(job, t)
	if key != expectedKey {
		t.Errorf("Expected requeue of job with key %s got %s", expectedKey, key)
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
	}{
		"only matching": {
			pods: []*v1.Pod{
				newPod("pod1", job),
				newPod("pod2", otherJob),
				{ObjectMeta: metav1.ObjectMeta{Name: "pod3", Namespace: job.Namespace}},
				newPod("pod4", job),
			},
			wantPods: []string{"pod1", "pod4"},
		},
		"adopt": {
			pods: []*v1.Pod{
				newPod("pod1", job),
				func() *v1.Pod {
					p := newPod("pod2", job)
					p.OwnerReferences = nil
					return p
				}(),
				newPod("pod3", otherJob),
			},
			wantPods: []string{"pod1", "pod2"},
		},
		"no adopt when deleting": {
			jobDeleted:        true,
			jobDeletedInCache: true,
			pods: []*v1.Pod{
				newPod("pod1", job),
				func() *v1.Pod {
					p := newPod("pod2", job)
					p.OwnerReferences = nil
					return p
				}(),
			},
			wantPods: []string{"pod1"},
		},
		"no adopt when deleting race": {
			jobDeleted: true,
			pods: []*v1.Pod{
				newPod("pod1", job),
				func() *v1.Pod {
					p := newPod("pod2", job)
					p.OwnerReferences = nil
					return p
				}(),
			},
			wantPods: []string{"pod1"},
		},
		"release": {
			pods: []*v1.Pod{
				newPod("pod1", job),
				func() *v1.Pod {
					p := newPod("pod2", job)
					p.Labels = nil
					return p
				}(),
			},
			wantPods: []string{"pod1"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
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

			pods, err := jm.getPodsForJob(job)
			if err != nil {
				t.Fatalf("getPodsForJob() error: %v", err)
			}
			got := make([]string, len(pods))
			for i, p := range pods {
				got[i] = p.Name
			}
			sort.Strings(got)
			if diff := cmp.Diff(tc.wantPods, got); diff != "" {
				t.Errorf("getPodsForJob() returned (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestAddPod(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

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

	jm.deletePod(pod1)
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

	jm.deletePod(pod2)
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

	jm.deletePod(pod1)
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
	manager.updateHandler = func(job *batch.Job) error { return nil }

	job := newJob(2, 2, 6, batch.NonIndexedCompletion)
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	pods := newPodList(2, v1.PodPending, job)
	podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
	podIndexer.Add(&pods[0])

	manager.expectations = FakeJobExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectations, the job
			// will create a new replica because it doesn't see this pod, but
			// has fulfilled its expectations.
			podIndexer.Add(&pods[1])
		},
	}
	manager.syncJob(testutil.GetKey(job, t))
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
	manager.syncHandler = func(key string) (bool, error) {
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
	go manager.Run(1, stopCh)

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
	manager.syncHandler = func(key string) (bool, error) {
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
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	pods := newPodList(1, v1.PodRunning, testJob)
	testPod := pods[0]
	testPod.Status.Phase = v1.PodFailed
	fakeWatch.Add(&testPod)

	t.Log("Waiting for pod to reach syncHandler")
	<-received
}

func bumpResourceVersion(obj metav1.Object) {
	ver, _ := strconv.ParseInt(obj.GetResourceVersion(), 10, 32)
	obj.SetResourceVersion(strconv.FormatInt(ver+1, 10))
}

type pods struct {
	pending int32
	active  int32
	succeed int32
	failed  int32
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
		manager.updateHandler = func(job *batch.Job) error {
			actual = job
			return nil
		}

		// job & pods setup
		job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
		key := testutil.GetKey(job, t)
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
		podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()

		setPodsStatuses(podIndexer, job, tc.pods[0].pending, tc.pods[0].active, tc.pods[0].succeed, tc.pods[0].failed)
		manager.queue.Add(key)
		manager.processNextWorkItem()
		retries := manager.queue.NumRequeues(key)
		if retries != 1 {
			t.Errorf("%s: expected exactly 1 retry, got %d", name, retries)
		}

		job = actual
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Replace([]interface{}{actual}, actual.ResourceVersion)
		setPodsStatuses(podIndexer, job, tc.pods[1].pending, tc.pods[1].active, tc.pods[1].succeed, tc.pods[1].failed)
		manager.processNextWorkItem()
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
		// inputs
		requeues int
		phase    v1.PodPhase

		// expectation
		backoff int
	}{
		"1st failure": {0, v1.PodFailed, 0},
		"2nd failure": {1, v1.PodFailed, 1},
		"3rd failure": {2, v1.PodFailed, 2},
		"1st success": {0, v1.PodSucceeded, 0},
		"2nd success": {1, v1.PodSucceeded, 0},
		"1st running": {0, v1.PodSucceeded, 0},
		"2nd running": {1, v1.PodSucceeded, 0},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
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

			if queue.duration.Nanoseconds() != int64(tc.backoff)*DefaultJobBackOff.Nanoseconds() {
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
			manager.updateHandler = func(job *batch.Job) error {
				actual = job
				return nil
			}

			// job & pods setup
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
			job.Spec.Template.Spec.RestartPolicy = v1.RestartPolicyOnFailure
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			for i, pod := range newPodList(int32(len(tc.restartCounts)), tc.podPhase, job) {
				pod.Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: tc.restartCounts[i]}}
				podIndexer.Add(&pod)
			}

			// run
			forget, err := manager.syncJob(testutil.GetKey(job, t))

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
		activePods      int32
		failedPods      int32

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
			manager.updateHandler = func(job *batch.Job) error {
				actual = job
				return nil
			}

			// job & pods setup
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit, batch.NonIndexedCompletion)
			job.Spec.Template.Spec.RestartPolicy = v1.RestartPolicyNever
			sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
			podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
			for _, pod := range newPodList(tc.failedPods, v1.PodFailed, job) {
				podIndexer.Add(&pod)
			}
			for _, pod := range newPodList(tc.activePods, tc.activePodsPhase, job) {
				podIndexer.Add(&pod)
			}

			// run
			forget, err := manager.syncJob(testutil.GetKey(job, t))

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
			expectList:   []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
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
			haveList:     []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "bar",
			expectList:   []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "bar", "")},
			expectUpdate: true,
		},
		{
			name:         "update true condition status",
			haveList:     []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionFalse,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionFalse, "foo", "")},
			expectUpdate: true,
		},
		{
			name:         "update false condition status",
			haveList:     []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionFalse, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			expectUpdate: true,
		},
		{
			name:         "condition already exists",
			haveList:     []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
			wantType:     batch.JobSuspended,
			wantStatus:   v1.ConditionTrue,
			wantReason:   "foo",
			expectList:   []batch.JobCondition{newCondition(batch.JobSuspended, v1.ConditionTrue, "foo", "")},
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
			for i := range gotList {
				// Make timestamps the same before comparing the two lists.
				gotList[i].LastProbeTime = tc.expectList[i].LastProbeTime
				gotList[i].LastTransitionTime = tc.expectList[i].LastTransitionTime
			}
			if diff := cmp.Diff(tc.expectList, gotList); diff != "" {
				t.Errorf("Unexpected JobCondition list: (-want,+got):\n%s", diff)
			}
		})
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
