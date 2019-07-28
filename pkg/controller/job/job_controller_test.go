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
	"strconv"
	"testing"
	"time"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	restclient "k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/testutil"
)

var alwaysReady = func() bool { return true }

func newJob(parallelism, completions, backoffLimit int32) *batch.Job {
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

func newJobControllerFromClient(kubeClient clientset.Interface, resyncPeriod controller.ResyncPeriodFunc) (*JobController, informers.SharedInformerFactory) {
	sharedInformers := informers.NewSharedInformerFactory(kubeClient, resyncPeriod())
	jm := NewJobController(sharedInformers.Core().V1().Pods(), sharedInformers.Batch().V1().Jobs(), kubeClient)
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

func TestControllerSyncJob(t *testing.T) {
	jobConditionComplete := batch.JobComplete
	jobConditionFailed := batch.JobFailed

	testCases := map[string]struct {
		// job setup
		parallelism  int32
		completions  int32
		backoffLimit int32
		deleting     bool
		podLimit     int

		// pod setup
		podControllerError error
		jobKeyForget       bool
		pendingPods        int32
		activePods         int32
		succeededPods      int32
		failedPods         int32

		// expectations
		expectedCreations       int32
		expectedDeletions       int32
		expectedActive          int32
		expectedSucceeded       int32
		expectedFailed          int32
		expectedCondition       *batch.JobConditionType
		expectedConditionReason string
	}{
		"job start": {
			2, 5, 6, false, 0,
			nil, true, 0, 0, 0, 0,
			2, 0, 2, 0, 0, nil, "",
		},
		"WQ job start": {
			2, -1, 6, false, 0,
			nil, true, 0, 0, 0, 0,
			2, 0, 2, 0, 0, nil, "",
		},
		"pending pods": {
			2, 5, 6, false, 0,
			nil, true, 2, 0, 0, 0,
			0, 0, 2, 0, 0, nil, "",
		},
		"correct # of pods": {
			2, 5, 6, false, 0,
			nil, true, 0, 2, 0, 0,
			0, 0, 2, 0, 0, nil, "",
		},
		"WQ job: correct # of pods": {
			2, -1, 6, false, 0,
			nil, true, 0, 2, 0, 0,
			0, 0, 2, 0, 0, nil, "",
		},
		"too few active pods": {
			2, 5, 6, false, 0,
			nil, true, 0, 1, 1, 0,
			1, 0, 2, 1, 0, nil, "",
		},
		"too few active pods with a dynamic job": {
			2, -1, 6, false, 0,
			nil, true, 0, 1, 0, 0,
			1, 0, 2, 0, 0, nil, "",
		},
		"too few active pods, with controller error": {
			2, 5, 6, false, 0,
			fmt.Errorf("fake error"), true, 0, 1, 1, 0,
			1, 0, 1, 1, 0, nil, "",
		},
		"too many active pods": {
			2, 5, 6, false, 0,
			nil, true, 0, 3, 0, 0,
			0, 1, 2, 0, 0, nil, "",
		},
		"too many active pods, with controller error": {
			2, 5, 6, false, 0,
			fmt.Errorf("fake error"), true, 0, 3, 0, 0,
			0, 1, 3, 0, 0, nil, "",
		},
		"failed + succeed pods: reset backoff delay": {
			2, 5, 6, false, 0,
			fmt.Errorf("fake error"), true, 0, 1, 1, 1,
			1, 0, 1, 1, 1, nil, "",
		},
		"only new failed pod": {
			2, 5, 6, false, 0,
			fmt.Errorf("fake error"), false, 0, 1, 0, 1,
			1, 0, 1, 0, 1, nil, "",
		},
		"job finish": {
			2, 5, 6, false, 0,
			nil, true, 0, 0, 5, 0,
			0, 0, 0, 5, 0, nil, "",
		},
		"WQ job finishing": {
			2, -1, 6, false, 0,
			nil, true, 0, 1, 1, 0,
			0, 0, 1, 1, 0, nil, "",
		},
		"WQ job all finished": {
			2, -1, 6, false, 0,
			nil, true, 0, 0, 2, 0,
			0, 0, 0, 2, 0, &jobConditionComplete, "",
		},
		"WQ job all finished despite one failure": {
			2, -1, 6, false, 0,
			nil, true, 0, 0, 1, 1,
			0, 0, 0, 1, 1, &jobConditionComplete, "",
		},
		"more active pods than completions": {
			2, 5, 6, false, 0,
			nil, true, 0, 10, 0, 0,
			0, 8, 2, 0, 0, nil, "",
		},
		"status change": {
			2, 5, 6, false, 0,
			nil, true, 0, 2, 2, 0,
			0, 0, 2, 2, 0, nil, "",
		},
		"deleting job": {
			2, 5, 6, true, 0,
			nil, true, 1, 1, 1, 0,
			0, 0, 2, 1, 0, nil, "",
		},
		"limited pods": {
			100, 200, 6, false, 10,
			nil, true, 0, 0, 0, 0,
			10, 0, 10, 0, 0, nil, "",
		},
		"too many job failures": {
			2, 5, 0, true, 0,
			nil, true, 0, 0, 0, 1,
			0, 0, 0, 0, 1, &jobConditionFailed, "BackoffLimitExceeded",
		},
	}

	for name, tc := range testCases {
		// job manager setup
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
		manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
		fakePodControl := controller.FakePodControl{Err: tc.podControllerError, CreateLimit: tc.podLimit}
		manager.podControl = &fakePodControl
		manager.podStoreSynced = alwaysReady
		manager.jobStoreSynced = alwaysReady
		var actual *batch.Job
		manager.updateHandler = func(job *batch.Job) error {
			actual = job
			return nil
		}

		// job & pods setup
		job := newJob(tc.parallelism, tc.completions, tc.backoffLimit)
		if tc.deleting {
			now := metav1.Now()
			job.DeletionTimestamp = &now
		}
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
		podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
		setPodsStatuses(podIndexer, job, tc.pendingPods, tc.activePods, tc.succeededPods, tc.failedPods)

		// run
		forget, err := manager.syncJob(testutil.GetKey(job, t))

		// We need requeue syncJob task if podController error
		if tc.podControllerError != nil {
			if err == nil {
				t.Errorf("%s: Syncing jobs would return error when podController exception", name)
			}
		} else {
			if err != nil && (tc.podLimit == 0 || fakePodControl.CreateCallCount < tc.podLimit) {
				t.Errorf("%s: unexpected error when syncing jobs %v", name, err)
			}
		}
		if forget != tc.jobKeyForget {
			t.Errorf("%s: unexpected forget value. Expected %v, saw %v\n", name, tc.jobKeyForget, forget)
		}
		// validate created/deleted pods
		if int32(len(fakePodControl.Templates)) != tc.expectedCreations {
			t.Errorf("%s: unexpected number of creates.  Expected %d, saw %d\n", name, tc.expectedCreations, len(fakePodControl.Templates))
		}
		if int32(len(fakePodControl.DeletePodName)) != tc.expectedDeletions {
			t.Errorf("%s: unexpected number of deletes.  Expected %d, saw %d\n", name, tc.expectedDeletions, len(fakePodControl.DeletePodName))
		}
		// Each create should have an accompanying ControllerRef.
		if len(fakePodControl.ControllerRefs) != int(tc.expectedCreations) {
			t.Errorf("%s: unexpected number of ControllerRefs.  Expected %d, saw %d\n", name, tc.expectedCreations, len(fakePodControl.ControllerRefs))
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
			t.Errorf("%s: unexpected number of active pods.  Expected %d, saw %d\n", name, tc.expectedActive, actual.Status.Active)
		}
		if actual.Status.Succeeded != tc.expectedSucceeded {
			t.Errorf("%s: unexpected number of succeeded pods.  Expected %d, saw %d\n", name, tc.expectedSucceeded, actual.Status.Succeeded)
		}
		if actual.Status.Failed != tc.expectedFailed {
			t.Errorf("%s: unexpected number of failed pods.  Expected %d, saw %d\n", name, tc.expectedFailed, actual.Status.Failed)
		}
		if actual.Status.StartTime == nil {
			t.Errorf("%s: .status.startTime was not set", name)
		}
		// validate conditions
		if tc.expectedCondition != nil && !getCondition(actual, *tc.expectedCondition, tc.expectedConditionReason) {
			t.Errorf("%s: expected completion condition.  Got %#v", name, actual.Status.Conditions)
		}
		// validate slow start
		expectedLimit := 0
		for pass := uint8(0); expectedLimit <= tc.podLimit; pass++ {
			expectedLimit += controller.SlowStartInitialBatchSize << pass
		}
		if tc.podLimit > 0 && fakePodControl.CreateCallCount > expectedLimit {
			t.Errorf("%s: Unexpected number of create calls.  Expected <= %d, saw %d\n", name, fakePodControl.CreateLimit*2, fakePodControl.CreateCallCount)
		}
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
		expectedConditionReason string
	}{
		"activeDeadlineSeconds less than single pod execution": {
			1, 1, 10, 15, 6,
			1, 0, 0,
			true, 1, 0, 0, 1, "DeadlineExceeded",
		},
		"activeDeadlineSeconds bigger than single pod execution": {
			1, 2, 10, 15, 6,
			1, 1, 0,
			true, 1, 0, 1, 1, "DeadlineExceeded",
		},
		"activeDeadlineSeconds times-out before any pod starts": {
			1, 1, 10, 10, 6,
			0, 0, 0,
			true, 0, 0, 0, 0, "DeadlineExceeded",
		},
		"activeDeadlineSeconds with backofflimit reach": {
			1, 1, 1, 10, 0,
			0, 0, 1,
			true, 0, 0, 0, 1, "BackoffLimitExceeded",
		},
	}

	for name, tc := range testCases {
		// job manager setup
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
		manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
		job := newJob(tc.parallelism, tc.completions, tc.backoffLimit)
		job.Spec.ActiveDeadlineSeconds = &tc.activeDeadlineSeconds
		start := metav1.Unix(metav1.Now().Time.Unix()-tc.startTime, 0)
		job.Status.StartTime = &start
		sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
		podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
		setPodsStatuses(podIndexer, job, 0, tc.activePods, tc.succeededPods, tc.failedPods)

		// run
		forget, err := manager.syncJob(testutil.GetKey(job, t))
		if err != nil {
			t.Errorf("%s: unexpected error when syncing jobs %v", name, err)
		}
		if forget != tc.expectedForGetKey {
			t.Errorf("%s: unexpected forget value. Expected %v, saw %v\n", name, tc.expectedForGetKey, forget)
		}
		// validate created/deleted pods
		if int32(len(fakePodControl.Templates)) != 0 {
			t.Errorf("%s: unexpected number of creates.  Expected 0, saw %d\n", name, len(fakePodControl.Templates))
		}
		if int32(len(fakePodControl.DeletePodName)) != tc.expectedDeletions {
			t.Errorf("%s: unexpected number of deletes.  Expected %d, saw %d\n", name, tc.expectedDeletions, len(fakePodControl.DeletePodName))
		}
		// validate status
		if actual.Status.Active != tc.expectedActive {
			t.Errorf("%s: unexpected number of active pods.  Expected %d, saw %d\n", name, tc.expectedActive, actual.Status.Active)
		}
		if actual.Status.Succeeded != tc.expectedSucceeded {
			t.Errorf("%s: unexpected number of succeeded pods.  Expected %d, saw %d\n", name, tc.expectedSucceeded, actual.Status.Succeeded)
		}
		if actual.Status.Failed != tc.expectedFailed {
			t.Errorf("%s: unexpected number of failed pods.  Expected %d, saw %d\n", name, tc.expectedFailed, actual.Status.Failed)
		}
		if actual.Status.StartTime == nil {
			t.Errorf("%s: .status.startTime was not set", name)
		}
		// validate conditions
		if !getCondition(actual, batch.JobFailed, tc.expectedConditionReason) {
			t.Errorf("%s: expected fail condition.  Got %#v", name, actual.Status.Conditions)
		}
	}
}

func getCondition(job *batch.Job, condition batch.JobConditionType, reason string) bool {
	for _, v := range job.Status.Conditions {
		if v.Type == condition && v.Status == v1.ConditionTrue && v.Reason == reason {
			return true
		}
	}
	return false
}

func TestSyncPastDeadlineJobFinished(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	var actual *batch.Job
	manager.updateHandler = func(job *batch.Job) error {
		actual = job
		return nil
	}

	job := newJob(1, 1, 6)
	activeDeadlineSeconds := int64(10)
	job.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	start := metav1.Unix(metav1.Now().Time.Unix()-15, 0)
	job.Status.StartTime = &start
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobFailed, "DeadlineExceeded", "Job was active longer than specified deadline"))
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
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady

	job := newJob(1, 1, 6)
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobComplete, "", ""))
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
	manager, _ := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	manager.updateHandler = func(job *batch.Job) error { return nil }
	job := newJob(2, 2, 6)
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
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	updateError := fmt.Errorf("update error")
	manager.updateHandler = func(job *batch.Job) error {
		manager.queue.AddRateLimited(testutil.GetKey(job, t))
		return updateError
	}
	job := newJob(2, 2, 6)
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
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
	job2.Name = "job2"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job2)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job2)
	pod3 := newPod("pod3", job1)
	// Make pod3 an orphan that doesn't match. It should be ignored.
	pod3.OwnerReferences = nil
	pod3.Labels = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod3)

	pods, err := jm.getPodsForJob(job1)
	if err != nil {
		t.Fatalf("getPodsForJob() error: %v", err)
	}
	if got, want := len(pods), 1; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
	if got, want := pods[0].Name, "pod1"; got != want {
		t.Errorf("pod.Name = %v, want %v", got, want)
	}

	pods, err = jm.getPodsForJob(job2)
	if err != nil {
		t.Fatalf("getPodsForJob() error: %v", err)
	}
	if got, want := len(pods), 1; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
	if got, want := pods[0].Name, "pod2"; got != want {
		t.Errorf("pod.Name = %v, want %v", got, want)
	}
}

func TestGetPodsForJobAdopt(t *testing.T) {
	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	clientset := fake.NewSimpleClientset(job1)
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job1)
	// Make this pod an orphan. It should still be returned because it's adopted.
	pod2.OwnerReferences = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	pods, err := jm.getPodsForJob(job1)
	if err != nil {
		t.Fatalf("getPodsForJob() error: %v", err)
	}
	if got, want := len(pods), 2; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
}

func TestGetPodsForJobNoAdoptIfBeingDeleted(t *testing.T) {
	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job1.DeletionTimestamp = &metav1.Time{}
	clientset := fake.NewSimpleClientset(job1)
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job1)
	// Make this pod an orphan. It should not be adopted because the Job is being deleted.
	pod2.OwnerReferences = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	pods, err := jm.getPodsForJob(job1)
	if err != nil {
		t.Fatalf("getPodsForJob() error: %v", err)
	}
	if got, want := len(pods), 1; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
	if got, want := pods[0].Name, pod1.Name; got != want {
		t.Errorf("pod.Name = %q, want %q", got, want)
	}
}

func TestGetPodsForJobNoAdoptIfBeingDeletedRace(t *testing.T) {
	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	// The up-to-date object says it's being deleted.
	job1.DeletionTimestamp = &metav1.Time{}
	clientset := fake.NewSimpleClientset(job1)
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	// The cache says it's NOT being deleted.
	cachedJob := *job1
	cachedJob.DeletionTimestamp = nil
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(&cachedJob)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job1)
	// Make this pod an orphan. It should not be adopted because the Job is being deleted.
	pod2.OwnerReferences = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	pods, err := jm.getPodsForJob(job1)
	if err != nil {
		t.Fatalf("getPodsForJob() error: %v", err)
	}
	if got, want := len(pods), 1; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
	if got, want := pods[0].Name, pod1.Name; got != want {
		t.Errorf("pod.Name = %q, want %q", got, want)
	}
}

func TestGetPodsForJobRelease(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	informer.Batch().V1().Jobs().Informer().GetIndexer().Add(job1)

	pod1 := newPod("pod1", job1)
	pod2 := newPod("pod2", job1)
	// Make this pod not match, even though it's owned. It should be released.
	pod2.Labels = nil
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod1)
	informer.Core().V1().Pods().Informer().GetIndexer().Add(pod2)

	pods, err := jm.getPodsForJob(job1)
	if err != nil {
		t.Fatalf("getPodsForJob() error: %v", err)
	}
	if got, want := len(pods), 1; got != want {
		t.Errorf("len(pods) = %v, want %v", got, want)
	}
	if got, want := pods[0].Name, "pod1"; got != want {
		t.Errorf("pod.Name = %v, want %v", got, want)
	}
}

func TestAddPod(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
	job2.Name = "job2"
	job3 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
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
	jm, informer := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	jm.podStoreSynced = alwaysReady
	jm.jobStoreSynced = alwaysReady

	job1 := newJob(1, 1, 6)
	job1.Name = "job1"
	job2 := newJob(1, 1, 6)
	job2.Name = "job2"
	job3 := newJob(1, 1, 6)
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
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.jobStoreSynced = alwaysReady
	manager.updateHandler = func(job *batch.Job) error { return nil }

	job := newJob(2, 2, 6)
	sharedInformerFactory.Batch().V1().Jobs().Informer().GetIndexer().Add(job)
	pods := newPodList(2, v1.PodPending, job)
	podIndexer := sharedInformerFactory.Core().V1().Pods().Informer().GetIndexer()
	podIndexer.Add(&pods[0])

	manager.expectations = FakeJobExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the job
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
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
	testJob := newJob(2, 2, 6)
	clientset := fake.NewSimpleClientset(testJob)
	fakeWatch := watch.NewFake()
	clientset.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))
	manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
		manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
		job := newJob(tc.parallelism, tc.completions, tc.backoffLimit)
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
		if getCondition(actual, batch.JobFailed, "BackoffLimitExceeded") {
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
	job := newJob(1, 1, 1)
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
			manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
			manager, sharedInformerFactory := newJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
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
			job := newJob(tc.parallelism, tc.completions, tc.backoffLimit)
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
			if tc.expectedCondition != nil && !getCondition(actual, *tc.expectedCondition, tc.expectedConditionReason) {
				t.Errorf("expected completion condition.  Got %#v", actual.Status.Conditions)
			}
		})
	}
}
