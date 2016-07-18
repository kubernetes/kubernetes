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
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/rand"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

var alwaysReady = func() bool { return true }

func newJob(parallelism, completions int32) *batch.Job {
	j := &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foobar",
			Namespace: api.NamespaceDefault,
		},
		Spec: batch.JobSpec{
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
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
	return j
}

func getKey(job *batch.Job, t *testing.T) string {
	if key, err := controller.KeyFunc(job); err != nil {
		t.Errorf("Unexpected error getting key for job %v: %v", job.Name, err)
		return ""
	} else {
		return key
	}
}

// create count pods with the given phase for the given job
func newPodList(count int32, status api.PodPhase, job *batch.Job) []api.Pod {
	pods := []api.Pod{}
	for i := int32(0); i < count; i++ {
		newPod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("pod-%v", rand.String(10)),
				Labels:    job.Spec.Selector.MatchLabels,
				Namespace: job.Namespace,
			},
			Status: api.PodStatus{Phase: status},
		}
		pods = append(pods, newPod)
	}
	return pods
}

func TestControllerSyncJob(t *testing.T) {
	testCases := map[string]struct {
		// job setup
		parallelism int32
		completions int32
		deleting    bool

		// pod setup
		podControllerError error
		pendingPods        int32
		activePods         int32
		succeededPods      int32
		failedPods         int32

		// expectations
		expectedCreations int32
		expectedDeletions int32
		expectedActive    int32
		expectedSucceeded int32
		expectedFailed    int32
		expectedComplete  bool
	}{
		"job start": {
			2, 5, false,
			nil, 0, 0, 0, 0,
			2, 0, 2, 0, 0, false,
		},
		"WQ job start": {
			2, -1, false,
			nil, 0, 0, 0, 0,
			2, 0, 2, 0, 0, false,
		},
		"pending pods": {
			2, 5, false,
			nil, 2, 0, 0, 0,
			0, 0, 2, 0, 0, false,
		},
		"correct # of pods": {
			2, 5, false,
			nil, 0, 2, 0, 0,
			0, 0, 2, 0, 0, false,
		},
		"WQ job: correct # of pods": {
			2, -1, false,
			nil, 0, 2, 0, 0,
			0, 0, 2, 0, 0, false,
		},
		"too few active pods": {
			2, 5, false,
			nil, 0, 1, 1, 0,
			1, 0, 2, 1, 0, false,
		},
		"too few active pods with a dynamic job": {
			2, -1, false,
			nil, 0, 1, 0, 0,
			1, 0, 2, 0, 0, false,
		},
		"too few active pods, with controller error": {
			2, 5, false,
			fmt.Errorf("Fake error"), 0, 1, 1, 0,
			1, 0, 1, 1, 0, false,
		},
		"too many active pods": {
			2, 5, false,
			nil, 0, 3, 0, 0,
			0, 1, 2, 0, 0, false,
		},
		"too many active pods, with controller error": {
			2, 5, false,
			fmt.Errorf("Fake error"), 0, 3, 0, 0,
			0, 1, 3, 0, 0, false,
		},
		"failed pod": {
			2, 5, false,
			nil, 0, 1, 1, 1,
			1, 0, 2, 1, 1, false,
		},
		"job finish": {
			2, 5, false,
			nil, 0, 0, 5, 0,
			0, 0, 0, 5, 0, true,
		},
		"WQ job finishing": {
			2, -1, false,
			nil, 0, 1, 1, 0,
			0, 0, 1, 1, 0, false,
		},
		"WQ job all finished": {
			2, -1, false,
			nil, 0, 0, 2, 0,
			0, 0, 0, 2, 0, true,
		},
		"WQ job all finished despite one failure": {
			2, -1, false,
			nil, 0, 0, 1, 1,
			0, 0, 0, 1, 1, true,
		},
		"more active pods than completions": {
			2, 5, false,
			nil, 0, 10, 0, 0,
			0, 8, 2, 0, 0, false,
		},
		"status change": {
			2, 5, false,
			nil, 0, 2, 2, 0,
			0, 0, 2, 2, 0, false,
		},
		"deleting job": {
			2, 5, true,
			nil, 1, 1, 1, 0,
			0, 0, 2, 1, 0, false,
		},
	}

	for name, tc := range testCases {
		// job manager setup
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
		manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
		fakePodControl := controller.FakePodControl{Err: tc.podControllerError}
		manager.podControl = &fakePodControl
		manager.podStoreSynced = alwaysReady
		var actual *batch.Job
		manager.updateHandler = func(job *batch.Job) error {
			actual = job
			return nil
		}

		// job & pods setup
		job := newJob(tc.parallelism, tc.completions)
		if tc.deleting {
			now := unversioned.Now()
			job.DeletionTimestamp = &now
		}
		manager.jobStore.Store.Add(job)
		for _, pod := range newPodList(tc.pendingPods, api.PodPending, job) {
			manager.podStore.Indexer.Add(&pod)
		}
		for _, pod := range newPodList(tc.activePods, api.PodRunning, job) {
			manager.podStore.Indexer.Add(&pod)
		}
		for _, pod := range newPodList(tc.succeededPods, api.PodSucceeded, job) {
			manager.podStore.Indexer.Add(&pod)
		}
		for _, pod := range newPodList(tc.failedPods, api.PodFailed, job) {
			manager.podStore.Indexer.Add(&pod)
		}

		// run
		err := manager.syncJob(getKey(job, t))
		if err != nil {
			t.Errorf("%s: unexpected error when syncing jobs %v", name, err)
		}

		// validate created/deleted pods
		if int32(len(fakePodControl.Templates)) != tc.expectedCreations {
			t.Errorf("%s: unexpected number of creates.  Expected %d, saw %d\n", name, tc.expectedCreations, len(fakePodControl.Templates))
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
		if tc.expectedComplete && !getCondition(actual, batch.JobComplete) {
			t.Errorf("%s: expected completion condition.  Got %#v", name, actual.Status.Conditions)
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

		// pod setup
		activePods    int32
		succeededPods int32
		failedPods    int32

		// expectations
		expectedDeletions int32
		expectedActive    int32
		expectedSucceeded int32
		expectedFailed    int32
	}{
		"activeDeadlineSeconds less than single pod execution": {
			1, 1, 10, 15,
			1, 0, 0,
			1, 0, 0, 1,
		},
		"activeDeadlineSeconds bigger than single pod execution": {
			1, 2, 10, 15,
			1, 1, 0,
			1, 0, 1, 1,
		},
		"activeDeadlineSeconds times-out before any pod starts": {
			1, 1, 10, 10,
			0, 0, 0,
			0, 0, 0, 0,
		},
	}

	for name, tc := range testCases {
		// job manager setup
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
		manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
		fakePodControl := controller.FakePodControl{}
		manager.podControl = &fakePodControl
		manager.podStoreSynced = alwaysReady
		var actual *batch.Job
		manager.updateHandler = func(job *batch.Job) error {
			actual = job
			return nil
		}

		// job & pods setup
		job := newJob(tc.parallelism, tc.completions)
		job.Spec.ActiveDeadlineSeconds = &tc.activeDeadlineSeconds
		start := unversioned.Unix(unversioned.Now().Time.Unix()-tc.startTime, 0)
		job.Status.StartTime = &start
		manager.jobStore.Store.Add(job)
		for _, pod := range newPodList(tc.activePods, api.PodRunning, job) {
			manager.podStore.Indexer.Add(&pod)
		}
		for _, pod := range newPodList(tc.succeededPods, api.PodSucceeded, job) {
			manager.podStore.Indexer.Add(&pod)
		}
		for _, pod := range newPodList(tc.failedPods, api.PodFailed, job) {
			manager.podStore.Indexer.Add(&pod)
		}

		// run
		err := manager.syncJob(getKey(job, t))
		if err != nil {
			t.Errorf("%s: unexpected error when syncing jobs %v", name, err)
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
		if !getCondition(actual, batch.JobFailed) {
			t.Errorf("%s: expected fail condition.  Got %#v", name, actual.Status.Conditions)
		}
	}
}

func getCondition(job *batch.Job, condition batch.JobConditionType) bool {
	for _, v := range job.Status.Conditions {
		if v.Type == condition && v.Status == api.ConditionTrue {
			return true
		}
	}
	return false
}

func TestSyncPastDeadlineJobFinished(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	var actual *batch.Job
	manager.updateHandler = func(job *batch.Job) error {
		actual = job
		return nil
	}

	job := newJob(1, 1)
	activeDeadlineSeconds := int64(10)
	job.Spec.ActiveDeadlineSeconds = &activeDeadlineSeconds
	start := unversioned.Unix(unversioned.Now().Time.Unix()-15, 0)
	job.Status.StartTime = &start
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobFailed, "DeadlineExceeded", "Job was active longer than specified deadline"))
	manager.jobStore.Store.Add(job)
	err := manager.syncJob(getKey(job, t))
	if err != nil {
		t.Errorf("Unexpected error when syncing jobs %v", err)
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
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady

	job := newJob(1, 1)
	job.Status.Conditions = append(job.Status.Conditions, newCondition(batch.JobComplete, "", ""))
	manager.jobStore.Store.Add(job)
	err := manager.syncJob(getKey(job, t))
	if err != nil {
		t.Fatalf("Unexpected error when syncing jobs %v", err)
	}
	uncastJob, _, err := manager.jobStore.Store.Get(job)
	if err != nil {
		t.Fatalf("Unexpected error when trying to get job from the store: %v", err)
	}
	actual := uncastJob.(*batch.Job)
	// Verify that after syncing a complete job, the conditions are the same.
	if got, expected := len(actual.Status.Conditions), 1; got != expected {
		t.Fatalf("Unexpected job status conditions amount; expected %d, got %d", expected, got)
	}
}

func TestSyncJobDeleted(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.updateHandler = func(job *batch.Job) error { return nil }
	job := newJob(2, 2)
	err := manager.syncJob(getKey(job, t))
	if err != nil {
		t.Errorf("Unexpected error when syncing jobs %v", err)
	}
	if len(fakePodControl.Templates) != 0 {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", 0, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != 0 {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", 0, len(fakePodControl.DeletePodName))
	}
}

func TestSyncJobUpdateRequeue(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.updateHandler = func(job *batch.Job) error { return fmt.Errorf("Fake error") }
	job := newJob(2, 2)
	manager.jobStore.Store.Add(job)
	err := manager.syncJob(getKey(job, t))
	if err != nil {
		t.Errorf("Unxpected error when syncing jobs, got %v", err)
	}
	t.Log("Waiting for a job in the queue")
	key, _ := manager.queue.Get()
	expectedKey := getKey(job, t)
	if key != expectedKey {
		t.Errorf("Expected requeue of job with key %s got %s", expectedKey, key)
	}
}

func TestJobPodLookup(t *testing.T) {
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady
	testCases := []struct {
		job *batch.Job
		pod *api.Pod

		expectedName string
	}{
		// pods without labels don't match any job
		{
			job: &batch.Job{
				ObjectMeta: api.ObjectMeta{Name: "basic"},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo1", Namespace: api.NamespaceAll},
			},
			expectedName: "",
		},
		// matching labels, different namespace
		{
			job: &batch.Job{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: batch.JobSpec{
					Selector: &unversioned.LabelSelector{
						MatchLabels: map[string]string{"foo": "bar"},
					},
				},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
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
				ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
				Spec: batch.JobSpec{
					Selector: &unversioned.LabelSelector{
						MatchExpressions: []unversioned.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: unversioned.LabelSelectorOpIn,
								Values:   []string{"bar"},
							},
						},
					},
				},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo3",
					Namespace: "ns",
					Labels:    map[string]string{"foo": "bar"},
				},
			},
			expectedName: "bar",
		},
	}
	for _, tc := range testCases {
		manager.jobStore.Add(tc.job)
		if job := manager.getPodJob(tc.pod); job != nil {
			if tc.expectedName != job.Name {
				t.Errorf("Got job %+v expected %+v", job.Name, tc.expectedName)
			}
		} else if tc.expectedName != "" {
			t.Errorf("Expected a job %v pod %v, found none", tc.expectedName, tc.pod.Name)
		}
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
	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.updateHandler = func(job *batch.Job) error { return nil }

	job := newJob(2, 2)
	manager.jobStore.Store.Add(job)
	pods := newPodList(2, api.PodPending, job)
	manager.podStore.Indexer.Add(&pods[0])

	manager.expectations = FakeJobExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the job
			// will create a new replica because it doesn't see this pod, but
			// has fulfilled its expectations.
			manager.podStore.Indexer.Add(&pods[1])
		},
	}
	manager.syncJob(getKey(job, t))
	if len(fakePodControl.Templates) != 0 {
		t.Errorf("Unexpected number of creates.  Expected %d, saw %d\n", 0, len(fakePodControl.Templates))
	}
	if len(fakePodControl.DeletePodName) != 0 {
		t.Errorf("Unexpected number of deletes.  Expected %d, saw %d\n", 0, len(fakePodControl.DeletePodName))
	}
}

type FakeWatcher struct {
	w *watch.FakeWatcher
	*testclient.Fake
}

func TestWatchJobs(t *testing.T) {
	clientset := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	clientset.PrependWatchReactor("jobs", core.DefaultWatchReactor(fakeWatch, nil))
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady

	var testJob batch.Job
	received := make(chan struct{})

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.jobStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find job under key %v", key)
		}
		job, ok := obj.(*batch.Job)
		if !ok {
			t.Fatalf("unexpected type: %v %#v", reflect.TypeOf(obj), obj)
		}
		if !api.Semantic.DeepDerivative(*job, testJob) {
			t.Errorf("Expected %#v, but got %#v", testJob, *job)
		}
		close(received)
		return nil
	}
	// Start only the job watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.Run(1, stopCh)

	// We're sending new job to see if it reaches syncHandler.
	testJob.Name = "foo"
	fakeWatch.Add(&testJob)
	t.Log("Waiting for job to reach syncHandler")
	<-received
}

func TestWatchPods(t *testing.T) {
	testJob := newJob(2, 2)
	clientset := fake.NewSimpleClientset(testJob)
	fakeWatch := watch.NewFake()
	clientset.PrependWatchReactor("pods", core.DefaultWatchReactor(fakeWatch, nil))
	manager := NewJobControllerFromClient(clientset, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady

	// Put one job and one pod into the store
	manager.jobStore.Store.Add(testJob)
	received := make(chan struct{})
	// The pod update sent through the fakeWatcher should figure out the managing job and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {
		obj, exists, err := manager.jobStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find job under key %v", key)
			close(received)
			return nil
		}
		job, ok := obj.(*batch.Job)
		if !ok {
			t.Errorf("unexpected type: %v %#v", reflect.TypeOf(obj), obj)
			close(received)
			return nil
		}
		if !api.Semantic.DeepDerivative(job, testJob) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testJob, job)
			close(received)
			return nil
		}
		close(received)
		return nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right job.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.internalPodInformer.Run(stopCh)
	go wait.Until(manager.worker, 10*time.Millisecond, stopCh)

	pods := newPodList(1, api.PodRunning, testJob)
	testPod := pods[0]
	testPod.Status.Phase = api.PodFailed
	fakeWatch.Add(&testPod)

	t.Log("Waiting for pod to reach syncHandler")
	<-received
}
