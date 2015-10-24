/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

// Give each test that starts a background controller up to 1/2 a second.
// Since we need to start up a goroutine to test watch, this routine needs
// to get cpu before the test can complete. If the test is starved of cpu,
// the watch test will take up to 1/2 a second before timing out.
const controllerTimeout = 500 * time.Millisecond

var alwaysReady = func() bool { return true }

func newJob(parallelism, completions int) *extensions.Job {
	return &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foobar",
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.JobSpec{
			Parallelism: &parallelism,
			Completions: &completions,
			Selector: &extensions.PodSelector{
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
}

func getKey(job *extensions.Job, t *testing.T) string {
	if key, err := controller.KeyFunc(job); err != nil {
		t.Errorf("Unexpected error getting key for job %v: %v", job.Name, err)
		return ""
	} else {
		return key
	}
}

// create count pods with the given phase for the given job
func newPodList(count int, status api.PodPhase, job *extensions.Job) []api.Pod {
	pods := []api.Pod{}
	for i := 0; i < count; i++ {
		newPod := api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("pod-%v", unversioned.Now().UnixNano()),
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
		parallelism int
		completions int

		// pod setup
		podControllerError error
		activePods         int
		succeededPods      int
		failedPods         int

		// expectations
		expectedCreations int
		expectedDeletions int
		expectedActive    int
		expectedSucceeded int
		expectedFailed    int
		expectedComplete  bool
	}{
		"job start": {
			2, 5,
			nil, 0, 0, 0,
			2, 0, 2, 0, 0, false,
		},
		"correct # of pods": {
			2, 5,
			nil, 2, 0, 0,
			0, 0, 2, 0, 0, false,
		},
		"too few active pods": {
			2, 5,
			nil, 1, 1, 0,
			1, 0, 2, 1, 0, false,
		},
		"too few active pods, with controller error": {
			2, 5,
			fmt.Errorf("Fake error"), 1, 1, 0,
			0, 0, 1, 1, 0, false,
		},
		"too many active pods": {
			2, 5,
			nil, 3, 0, 0,
			0, 1, 2, 0, 0, false,
		},
		"too many active pods, with controller error": {
			2, 5,
			fmt.Errorf("Fake error"), 3, 0, 0,
			0, 0, 3, 0, 0, false,
		},
		"failed pod": {
			2, 5,
			nil, 1, 1, 1,
			1, 0, 2, 1, 1, false,
		},
		"job finish": {
			2, 5,
			nil, 0, 5, 0,
			0, 0, 0, 5, 0, true,
		},
		"more active pods than completions": {
			2, 5,
			nil, 10, 0, 0,
			0, 8, 2, 0, 0, false,
		},
		"status change": {
			2, 5,
			nil, 2, 2, 0,
			0, 0, 2, 2, 0, false,
		},
	}

	for name, tc := range testCases {
		// job manager setup
		client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Default.GroupAndVersion()})
		manager := NewJobController(client, controller.NoResyncPeriodFunc)
		fakePodControl := controller.FakePodControl{Err: tc.podControllerError}
		manager.podControl = &fakePodControl
		manager.podStoreSynced = alwaysReady
		var actual *extensions.Job
		manager.updateHandler = func(job *extensions.Job) error {
			actual = job
			return nil
		}

		// job & pods setup
		job := newJob(tc.parallelism, tc.completions)
		manager.jobStore.Store.Add(job)
		for _, pod := range newPodList(tc.activePods, api.PodRunning, job) {
			manager.podStore.Store.Add(&pod)
		}
		for _, pod := range newPodList(tc.succeededPods, api.PodSucceeded, job) {
			manager.podStore.Store.Add(&pod)
		}
		for _, pod := range newPodList(tc.failedPods, api.PodFailed, job) {
			manager.podStore.Store.Add(&pod)
		}

		// run
		err := manager.syncJob(getKey(job, t))
		if err != nil {
			t.Errorf("%s: unexpected error when syncing jobs %v", err)
		}

		// validate created/deleted pods
		if len(fakePodControl.Templates) != tc.expectedCreations {
			t.Errorf("%s: unexpected number of creates.  Expected %d, saw %d\n", name, tc.expectedCreations, len(fakePodControl.Templates))
		}
		if len(fakePodControl.DeletePodName) != tc.expectedDeletions {
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
		// validate conditions
		if tc.expectedComplete {
			completed := false
			for _, v := range actual.Status.Conditions {
				if v.Type == extensions.JobComplete && v.Status == api.ConditionTrue {
					completed = true
					break
				}
			}
			if !completed {
				t.Errorf("%s: expected completion condition.  Got %v", name, actual.Status.Conditions)
			}
		}
	}
}

func TestSyncJobDeleted(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Default.GroupAndVersion()})
	manager := NewJobController(client, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.updateHandler = func(job *extensions.Job) error { return nil }
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
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Default.GroupAndVersion()})
	manager := NewJobController(client, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.updateHandler = func(job *extensions.Job) error { return fmt.Errorf("Fake error") }
	job := newJob(2, 2)
	manager.jobStore.Store.Add(job)
	err := manager.syncJob(getKey(job, t))
	if err != nil {
		t.Errorf("Unxpected error when syncing jobs, got %v", err)
	}
	ch := make(chan interface{})
	go func() {
		item, _ := manager.queue.Get()
		ch <- item
	}()
	select {
	case key := <-ch:
		expectedKey := getKey(job, t)
		if key != expectedKey {
			t.Errorf("Expected requeue of job with key %s got %s", expectedKey, key)
		}
	case <-time.After(controllerTimeout):
		manager.queue.ShutDown()
		t.Errorf("Expected to find a job in the queue, found none.")
	}
}

func TestJobPodLookup(t *testing.T) {
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Default.GroupAndVersion()})
	manager := NewJobController(client, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady
	testCases := []struct {
		job *extensions.Job
		pod *api.Pod

		expectedName string
	}{
		// pods without labels don't match any job
		{
			job: &extensions.Job{
				ObjectMeta: api.ObjectMeta{Name: "basic"},
			},
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo1", Namespace: api.NamespaceAll},
			},
			expectedName: "",
		},
		// matching labels, different namespace
		{
			job: &extensions.Job{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: extensions.JobSpec{
					Selector: &extensions.PodSelector{
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
			job: &extensions.Job{
				ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
				Spec: extensions.JobSpec{
					Selector: &extensions.PodSelector{
						MatchExpressions: []extensions.PodSelectorRequirement{
							{
								Key:      "foo",
								Operator: extensions.PodSelectorOpIn,
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
	client := client.NewOrDie(&client.Config{Host: "", Version: testapi.Default.GroupAndVersion()})
	manager := NewJobController(client, controller.NoResyncPeriodFunc)
	fakePodControl := controller.FakePodControl{}
	manager.podControl = &fakePodControl
	manager.podStoreSynced = alwaysReady
	manager.updateHandler = func(job *extensions.Job) error { return nil }

	job := newJob(2, 2)
	manager.jobStore.Store.Add(job)
	pods := newPodList(2, api.PodPending, job)
	manager.podStore.Store.Add(&pods[0])

	manager.expectations = FakeJobExpectations{
		controller.NewControllerExpectations(), true, func() {
			// If we check active pods before checking expectataions, the job
			// will create a new replica because it doesn't see this pod, but
			// has fulfilled its expectations.
			manager.podStore.Store.Add(&pods[1])
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
	fakeWatch := watch.NewFake()
	client := &testclient.Fake{}
	client.AddWatchReactor("*", testclient.DefaultWatchReactor(fakeWatch, nil))
	manager := NewJobController(client, controller.NoResyncPeriodFunc)
	manager.podStoreSynced = alwaysReady

	var testJob extensions.Job
	received := make(chan string)

	// The update sent through the fakeWatcher should make its way into the workqueue,
	// and eventually into the syncHandler.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.jobStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find job under key %v", key)
		}
		job := *obj.(*extensions.Job)
		if !api.Semantic.DeepDerivative(job, testJob) {
			t.Errorf("Expected %#v, but got %#v", testJob, job)
		}
		received <- key
		return nil
	}
	// Start only the job watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.jobController.Run(stopCh)
	go util.Until(manager.worker, 10*time.Millisecond, stopCh)

	// We're sending new job to see if it reaches syncHandler.
	testJob.Name = "foo"
	fakeWatch.Add(&testJob)
	select {
	case <-received:
	case <-time.After(controllerTimeout):
		t.Errorf("Expected 1 call but got 0")
	}

	// We're sending fake finished job, to see if it reaches syncHandler - it should not,
	// since we're filtering out finished jobs.
	testJobv2 := extensions.Job{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Status: extensions.JobStatus{
			Conditions: []extensions.JobCondition{{
				Type:               extensions.JobComplete,
				Status:             api.ConditionTrue,
				LastProbeTime:      unversioned.Now(),
				LastTransitionTime: unversioned.Now(),
			}},
		},
	}
	fakeWatch.Modify(&testJobv2)

	select {
	case <-received:
		t.Errorf("Expected 0 call but got 1")
	case <-time.After(controllerTimeout):
	}
}

func TestWatchPods(t *testing.T) {
	fakeWatch := watch.NewFake()
	client := &testclient.Fake{}
	client.AddWatchReactor("*", testclient.DefaultWatchReactor(fakeWatch, nil))
	manager := NewJobController(client, controller.NoResyncPeriodFunc)

	manager.podStoreSynced = alwaysReady

	// Put one job and one pod into the store
	testJob := newJob(2, 2)
	manager.jobStore.Store.Add(testJob)
	received := make(chan string)
	// The pod update sent through the fakeWatcher should figure out the managing job and
	// send it into the syncHandler.
	manager.syncHandler = func(key string) error {

		obj, exists, err := manager.jobStore.Store.GetByKey(key)
		if !exists || err != nil {
			t.Errorf("Expected to find job under key %v", key)
		}
		job := obj.(*extensions.Job)
		if !api.Semantic.DeepDerivative(job, testJob) {
			t.Errorf("\nExpected %#v,\nbut got %#v", testJob, job)
		}
		close(received)
		return nil
	}
	// Start only the pod watcher and the workqueue, send a watch event,
	// and make sure it hits the sync method for the right job.
	stopCh := make(chan struct{})
	defer close(stopCh)
	go manager.podController.Run(stopCh)
	go util.Until(manager.worker, 10*time.Millisecond, stopCh)

	pods := newPodList(1, api.PodRunning, testJob)
	testPod := pods[0]
	testPod.Status.Phase = api.PodFailed
	fakeWatch.Add(&testPod)

	select {
	case <-received:
	case <-time.After(controllerTimeout):
		t.Errorf("Expected 1 call but got 0")
	}
}
