/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package workflow

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/controller"

	client "k8s.io/kubernetes/pkg/client/unversioned"
)

// utility function to create a JobTemplateSpec
func newJobTemplateSpec() *extensions.JobTemplateSpec {
	return &extensions.JobTemplateSpec{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: extensions.JobSpec{
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

func newJobTemplateStatus() extensions.WorkflowStepStatus {
	return extensions.WorkflowStepStatus{
		Complete: false,
		Reference: api.ObjectReference{
			Kind:      "Job",
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
	}
}

func getKey(workflow *extensions.Workflow, t *testing.T) string {
	key, err := controller.KeyFunc(workflow)
	if err != nil {
		t.Errorf("Unexpected error getting key for workflow %v: %v", workflow.Name, err)
		return ""
	}
	return key
}

func TestControllerSyncWorkflow(t *testing.T) {
	testCases := map[string]struct {
		workflow           *extensions.Workflow
		jobs               []extensions.Job
		checkWorkflow      func(testName string, workflow *extensions.Workflow, t *testing.T)
		expectedStartedJob int
	}{

		"workflow start": {
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"myJob": {
							JobTemplate: newJobTemplateSpec(),
						},
					},
				},
			},
			jobs:               []extensions.Job{},
			expectedStartedJob: 1,
		},
		"workflow status update": {
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"myJob": {
							JobTemplate: newJobTemplateSpec(),
						},
					},
				},
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{},
					Statuses: map[string]extensions.WorkflowStepStatus{
						"myJob": newJobTemplateStatus(),
					},
				},
			},
			jobs: []extensions.Job{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
						Labels: map[string]string{
							"foo": "bar",
							controller.WorkflowStepLabelKey: "myJob",
						},
						SelfLink: "/apis/v1/jobs/foo",
					},
					Spec:   extensions.JobSpec{},
					Status: extensions.JobStatus{},
				},
			},
			checkWorkflow: func(testName string, workflow *extensions.Workflow, t *testing.T) {
				stepStatus, ok := workflow.Status.Statuses["myJob"]
				if !ok {
					t.Errorf("%s, Workflow step not updated", testName)
					return
				}
				if stepStatus.Complete {
					t.Errorf("%s, Workflow wrongly updated", testName)
				}
			},
			expectedStartedJob: 0,
		},
		"workflow step status update to complete": {
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"myJob": {
							JobTemplate: newJobTemplateSpec(),
						},
					},
				},
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{},
					Statuses: map[string]extensions.WorkflowStepStatus{
						"myJob": newJobTemplateStatus(),
					},
				},
			},
			jobs: []extensions.Job{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
						Labels: map[string]string{
							"foo": "bar",
							controller.WorkflowStepLabelKey: "myJob",
						},
						SelfLink: "/apis/v1/jobs/foo",
					},
					Spec: extensions.JobSpec{},
					Status: extensions.JobStatus{
						Conditions: []extensions.JobCondition{
							{
								Type:   extensions.JobComplete,
								Status: api.ConditionTrue,
							},
						},
					},
				},
			},
			checkWorkflow: func(testName string, workflow *extensions.Workflow, t *testing.T) {
				stepStatus, ok := workflow.Status.Statuses["myJob"]
				if !ok {
					t.Errorf("%s, Workflow step not updated", testName)
					return
				}
				if !stepStatus.Complete {
					t.Errorf("%s, Workflow wrongly updated", testName)
				}
			},
			expectedStartedJob: 0,
		},
		"workflow status update to complete": {
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"myJob": {
							JobTemplate: newJobTemplateSpec(),
						},
					},
				},
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{},
					Statuses: map[string]extensions.WorkflowStepStatus{
						"myJob": {
							Complete: true,
							Reference: api.ObjectReference{
								Kind:      "Job",
								Name:      "foo",
								Namespace: api.NamespaceDefault,
							},
						},
					},
				},
			},
			jobs: []extensions.Job{}, // jobs no retrieved step only
			checkWorkflow: func(testName string, workflow *extensions.Workflow, t *testing.T) {
				if !isWorkflowFinished(workflow) {
					t.Errorf("%s, Workflow should be finished:\n %#v", testName, workflow)
				}
				if workflow.Status.CompletionTime == nil {
					t.Errorf("%s, CompletionTime not set", testName)
				}
			},
			expectedStartedJob: 0,
		},
		"workflow step dependency complete 3": {
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"one": {
							JobTemplate: newJobTemplateSpec(),
						},
						"two": {
							JobTemplate:  newJobTemplateSpec(),
							Dependencies: []string{"one"},
						},
						"three": {
							JobTemplate:  newJobTemplateSpec(),
							Dependencies: []string{"one"},
						},
						"four": {
							JobTemplate:  newJobTemplateSpec(),
							Dependencies: []string{"one"},
						},
						"five": {
							JobTemplate:  newJobTemplateSpec(),
							Dependencies: []string{"two", "three", "four"},
						},
					},
				},
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{},
					Statuses: map[string]extensions.WorkflowStepStatus{
						"one": {
							Complete: true,
							Reference: api.ObjectReference{
								Kind:      "Job",
								Name:      "foo",
								Namespace: api.NamespaceDefault,
							},
						},
					},
				},
			},
			jobs: []extensions.Job{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
						Labels: map[string]string{
							"foo": "bar",
							controller.WorkflowStepLabelKey: "one",
						},
						SelfLink: "/apis/v1/jobs/foo",
					},
					Spec:   extensions.JobSpec{},
					Status: extensions.JobStatus{},
				},
			},
			checkWorkflow:      func(testName string, workflow *extensions.Workflow, t *testing.T) {},
			expectedStartedJob: 3,
		},
		"workflow step dependency not complete": {
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"one": {
							JobTemplate: newJobTemplateSpec(),
						},
						"two": {
							JobTemplate:  newJobTemplateSpec(),
							Dependencies: []string{"one"},
						},
					},
				},
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{},
					Statuses: map[string]extensions.WorkflowStepStatus{
						"one": {
							Complete: false,
							Reference: api.ObjectReference{
								Kind:      "Job",
								Name:      "foo",
								Namespace: api.NamespaceDefault,
							},
						},
					},
				},
			},
			jobs: []extensions.Job{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
						Labels: map[string]string{
							"foo": "bar",
							controller.WorkflowStepLabelKey: "one",
						},
						SelfLink: "/apis/v1/jobs/foo",
					},
					Spec:   extensions.JobSpec{},
					Status: extensions.JobStatus{},
				},
			},
			checkWorkflow:      func(testName string, workflow *extensions.Workflow, t *testing.T) {},
			expectedStartedJob: 0,
		},
	}
	for name, tc := range testCases {
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
		oldClient := client.NewOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

		manager := NewWorkflowController(oldClient, clientset, controller.NoResyncPeriodFunc)
		fakeJobControl := controller.FakeJobControl{}
		manager.jobControl = &fakeJobControl
		manager.jobStoreSynced = func() bool { return true }
		var actual *extensions.Workflow
		manager.updateHandler = func(workflow *extensions.Workflow) error {
			actual = workflow
			return nil
		}
		// setup workflow, jobs
		manager.workflowStore.Store.Add(tc.workflow)
		for _, job := range tc.jobs {
			manager.jobStore.Store.Add(&job)
		}
		err := manager.syncWorkflow(getKey(tc.workflow, t))
		if err != nil {
			t.Errorf("%s: unexpected error syncing workflow %v", name, err)
			continue
		}
		if len(fakeJobControl.CreatedJobTemplates) != tc.expectedStartedJob {
			t.Errorf("%s: unexpected # of created jobs: expected %d got %d", name, tc.expectedStartedJob, len(fakeJobControl.CreatedJobTemplates))
			continue

		}
		if tc.checkWorkflow != nil {
			tc.checkWorkflow(name, actual, t)
		}
	}
}

func TestSyncWorkflowPastDeadline(t *testing.T) {
	testCases := map[string]struct {
		startTime             int64
		activeDeadlineSeconds int64
		workflow              *extensions.Workflow
		jobs                  []extensions.Job
		checkWorkflow         func(testName string, workflow *extensions.Workflow, t *testing.T)
	}{
		"activeDeadllineSeconds one": {
			startTime:             10,
			activeDeadlineSeconds: 5,
			workflow: &extensions.Workflow{
				ObjectMeta: api.ObjectMeta{
					Name:      "mydag",
					Namespace: api.NamespaceDefault,
					SelfLink:  "/apis/v1/workflows/mydag",
				},
				Spec: extensions.WorkflowSpec{
					Steps: map[string]extensions.WorkflowStep{
						"myJob": {
							JobTemplate: newJobTemplateSpec(),
						},
					},
				},
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{},
					Statuses: map[string]extensions.WorkflowStepStatus{
						"myJob": newJobTemplateStatus(),
					},
				},
			},
			jobs: []extensions.Job{
				{
					ObjectMeta: api.ObjectMeta{
						Name:      "foo",
						Namespace: api.NamespaceDefault,
						Labels: map[string]string{
							"foo": "bar",
							controller.WorkflowStepLabelKey: "myJob",
						},
						SelfLink: "/apis/v1/jobs/foo",
					},
					Spec:   extensions.JobSpec{},
					Status: extensions.JobStatus{},
				},
			},
			checkWorkflow: func(testName string, workflow *extensions.Workflow, t *testing.T) {
				if !isWorkflowFinished(workflow) {
					t.Errorf("%s, Workflow should be finished:\n %#v", testName, workflow)
				}
				if workflow.Status.CompletionTime == nil {
					t.Errorf("%s, CompletionTime not set", testName)
				}
			},
		},
	}
	for name, tc := range testCases {
		clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})
		oldClient := client.NewOrDie(&restclient.Config{Host: "", ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

		manager := NewWorkflowController(oldClient, clientset, controller.NoResyncPeriodFunc)
		fakeJobControl := controller.FakeJobControl{}
		manager.jobControl = &fakeJobControl
		manager.jobStoreSynced = func() bool { return true }
		var actual *extensions.Workflow
		manager.updateHandler = func(workflow *extensions.Workflow) error {
			actual = workflow
			return nil
		}
		startTime := unversioned.Unix(unversioned.Now().Time.Unix()-tc.startTime, 0)
		tc.workflow.Status.StartTime = &startTime
		tc.workflow.Spec.ActiveDeadlineSeconds = &tc.activeDeadlineSeconds
		manager.workflowStore.Store.Add(tc.workflow)
		for _, job := range tc.jobs {
			manager.jobStore.Store.Add(&job)
		}
		err := manager.syncWorkflow(getKey(tc.workflow, t))
		if err != nil {
			t.Errorf("%s: unexpected error syncing workflow %v", name, err)
			continue
		}
	}
}

func TestSyncWorkflowDelete(t *testing.T) {
	// @sdminonne: TODO
}

func TestWatchWorkflows(t *testing.T) {
	// @sdminonne: TODO
}

func TestIsWorkflowFinished(t *testing.T) {
	cases := []struct {
		name     string
		finished bool
		workflow *extensions.Workflow
	}{
		{
			name:     "Complete and True",
			finished: true,
			workflow: &extensions.Workflow{
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{
						{
							Type:   extensions.WorkflowComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
		},
		{
			name:     "Failed and True",
			finished: true,
			workflow: &extensions.Workflow{
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{
						{
							Type:   extensions.WorkflowFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
		},
		{
			name:     "Complete and False",
			finished: false,
			workflow: &extensions.Workflow{
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{
						{
							Type:   extensions.WorkflowComplete,
							Status: api.ConditionFalse,
						},
					},
				},
			},
		},
		{
			name:     "Failed and False",
			finished: false,
			workflow: &extensions.Workflow{
				Status: extensions.WorkflowStatus{
					Conditions: []extensions.WorkflowCondition{
						{
							Type:   extensions.WorkflowComplete,
							Status: api.ConditionFalse,
						},
					},
				},
			},
		},
	}

	for _, tc := range cases {
		if isWorkflowFinished(tc.workflow) != tc.finished {
			t.Errorf("%s - Expected %v got %v", tc.name, tc.finished, isWorkflowFinished(tc.workflow))
		}
	}
}

func TestWatchJobs(t *testing.T) {
	/* @sdminonne: TODO */
}
