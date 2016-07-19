/*
Copyright 2016 The Kubernetes Authors.

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

package unversioned_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getScheduledJobsResource() string {
	return "scheduledjobs"
}

func TestListScheduledJob(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Batch.ResourcePath(getScheduledJobsResource(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &batch.ScheduledJobList{
				Items: []batch.ScheduledJob{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: batch.ScheduledJobSpec{
							JobTemplate: batch.JobTemplateSpec{
								Spec: batch.JobSpec{
									Template: api.PodTemplateSpec{},
								},
							},
						},
					},
				},
			},
		},
		ResourceGroup: batch.GroupName,
	}
	receivedScheduledJobList, err := c.Setup(t).Batch().ScheduledJobs(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedScheduledJobList, err)
}

func TestGetScheduledJob(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Batch.ResourcePath(getScheduledJobsResource(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.ScheduledJob{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.ScheduledJobSpec{
					JobTemplate: batch.JobTemplateSpec{
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
		ResourceGroup: batch.GroupName,
	}
	receivedScheduledJob, err := c.Setup(t).Batch().ScheduledJobs(ns).Get("foo")
	defer c.Close()
	c.Validate(t, receivedScheduledJob, err)
}

func TestUpdateScheduledJob(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	ns := api.NamespaceDefault
	requestScheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Batch.ResourcePath(getScheduledJobsResource(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.ScheduledJob{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.ScheduledJobSpec{
					JobTemplate: batch.JobTemplateSpec{
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
		ResourceGroup: batch.GroupName,
	}
	receivedScheduledJob, err := c.Setup(t).Batch().ScheduledJobs(ns).Update(requestScheduledJob)
	defer c.Close()
	c.Validate(t, receivedScheduledJob, err)
}

func TestUpdateScheduledJobStatus(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	ns := api.NamespaceDefault
	requestScheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Batch.ResourcePath(getScheduledJobsResource(), ns, "foo") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.ScheduledJob{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.ScheduledJobSpec{
					ConcurrencyPolicy: batch.AllowConcurrent,
					JobTemplate: batch.JobTemplateSpec{
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
				Status: batch.ScheduledJobStatus{
					Active: []api.ObjectReference{{Name: "ref"}},
				},
			},
		},
		ResourceGroup: batch.GroupName,
	}
	receivedScheduledJob, err := c.Setup(t).Batch().ScheduledJobs(ns).UpdateStatus(requestScheduledJob)
	defer c.Close()
	c.Validate(t, receivedScheduledJob, err)
}

func TestDeleteScheduledJob(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   testapi.Batch.ResourcePath(getScheduledJobsResource(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response:      simple.Response{StatusCode: 200},
		ResourceGroup: batch.GroupName,
	}
	err := c.Setup(t).Batch().ScheduledJobs(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateScheduledJob(t *testing.T) {
	// scheduled jobs should be tested only when batch/v2alpha1 is enabled
	if *testapi.Batch.GroupVersion() != v2alpha1.SchemeGroupVersion {
		return
	}

	ns := api.NamespaceDefault
	requestScheduledJob := &batch.ScheduledJob{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Batch.ResourcePath(getScheduledJobsResource(), ns, ""),
			Body:   requestScheduledJob,
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.ScheduledJob{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.ScheduledJobSpec{
					JobTemplate: batch.JobTemplateSpec{
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
		ResourceGroup: batch.GroupName,
	}
	receivedScheduledJob, err := c.Setup(t).Batch().ScheduledJobs(ns).Create(requestScheduledJob)
	defer c.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, receivedScheduledJob, err)
}
