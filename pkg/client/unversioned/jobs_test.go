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

package unversioned

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/labels"
)

func getJobResourceName() string {
	return "jobs"
}

func TestListJobs(t *testing.T) {
	ns := api.NamespaceAll
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getJobResourceName(), ns, ""),
		},
		Response: Response{StatusCode: 200,
			Body: &expapi.JobList{
				Items: []expapi.Job{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: expapi.JobSpec{
							Template: &api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedJobList, err := c.Setup().Jobs(ns).List(labels.Everything())
	c.Validate(t, receivedJobList, err)
}

func TestGetJob(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.ResourcePath(getJobResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &expapi.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: expapi.JobSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup().Jobs(ns).Get("foo")
	c.Validate(t, receivedJob, err)
}

func TestGetJobWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedJob, err := c.Setup().Jobs(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedJob, err)
}

func TestUpdateJob(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &expapi.Job{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: testapi.ResourcePath(getJobResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &expapi.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: expapi.JobSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup().Jobs(ns).Update(requestJob)
	c.Validate(t, receivedJob, err)
}

func TestDeleteJob(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getJobResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Jobs(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreateJob(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &expapi.Job{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: testapi.ResourcePath(getJobResourceName(), ns, ""), Body: requestJob, Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &expapi.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: expapi.JobSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup().Jobs(ns).Create(requestJob)
	c.Validate(t, receivedJob, err)
}
