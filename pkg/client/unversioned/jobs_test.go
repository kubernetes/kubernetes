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

package unversioned_test

import (
	. "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func getJobResourceName() string {
	return "jobs"
}

func TestListJobsViaExtensions(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getJobResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &extensions.JobList{
				Items: []extensions.Job{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: extensions.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedJobList, err := c.Setup(t).Extensions().Jobs(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedJobList, err)
}

func TestGetJobViaExtensions(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getJobResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Extensions().Jobs(ns).Get("foo")
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestGetJobWithNoNameViaExtensions(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedJob, err := c.Setup(t).Extensions().Jobs(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedJob, err)
}

func TestUpdateJobViaExtensions(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Extensions.ResourcePath(getJobResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Extensions().Jobs(ns).Update(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestUpdateJobStatusViaExtensions(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Extensions.ResourcePath(getJobResourceName(), ns, "foo") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
				Status: extensions.JobStatus{
					Active: 1,
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Extensions().Jobs(ns).UpdateStatus(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestDeleteJobViaExtensions(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   testapi.Extensions.ResourcePath(getJobResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Extensions().Jobs(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateJobViaExtensions(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Extensions.ResourcePath(getJobResourceName(), ns, ""),
			Body:   requestJob,
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Extensions().Jobs(ns).Create(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

// Tests below are a copy of the above tests.  Once job is removed from extensions, above test cases can be deleted.

func TestListJobsViaBatch(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Batch.ResourcePath(getJobResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &extensions.JobList{
				Items: []extensions.Job{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: extensions.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedJobList, err := c.Setup(t).Batch().Jobs(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedJobList, err)
}

func TestGetJobViaBatch(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Batch.ResourcePath(getJobResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Batch().Jobs(ns).Get("foo")
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestGetJobWithNoNameViaBatch(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedJob, err := c.Setup(t).Batch().Jobs(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedJob, err)
}

func TestUpdateJobViaBatch(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Batch.ResourcePath(getJobResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Batch().Jobs(ns).Update(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestUpdateJobStatusViaBatch(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Batch.ResourcePath(getJobResourceName(), ns, "foo") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
				Status: extensions.JobStatus{
					Active: 1,
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Batch().Jobs(ns).UpdateStatus(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestDeleteJobViaBatch(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   testapi.Batch.ResourcePath(getJobResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Batch().Jobs(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateJobViaBatch(t *testing.T) {
	ns := api.NamespaceDefault
	requestJob := &extensions.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Batch.ResourcePath(getJobResourceName(), ns, ""),
			Body:   requestJob,
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedJob, err := c.Setup(t).Batch().Jobs(ns).Create(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}
