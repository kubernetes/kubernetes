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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getJobsResource(gv *unversioned.GroupVersion) unversioned.GroupVersionResource {
	return gv.WithResource("jobs")
}

func getJobClient(t *testing.T, c *simple.Client, ns string, gv *unversioned.GroupVersion) client.JobInterface {
	switch gv.Group {
	case batch.GroupName:
		return c.Setup(t).Batch().Jobs(ns)
	case extensions.GroupName:
		return c.Setup(t).Extensions().Jobs(ns)
	default:
		t.Fatalf("Unknown group %v", gv.Group)
	}
	return nil
}

func testListJob(t *testing.T, group testapi.TestGroup, gv *unversioned.GroupVersion) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   group.ResourcePath(getJobsResource(gv), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &batch.JobList{
				Items: []batch.Job{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: batch.JobSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
		ResourceGroup: gv.Group,
		GroupVersion:  gv,
	}
	receivedJobList, err := getJobClient(t, c, ns, gv).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedJobList, err)
}

func TestListJob(t *testing.T) {
	testListJob(t, testapi.Extensions, &v1beta1.SchemeGroupVersion)
	testListJob(t, testapi.Batch, &v1.SchemeGroupVersion)
	testListJob(t, testapi.Batch, &v2alpha1.SchemeGroupVersion)
}

func testGetJob(t *testing.T, group testapi.TestGroup, gv *unversioned.GroupVersion) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   group.ResourcePath(getJobsResource(gv), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
		ResourceGroup: gv.Group,
		GroupVersion:  gv,
	}
	receivedJob, err := getJobClient(t, c, ns, gv).Get("foo")
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestGetJob(t *testing.T) {
	testGetJob(t, testapi.Extensions, &v1beta1.SchemeGroupVersion)
	testGetJob(t, testapi.Batch, &v1.SchemeGroupVersion)
	testGetJob(t, testapi.Batch, &v2alpha1.SchemeGroupVersion)
}

func testUpdateJob(t *testing.T, group testapi.TestGroup, gv *unversioned.GroupVersion) {
	ns := api.NamespaceDefault
	requestJob := &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   group.ResourcePath(getJobsResource(gv), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
		ResourceGroup: gv.Group,
		GroupVersion:  gv,
	}
	receivedJob, err := getJobClient(t, c, ns, gv).Update(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestUpdateJob(t *testing.T) {
	testUpdateJob(t, testapi.Extensions, &v1beta1.SchemeGroupVersion)
	testUpdateJob(t, testapi.Batch, &v1.SchemeGroupVersion)
	testUpdateJob(t, testapi.Batch, &v2alpha1.SchemeGroupVersion)
}

func testUpdateJobStatus(t *testing.T, group testapi.TestGroup, gv *unversioned.GroupVersion) {
	ns := api.NamespaceDefault
	requestJob := &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   group.ResourcePath(getJobsResource(gv), ns, "foo") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{},
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
		},
		ResourceGroup: gv.Group,
		GroupVersion:  gv,
	}
	receivedJob, err := getJobClient(t, c, ns, gv).UpdateStatus(requestJob)
	defer c.Close()
	c.Validate(t, receivedJob, err)
}

func TestUpdateJobStatus(t *testing.T) {
	testUpdateJobStatus(t, testapi.Extensions, &v1beta1.SchemeGroupVersion)
	testUpdateJobStatus(t, testapi.Batch, &v1.SchemeGroupVersion)
	testUpdateJobStatus(t, testapi.Batch, &v2alpha1.SchemeGroupVersion)
}

func testDeleteJob(t *testing.T, group testapi.TestGroup, gv *unversioned.GroupVersion) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   group.ResourcePath(getJobsResource(gv), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response:      simple.Response{StatusCode: 200},
		ResourceGroup: gv.Group,
		GroupVersion:  gv,
	}
	err := getJobClient(t, c, ns, gv).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestDeleteJob(t *testing.T) {
	testDeleteJob(t, testapi.Extensions, &v1beta1.SchemeGroupVersion)
	testDeleteJob(t, testapi.Batch, &v1.SchemeGroupVersion)
	testDeleteJob(t, testapi.Batch, &v2alpha1.SchemeGroupVersion)
}

func testCreateJob(t *testing.T, group testapi.TestGroup, gv *unversioned.GroupVersion) {
	ns := api.NamespaceDefault
	requestJob := &batch.Job{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   group.ResourcePath(getJobsResource(gv), ns, ""),
			Body:   requestJob,
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{
			StatusCode: 200,
			Body: &batch.Job{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: batch.JobSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
		ResourceGroup: gv.Group,
		GroupVersion:  gv,
	}
	receivedJob, err := getJobClient(t, c, ns, gv).Create(requestJob)
	defer c.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, receivedJob, err)
}

func TestCreateJob(t *testing.T) {
	testCreateJob(t, testapi.Extensions, &v1beta1.SchemeGroupVersion)
	testCreateJob(t, testapi.Batch, &v1.SchemeGroupVersion)
	testCreateJob(t, testapi.Batch, &v2alpha1.SchemeGroupVersion)
}
