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
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getStatefulSetResourceName() string {
	return "statefulsets"
}

func TestListStatefulSets(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Apps.ResourcePath(getStatefulSetResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &apps.StatefulSetList{
				Items: []apps.StatefulSet{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: apps.StatefulSetSpec{
							Replicas: 2,
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedRSList, err := c.Setup(t).Apps().StatefulSets(ns).List(api.ListOptions{})
	c.Validate(t, receivedRSList, err)
}

func TestGetStatefulSet(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Apps.ResourcePath(getStatefulSetResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &apps.StatefulSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: apps.StatefulSetSpec{
					Replicas: 2,
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedRS, err := c.Setup(t).Apps().StatefulSets(ns).Get("foo")
	c.Validate(t, receivedRS, err)
}

func TestGetStatefulSetWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Apps().StatefulSets(ns).Get("")
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateStatefulSet(t *testing.T) {
	ns := api.NamespaceDefault
	requestRS := &apps.StatefulSet{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Apps.ResourcePath(getStatefulSetResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &apps.StatefulSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: apps.StatefulSetSpec{
					Replicas: 2,
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedRS, err := c.Setup(t).Apps().StatefulSets(ns).Update(requestRS)
	c.Validate(t, receivedRS, err)
}

func TestDeleteStatefulSet(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Apps.ResourcePath(getStatefulSetResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Apps().StatefulSets(ns).Delete("foo", nil)
	c.Validate(t, nil, err)
}

func TestCreateStatefulSet(t *testing.T) {
	ns := api.NamespaceDefault
	requestRS := &apps.StatefulSet{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Apps.ResourcePath(getStatefulSetResourceName(), ns, ""), Body: requestRS, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &apps.StatefulSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: apps.StatefulSetSpec{
					Replicas: 2,
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedRS, err := c.Setup(t).Apps().StatefulSets(ns).Create(requestRS)
	c.Validate(t, receivedRS, err)
}

// TODO: Test Status actions.
