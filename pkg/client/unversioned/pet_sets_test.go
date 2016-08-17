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

func getPetSetResourceName() string {
	return "petsets"
}

func TestListPetSets(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Apps.ResourcePath(getPetSetResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &apps.PetSetList{
				Items: []apps.PetSet{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: apps.PetSetSpec{
							Replicas: 2,
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedRSList, err := c.Setup(t).Apps().PetSets(ns).List(api.ListOptions{})
	c.Validate(t, receivedRSList, err)
}

func TestGetPetSet(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Apps.ResourcePath(getPetSetResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &apps.PetSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: apps.PetSetSpec{
					Replicas: 2,
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedRS, err := c.Setup(t).Apps().PetSets(ns).Get("foo")
	c.Validate(t, receivedRS, err)
}

func TestGetPetSetWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Apps().PetSets(ns).Get("")
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdatePetSet(t *testing.T) {
	ns := api.NamespaceDefault
	requestRS := &apps.PetSet{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Apps.ResourcePath(getPetSetResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &apps.PetSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: apps.PetSetSpec{
					Replicas: 2,
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedRS, err := c.Setup(t).Apps().PetSets(ns).Update(requestRS)
	c.Validate(t, receivedRS, err)
}

func TestDeletePetSet(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Apps.ResourcePath(getPetSetResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Apps().PetSets(ns).Delete("foo", nil)
	c.Validate(t, nil, err)
}

func TestCreatePetSet(t *testing.T) {
	ns := api.NamespaceDefault
	requestRS := &apps.PetSet{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Apps.ResourcePath(getPetSetResourceName(), ns, ""), Body: requestRS, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &apps.PetSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: apps.PetSetSpec{
					Replicas: 2,
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedRS, err := c.Setup(t).Apps().PetSets(ns).Create(requestRS)
	c.Validate(t, receivedRS, err)
}

// TODO: Test Status actions.
