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
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getDMResourceName() string {
	return "dedicatedmachines"
}

func TestListDedicatedMachines(t *testing.T) {
	ns := api.NamespaceAll
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getDMResourceName(), ns, ""),
		},
		Response: Response{StatusCode: 200,
			Body: &extensions.DedicatedMachineList{
				Items: []extensions.DedicatedMachine{
					{
						ObjectMeta: api.ObjectMeta{
							Name:      "foo",
							Namespace: api.NamespaceDefault,
						},
						Spec: extensions.DedicatedMachineSpec{
							LabelValue: "bar",
						},
					},
				},
			},
		},
	}
	receivedDSs, err := c.Setup(t).Extensions().DedicatedMachines(ns).List(labels.Everything(), fields.Everything(), unversioned.ListOptions{})
	c.Validate(t, receivedDSs, err)

}

func TestGetDedicatedMachine(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.Extensions.ResourcePath(getDMResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &extensions.DedicatedMachine{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.DedicatedMachineSpec{
					LabelValue: "bar",
				},
			},
		},
	}
	receivedDedicatedMachine, err := c.Setup(t).Extensions().DedicatedMachines(ns).Get("foo")
	c.Validate(t, receivedDedicatedMachine, err)
}

func TestGetDedicatedMachineWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup(t).Extensions().DedicatedMachines(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateDedicatedMachine(t *testing.T) {
	ns := api.NamespaceDefault
	requestDedicatedMachine := &extensions.DedicatedMachine{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: testapi.Extensions.ResourcePath(getDMResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &extensions.DedicatedMachine{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.DedicatedMachineSpec{
					LabelValue: "bar",
				},
			},
		},
	}
	receivedDedicatedMachine, err := c.Setup(t).Extensions().DedicatedMachines(ns).Update(requestDedicatedMachine)
	c.Validate(t, receivedDedicatedMachine, err)
}

func TestDeleteDedicatedMachine(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Extensions.ResourcePath(getDMResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Extensions().DedicatedMachines(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreateDedicatedMachine(t *testing.T) {
	ns := api.NamespaceDefault
	requestDedicatedMachine := &extensions.DedicatedMachine{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: testapi.Extensions.ResourcePath(getDMResourceName(), ns, ""), Body: requestDedicatedMachine, Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &extensions.DedicatedMachine{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: api.NamespaceDefault,
				},
				Spec: extensions.DedicatedMachineSpec{
					LabelValue: "bar",
				},
			},
		},
	}
	receivedDedicatedMachine, err := c.Setup(t).Extensions().DedicatedMachines(ns).Create(requestDedicatedMachine)
	c.Validate(t, receivedDedicatedMachine, err)
}

func TestDedicatedMachineWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePathWithPrefix("watch", getDMResourceName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}},
		},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).DedicatedMachines(api.NamespaceAll).Watch(unversioned.ListOptions{})
	c.Validate(t, nil, err)
}
