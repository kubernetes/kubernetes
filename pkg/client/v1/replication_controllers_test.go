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

package v1

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/labels"
)

func getRCResourceName() string {
	return "replicationcontrollers"
}

func TestListControllers(t *testing.T) {
	ns := v1.NamespaceAll
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getRCResourceName(), ns, ""),
		},
		Response: Response{StatusCode: 200,
			Body: &v1.ReplicationControllerList{
				Items: []v1.ReplicationController{
					{
						ObjectMeta: v1.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: v1.ReplicationControllerSpec{
							Replicas: intPointer(2),
							Template: &v1.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedControllerList, err := c.Setup(t).ReplicationControllers(ns).List(labels.Everything())
	c.Validate(t, receivedControllerList, err)

}

func TestGetController(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &v1.ReplicationController{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: intPointer(2),
					Template: &v1.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).Get("foo")
	c.Validate(t, receivedController, err)
}

func TestGetControllerWithNoName(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup(t).ReplicationControllers(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateController(t *testing.T) {
	ns := v1.NamespaceDefault
	requestController := &v1.ReplicationController{
		ObjectMeta: v1.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &v1.ReplicationController{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: intPointer(2),
					Template: &v1.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).Update(requestController)
	c.Validate(t, receivedController, err)
}

func TestDeleteController(t *testing.T) {
	ns := v1.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).ReplicationControllers(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreateController(t *testing.T) {
	ns := v1.NamespaceDefault
	requestController := &v1.ReplicationController{
		ObjectMeta: v1.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, ""), Body: requestController, Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &v1.ReplicationController{
				ObjectMeta: v1.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Replicas: intPointer(2),
					Template: &v1.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).Create(requestController)
	c.Validate(t, receivedController, err)
}

func intPointer(x int) *int { return &x }
