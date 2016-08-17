/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getRCResourceName() string {
	return "replicationcontrollers"
}

func TestListControllers(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getRCResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &api.ReplicationControllerList{
				Items: []api.ReplicationController{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: api.ReplicationControllerSpec{
							Replicas: 2,
							Template: &api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedControllerList, err := c.Setup(t).ReplicationControllers(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedControllerList, err)

}

func TestGetController(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).Get("foo")
	defer c.Close()
	c.Validate(t, receivedController, err)
}

func TestGetControllerWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).ReplicationControllers(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateController(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).Update(requestController)
	defer c.Close()
	c.Validate(t, receivedController, err)
}

func TestUpdateStatusController(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo") + "/status", Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
				Status: api.ReplicationControllerStatus{
					Replicas: 2,
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).UpdateStatus(requestController)
	defer c.Close()
	c.Validate(t, receivedController, err)
}
func TestDeleteController(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).ReplicationControllers(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateController(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Default.ResourcePath(getRCResourceName(), ns, ""), Body: requestController, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &api.ReplicationController{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: api.ReplicationControllerSpec{
					Replicas: 2,
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup(t).ReplicationControllers(ns).Create(requestController)
	defer c.Close()
	c.Validate(t, receivedController, err)
}
