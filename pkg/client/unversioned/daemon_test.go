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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/testapi"
	"k8s.io/kubernetes/pkg/labels"
)

func getDCResourceName() string {
	return "daemons"
}

func TestListDaemons(t *testing.T) {
	ns := api.NamespaceAll
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getDCResourceName(), ns, ""),
		},
		Response: Response{StatusCode: 200,
			Body: &expapi.DaemonList{
				Items: []expapi.Daemon{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: expapi.DaemonSpec{
							Template: &api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedControllerList, err := c.Setup().Daemons(ns).List(labels.Everything())
	c.Validate(t, receivedControllerList, err)

}

func TestGetDaemon(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{Method: "GET", Path: testapi.ResourcePath(getDCResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &expapi.Daemon{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: expapi.DaemonSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup().Daemons(ns).Get("foo")
	c.Validate(t, receivedController, err)
}

func TestGetDaemonWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().Daemons(ns).Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateDaemon(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &expapi.Daemon{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &testClient{
		Request: testRequest{Method: "PUT", Path: testapi.ResourcePath(getDCResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &expapi.Daemon{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: expapi.DaemonSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup().Daemons(ns).Update(requestController)
	c.Validate(t, receivedController, err)
}

func TestDeleteDaemon(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getDCResourceName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Daemons(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestCreateDaemon(t *testing.T) {
	ns := api.NamespaceDefault
	requestController := &expapi.Daemon{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &testClient{
		Request: testRequest{Method: "POST", Path: testapi.ResourcePath(getDCResourceName(), ns, ""), Body: requestController, Query: buildQueryValues(nil)},
		Response: Response{
			StatusCode: 200,
			Body: &expapi.Daemon{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: expapi.DaemonSpec{
					Template: &api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedController, err := c.Setup().Daemons(ns).Create(requestController)
	c.Validate(t, receivedController, err)
}
