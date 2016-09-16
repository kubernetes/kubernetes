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
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func getDSResourceName() string {
	return "daemonsets"
}

func TestListDaemonSets(t *testing.T) {
	ns := api.NamespaceAll
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getDSResourceName(), ns, ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &extensions.DaemonSetList{
				Items: []extensions.DaemonSet{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Spec: extensions.DaemonSetSpec{
							Template: api.PodTemplateSpec{},
						},
					},
				},
			},
		},
	}
	receivedDSs, err := c.Setup(t).Extensions().DaemonSets(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedDSs, err)

}

func TestGetDaemonSet(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Extensions.ResourcePath(getDSResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.DaemonSetSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedDaemonSet, err := c.Setup(t).Extensions().DaemonSets(ns).Get("foo")
	defer c.Close()
	c.Validate(t, receivedDaemonSet, err)
}

func TestGetDaemonSetWithNoName(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Extensions().DaemonSets(ns).Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateDaemonSet(t *testing.T) {
	ns := api.NamespaceDefault
	requestDaemonSet := &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Extensions.ResourcePath(getDSResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.DaemonSetSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedDaemonSet, err := c.Setup(t).Extensions().DaemonSets(ns).Update(requestDaemonSet)
	defer c.Close()
	c.Validate(t, receivedDaemonSet, err)
}

func TestUpdateDaemonSetUpdateStatus(t *testing.T) {
	ns := api.NamespaceDefault
	requestDaemonSet := &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Extensions.ResourcePath(getDSResourceName(), ns, "foo") + "/status", Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.DaemonSetSpec{
					Template: api.PodTemplateSpec{},
				},
				Status: extensions.DaemonSetStatus{},
			},
		},
	}
	receivedDaemonSet, err := c.Setup(t).Extensions().DaemonSets(ns).UpdateStatus(requestDaemonSet)
	defer c.Close()
	c.Validate(t, receivedDaemonSet, err)
}

func TestDeleteDaemon(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Extensions.ResourcePath(getDSResourceName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Extensions().DaemonSets(ns).Delete("foo")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateDaemonSet(t *testing.T) {
	ns := api.NamespaceDefault
	requestDaemonSet := &extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Extensions.ResourcePath(getDSResourceName(), ns, ""), Body: requestDaemonSet, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.DaemonSet{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Spec: extensions.DaemonSetSpec{
					Template: api.PodTemplateSpec{},
				},
			},
		},
	}
	receivedDaemonSet, err := c.Setup(t).Extensions().DaemonSets(ns).Create(requestDaemonSet)
	defer c.Close()
	c.Validate(t, receivedDaemonSet, err)
}
