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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getPersistentVolumeSetsResoureName() string {
	return "persistentvolumesets"
}

func TestPersistentVolumeSetCreate(t *testing.T) {
	pvctl := &api.PersistentVolumeSet{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.PersistentVolumeSetSpec{
			MinimumReplicas: 5,
			MaximumReplicas: 10,
			Selector:        map[string]string{"a": "b"},
			Template: &api.PersistentVolumeTemplateSpec{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/foo"},
					},
				},
			},
		},
	}

	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeSetsResoureName(), "", ""),
			Query:  buildQueryValues(nil),
			Body:   pvctl,
		},
		Response: Response{StatusCode: 200, Body: pvctl},
	}

	response, err := c.Setup(t).PersistentVolumeSets().Create(pvctl)
	c.Validate(t, response, err)
}

func TestPersistentVolumeSetGet(t *testing.T) {
	pvctl := &api.PersistentVolumeSet{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.PersistentVolumeSetSpec{
			MinimumReplicas: 5,
			MaximumReplicas: 10,
			Selector:        map[string]string{"a": "b"},
			Template: &api.PersistentVolumeTemplateSpec{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/foo"},
					},
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeSetsResoureName(), "", "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: pvctl},
	}

	response, err := c.Setup(t).PersistentVolumeSets().Get("abc")
	c.Validate(t, response, err)
}

func TestPersistentVolumeSetList(t *testing.T) {
	persistentVolumeList := &api.PersistentVolumeSetList{
		Items: []api.PersistentVolumeSet{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeSetsResoureName(), "", ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolumeList},
	}
	response, err := c.Setup(t).PersistentVolumeSets().List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)
}

func TestPersistentVolumeSetUpdate(t *testing.T) {
	pvctl := &api.PersistentVolumeSet{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: api.PersistentVolumeSetSpec{
			MinimumReplicas: 5,
			MaximumReplicas: 10,
			Selector:        map[string]string{"a": "b"},
			Template: &api.PersistentVolumeTemplateSpec{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/foo"},
					},
				},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath(getPersistentVolumeSetsResoureName(), "", "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: pvctl},
	}
	response, err := c.Setup(t).PersistentVolumeSets().Update(pvctl)
	c.Validate(t, response, err)
}

func TestPersistentVolumeSetStatusUpdate(t *testing.T) {
	pvctl := &api.PersistentVolumeSet{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.PersistentVolumeSetSpec{
			MinimumReplicas: 5,
			MaximumReplicas: 10,
			Selector:        map[string]string{"a": "b"},
			Template: &api.PersistentVolumeTemplateSpec{
				Spec: api.PersistentVolumeSpec{
					PersistentVolumeSource: api.PersistentVolumeSource{
						HostPath: &api.HostPathVolumeSource{Path: "/foo"},
					},
				},
			},
		},
		Status: api.PersistentVolumeSetStatus{
			BoundReplicas:     5,
			AvailableReplicas: 5,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeSetsResoureName(), "", "abc") + "/status",
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: pvctl},
	}
	response, err := c.Setup(t).PersistentVolumeSets().UpdateStatus(pvctl)
	c.Validate(t, response, err)
}

func TestPersistentVolumeSetDelete(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath(getPersistentVolumeSetsResoureName(), "", "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).PersistentVolumeSets().Delete("foo")
	c.Validate(t, nil, err)
}

func TestPersistentVolumeSetWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/api/" + testapi.Default.Version() + "/watch/" + getPersistentVolumeSetsResoureName(),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).PersistentVolumeSets().Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
