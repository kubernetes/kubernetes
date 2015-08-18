/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package client

import (
	"net/url"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func getPersistentVolumesResoureName() string {
	return "persistentvolumes"
}

func TestPersistentVolumeCreate(t *testing.T) {
	pv := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}

	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath(getPersistentVolumesResoureName(), "", ""),
			Query:  buildQueryValues("", nil),
			Body:   pv,
		},
		Response: Response{StatusCode: 200, Body: pv},
	}

	response, err := c.Setup().PersistentVolumes().Create(pv)
	c.Validate(t, response, err)
}

func TestPersistentVolumeGet(t *testing.T) {
	persistentVolume := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getPersistentVolumesResoureName(), "", "abc"),
			Query:  buildQueryValues("", nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolume},
	}

	response, err := c.Setup().PersistentVolumes().Get("abc")
	c.Validate(t, response, err)
}

func TestPersistentVolumeList(t *testing.T) {
	persistentVolumeList := &api.PersistentVolumeList{
		Items: []api.PersistentVolume{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getPersistentVolumesResoureName(), "", ""),
			Query:  buildQueryValues("", nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolumeList},
	}
	response, err := c.Setup().PersistentVolumes().List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)
}

func TestPersistentVolumeUpdate(t *testing.T) {
	persistentVolume := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.ResourcePath(getPersistentVolumesResoureName(), "", "abc"), Query: buildQueryValues("", nil)},
		Response: Response{StatusCode: 200, Body: persistentVolume},
	}
	response, err := c.Setup().PersistentVolumes().Update(persistentVolume)
	c.Validate(t, response, err)
}

func TestPersistentVolumeStatusUpdate(t *testing.T) {
	persistentVolume := &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{Path: "/foo"},
			},
		},
		Status: api.PersistentVolumeStatus{
			Phase:   api.VolumeBound,
			Message: "foo",
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.ResourcePath(getPersistentVolumesResoureName(), "", "abc") + "/status",
			Query:  buildQueryValues("", nil)},
		Response: Response{StatusCode: 200, Body: persistentVolume},
	}
	response, err := c.Setup().PersistentVolumes().UpdateStatus(persistentVolume)
	c.Validate(t, response, err)
}

func TestPersistentVolumeDelete(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getPersistentVolumesResoureName(), "", "foo"), Query: buildQueryValues("", nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().PersistentVolumes().Delete("foo")
	c.Validate(t, nil, err)
}

func TestPersistentVolumeWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/api/" + testapi.Version() + "/watch/" + getPersistentVolumesResoureName(),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup().PersistentVolumes().Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
