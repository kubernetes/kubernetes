/*
Copyright 2014 The Kubernetes Authors.

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
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
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

	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   pv,
		},
		Response: simple.Response{StatusCode: 200, Body: pv},
	}

	response, err := c.Setup(t).PersistentVolumes().Create(pv)
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: persistentVolume},
	}

	response, err := c.Setup(t).PersistentVolumes().Get("abc")
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: persistentVolumeList},
	}
	response, err := c.Setup(t).PersistentVolumes().List(api.ListOptions{})
	defer c.Close()
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
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "abc"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: persistentVolume},
	}
	response, err := c.Setup(t).PersistentVolumes().Update(persistentVolume)
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "abc") + "/status",
			Query:  simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: persistentVolume},
	}
	response, err := c.Setup(t).PersistentVolumes().UpdateStatus(persistentVolume)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestPersistentVolumeDelete(t *testing.T) {
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).PersistentVolumes().Delete("foo")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestPersistentVolumeWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getPersistentVolumesResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).PersistentVolumes().Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}
