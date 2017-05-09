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

package v1

import (
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getPersistentVolumesResoureName() string {
	return "persistentvolumes"
}

func TestPersistentVolumeCreate(t *testing.T) {
	pv := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{
			Name: "abc",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}

	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", ""),
			Query:  buildQueryValues(nil),
			Body:   pv,
		},
		Response: Response{StatusCode: 200, Body: pv},
	}

	response, err := c.Setup(t).PersistentVolumes().Create(pv)
	c.Validate(t, response, err)
}

func TestPersistentVolumeGet(t *testing.T) {
	persistentVolume := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolume},
	}

	response, err := c.Setup(t).PersistentVolumes().Get("abc")
	c.Validate(t, response, err)
}

func TestPersistentVolumeList(t *testing.T) {
	persistentVolumeList := &v1.PersistentVolumeList{
		Items: []v1.PersistentVolume{
			{
				ObjectMeta: v1.ObjectMeta{Name: "foo"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolumeList},
	}
	response, err := c.Setup(t).PersistentVolumes().List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)
}

func TestPersistentVolumeUpdate(t *testing.T) {
	persistentVolume := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/foo"},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: persistentVolume},
	}
	response, err := c.Setup(t).PersistentVolumes().Update(persistentVolume)
	c.Validate(t, response, err)
}

func TestPersistentVolumeStatusUpdate(t *testing.T) {
	persistentVolume := &v1.PersistentVolume{
		ObjectMeta: v1.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("10G"),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "/foo"},
			},
		},
		Status: v1.PersistentVolumeStatus{
			Phase:   v1.VolumeBound,
			Message: "foo",
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "abc") + "/status",
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: persistentVolume},
	}
	response, err := c.Setup(t).PersistentVolumes().UpdateStatus(persistentVolume)
	c.Validate(t, response, err)
}

func TestPersistentVolumeDelete(t *testing.T) {
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.Default.ResourcePath(getPersistentVolumesResoureName(), "", "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).PersistentVolumes().Delete("foo")
	c.Validate(t, nil, err)
}

func TestPersistentVolumeWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getPersistentVolumesResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).PersistentVolumes().Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
