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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getPersistentVolumeClaimsResoureName() string {
	return "persistentvolumeclaims"
}

func TestPersistentVolumeClaimCreate(t *testing.T) {
	ns := api.NamespaceDefault
	pv := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name: "abc",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
	}

	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   pv,
		},
		Response: Response{StatusCode: 200, Body: pv},
	}

	response, err := c.Setup().PersistentVolumeClaims(ns).Create(pv)
	c.Validate(t, response, err)
}

func TestPersistentVolumeClaimGet(t *testing.T) {
	ns := api.NamespaceDefault
	persistentVolumeClaim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolumeClaim},
	}

	response, err := c.Setup().PersistentVolumeClaims(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestPersistentVolumeClaimList(t *testing.T) {
	ns := api.NamespaceDefault
	persistentVolumeList := &api.PersistentVolumeClaimList{
		Items: []api.PersistentVolumeClaim{
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "ns"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: persistentVolumeList},
	}
	response, err := c.Setup().PersistentVolumeClaims(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)
}

func TestPersistentVolumeClaimUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	persistentVolumeClaim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "abc"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: persistentVolumeClaim},
	}
	response, err := c.Setup().PersistentVolumeClaims(ns).Update(persistentVolumeClaim)
	c.Validate(t, response, err)
}

func TestPersistentVolumeClaimStatusUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	persistentVolumeClaim := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			ResourceVersion: "1",
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
			},
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "abc") + "/status",
			Query:  buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: persistentVolumeClaim},
	}
	response, err := c.Setup().PersistentVolumeClaims(ns).UpdateStatus(persistentVolumeClaim)
	c.Validate(t, response, err)
}

func TestPersistentVolumeClaimDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "foo"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().PersistentVolumeClaims(ns).Delete("foo")
	c.Validate(t, nil, err)
}

func TestPersistentVolumeClaimWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   "/api/" + testapi.Version() + "/watch/" + getPersistentVolumeClaimsResoureName(),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup().PersistentVolumeClaims(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
