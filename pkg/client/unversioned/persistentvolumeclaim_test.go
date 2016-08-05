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

	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   pv,
		},
		Response: simple.Response{StatusCode: 200, Body: pv},
	}

	response, err := c.Setup(t).PersistentVolumeClaims(ns).Create(pv)
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: persistentVolumeClaim},
	}

	response, err := c.Setup(t).PersistentVolumeClaims(ns).Get("abc")
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: persistentVolumeList},
	}
	response, err := c.Setup(t).PersistentVolumeClaims(ns).List(api.ListOptions{})
	defer c.Close()
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
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "abc"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: persistentVolumeClaim},
	}
	response, err := c.Setup(t).PersistentVolumeClaims(ns).Update(persistentVolumeClaim)
	defer c.Close()
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
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Default.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "abc") + "/status",
			Query:  simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: persistentVolumeClaim},
	}
	response, err := c.Setup(t).PersistentVolumeClaims(ns).UpdateStatus(persistentVolumeClaim)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestPersistentVolumeClaimDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath(getPersistentVolumeClaimsResoureName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).PersistentVolumeClaims(ns).Delete("foo")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestPersistentVolumeClaimWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getPersistentVolumeClaimsResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).PersistentVolumeClaims(api.NamespaceAll).Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}
