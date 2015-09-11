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
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/expapi/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getLocksResourceName() string {
	return "locks"
}

func TestLockCreate(t *testing.T) {
	ns := api.NamespaceDefault
	lock := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: "aprocess",
		},
		Spec: expapi.LockSpec{
			HeldBy:       "app1",
			LeaseSeconds: 30,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath(getLocksResourceName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   lock,
		},
		Response: Response{StatusCode: 200, Body: lock},
	}

	response, err := c.Setup(t).Experimental().Locks(ns).Create(lock)
	c.Validate(t, response, err)
}

func TestLockGet(t *testing.T) {
	name := "aprocess"
	ns := api.NamespaceDefault
	lock := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: expapi.LockSpec{
			HeldBy:       "app1",
			LeaseSeconds: 30,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getLocksResourceName(), ns, name),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: lock},
	}

	response, err := c.Setup(t).Experimental().Locks(ns).Get(name)
	c.Validate(t, response, err)
}

func TestLockList(t *testing.T) {
	ns := api.NamespaceDefault

	lockList := &expapi.LockList{
		Items: []expapi.Lock{
			{
				ObjectMeta: api.ObjectMeta{Name: "aprocess"},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getLocksResourceName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: lockList},
	}
	response, err := c.Setup(t).Experimental().Locks(ns).List(labels.Everything())
	c.Validate(t, response, err)
}

func TestLockUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	lock := &expapi.Lock{
		ObjectMeta: api.ObjectMeta{
			Name: "aprocess",
		},
		Spec: expapi.LockSpec{
			HeldBy:       "app1",
			LeaseSeconds: 30,
		},
	}
	c := &testClient{
		Request:  testRequest{Method: "PUT", Path: testapi.ResourcePath(getLocksResourceName(), ns, "aprocess"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200, Body: lock},
	}
	response, err := c.Setup(t).Experimental().Locks(ns).Update(lock)
	c.Validate(t, response, err)
}

func TestLockDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request:  testRequest{Method: "DELETE", Path: testapi.ResourcePath(getLocksResourceName(), ns, "aprocess"), Query: buildQueryValues(nil)},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Experimental().Locks(ns).Delete("aprocess")
	c.Validate(t, nil, err)
}

func TestLockWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePathWithPrefix("watch", getLocksResourceName(), api.NamespaceAll, ""),
			Query:  url.Values{"name": []string{}}},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).Experimental().Locks(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
