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
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getPodTemplatesResoureName() string {
	return "podtemplates"
}

func TestPodTemplateCreate(t *testing.T) {
	ns := api.NamespaceDefault
	podTemplate := api.PodTemplate{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
		Template: api.PodTemplateSpec{},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Default.ResourcePath(getPodTemplatesResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   &podTemplate,
		},
		Response: simple.Response{StatusCode: 200, Body: &podTemplate},
	}

	response, err := c.Setup(t).PodTemplates(ns).Create(&podTemplate)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestPodTemplateGet(t *testing.T) {
	ns := api.NamespaceDefault
	podTemplate := &api.PodTemplate{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
		Template: api.PodTemplateSpec{},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPodTemplatesResoureName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: podTemplate},
	}

	response, err := c.Setup(t).PodTemplates(ns).Get("abc")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestPodTemplateList(t *testing.T) {
	ns := api.NamespaceDefault
	podTemplateList := &api.PodTemplateList{
		Items: []api.PodTemplate{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: ns,
				},
			},
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePath(getPodTemplatesResoureName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: podTemplateList},
	}
	response, err := c.Setup(t).PodTemplates(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, response, err)
}

func TestPodTemplateUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	podTemplate := &api.PodTemplate{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
		Template: api.PodTemplateSpec{},
	}
	c := &simple.Client{
		Request:  simple.Request{Method: "PUT", Path: testapi.Default.ResourcePath(getPodTemplatesResoureName(), ns, "abc"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200, Body: podTemplate},
	}
	response, err := c.Setup(t).PodTemplates(ns).Update(podTemplate)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestPodTemplateDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Default.ResourcePath(getPodTemplatesResoureName(), ns, "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).PodTemplates(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestPodTemplateWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Default.ResourcePathWithPrefix("watch", getPodTemplatesResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}}},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).PodTemplates(api.NamespaceAll).Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}
