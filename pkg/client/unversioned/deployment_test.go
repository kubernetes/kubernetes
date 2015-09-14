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
	"k8s.io/kubernetes/pkg/apis/experimental"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
)

func getDeploymentsResoureName() string {
	return "deployments"
}

func TestDeploymentCreate(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.Experimental.ResourcePath(getDeploymentsResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   &deployment,
		},
		Response: Response{StatusCode: 200, Body: &deployment},
	}

	response, err := c.Setup(t).Deployments(ns).Create(&deployment)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, response, err)
}

func TestDeploymentGet(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := &experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Experimental.ResourcePath(getDeploymentsResoureName(), ns, "abc"),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: deployment},
	}

	response, err := c.Setup(t).Deployments(ns).Get("abc")
	c.Validate(t, response, err)
}

func TestDeploymentList(t *testing.T) {
	ns := api.NamespaceDefault
	deploymentList := &experimental.DeploymentList{
		Items: []experimental.Deployment{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: ns,
				},
			},
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Experimental.ResourcePath(getDeploymentsResoureName(), ns, ""),
			Query:  buildQueryValues(nil),
			Body:   nil,
		},
		Response: Response{StatusCode: 200, Body: deploymentList},
	}
	response, err := c.Setup(t).Deployments(ns).List(labels.Everything(), fields.Everything())
	c.Validate(t, response, err)
}

func TestDeploymentUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := &experimental.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.Experimental.ResourcePath(getDeploymentsResoureName(), ns, "abc"),
			Query:  buildQueryValues(nil),
		},
		Response: Response{StatusCode: 200, Body: deployment},
	}
	response, err := c.Setup(t).Deployments(ns).Update(deployment)
	c.Validate(t, response, err)
}

func TestDeploymentDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &testClient{
		Request: testRequest{
			Method: "DELETE",
			Path:   testapi.Experimental.ResourcePath(getDeploymentsResoureName(), ns, "foo"),
			Query:  buildQueryValues(nil),
		},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup(t).Deployments(ns).Delete("foo", nil)
	c.Validate(t, nil, err)
}

func TestDeploymentWatch(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.Experimental.ResourcePathWithPrefix("watch", getDeploymentsResoureName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}},
		},
		Response: Response{StatusCode: 200},
	}
	_, err := c.Setup(t).Deployments(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), "")
	c.Validate(t, nil, err)
}
