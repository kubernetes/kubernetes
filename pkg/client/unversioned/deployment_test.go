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
	"net/http"
	"net/url"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
	"k8s.io/kubernetes/pkg/labels"
)

func getDeploymentsResourceName() string {
	return "deployments"
}

func TestDeploymentCreate(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   &deployment,
		},
		Response: simple.Response{StatusCode: 200, Body: &deployment},
	}

	response, err := c.Setup(t).Deployments(ns).Create(&deployment)
	defer c.Close()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c.Validate(t, response, err)
}

func TestDeploymentGet(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: ns,
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: deployment},
	}

	response, err := c.Setup(t).Deployments(ns).Get("abc")
	defer c.Close()
	c.Validate(t, response, err)
}

func TestDeploymentList(t *testing.T) {
	ns := api.NamespaceDefault
	deploymentList := &extensions.DeploymentList{
		Items: []extensions.Deployment{
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
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, ""),
			Query:  simple.BuildQueryValues(nil),
			Body:   nil,
		},
		Response: simple.Response{StatusCode: 200, Body: deploymentList},
	}
	response, err := c.Setup(t).Deployments(ns).List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, response, err)
}

func TestDeploymentUpdate(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, "abc"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{StatusCode: 200, Body: deployment},
	}
	response, err := c.Setup(t).Deployments(ns).Update(deployment)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestDeploymentUpdateStatus(t *testing.T) {
	ns := api.NamespaceDefault
	deployment := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name:            "abc",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "PUT",
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, "abc") + "/status",
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{StatusCode: 200, Body: deployment},
	}
	response, err := c.Setup(t).Deployments(ns).UpdateStatus(deployment)
	defer c.Close()
	c.Validate(t, response, err)
}

func TestDeploymentDelete(t *testing.T) {
	ns := api.NamespaceDefault
	c := &simple.Client{
		Request: simple.Request{
			Method: "DELETE",
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, "foo"),
			Query:  simple.BuildQueryValues(nil),
		},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Deployments(ns).Delete("foo", nil)
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestDeploymentWatch(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePathWithPrefix("watch", getDeploymentsResourceName(), "", ""),
			Query:  url.Values{"resourceVersion": []string{}},
		},
		Response: simple.Response{StatusCode: 200},
	}
	_, err := c.Setup(t).Deployments(api.NamespaceAll).Watch(api.ListOptions{})
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestListDeploymentsLabels(t *testing.T) {
	ns := api.NamespaceDefault
	labelSelectorQueryParamName := unversioned.LabelSelectorQueryParam(testapi.Extensions.GroupVersion().String())
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath("deployments", ns, ""),
			Query:  simple.BuildQueryValues(url.Values{labelSelectorQueryParamName: []string{"foo=bar,name=baz"}})},
		Response: simple.Response{
			StatusCode: http.StatusOK,
			Body: &extensions.DeploymentList{
				Items: []extensions.Deployment{
					{
						ObjectMeta: api.ObjectMeta{
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
					},
				},
			},
		},
	}
	c.Setup(t)
	defer c.Close()
	c.QueryValidator[labelSelectorQueryParamName] = simple.ValidateLabels
	selector := labels.Set{"foo": "bar", "name": "baz"}.AsSelector()
	options := api.ListOptions{LabelSelector: selector}
	receivedPodList, err := c.Deployments(ns).List(options)
	c.Validate(t, receivedPodList, err)
}

func TestDeploymentRollback(t *testing.T) {
	ns := api.NamespaceDefault
	deploymentRollback := &extensions.DeploymentRollback{
		Name:               "abc",
		UpdatedAnnotations: map[string]string{},
		RollbackTo:         extensions.RollbackConfig{Revision: 1},
	}
	c := &simple.Client{
		Request: simple.Request{
			Method: "POST",
			Path:   testapi.Extensions.ResourcePath(getDeploymentsResourceName(), ns, "abc") + "/rollback",
			Query:  simple.BuildQueryValues(nil),
			Body:   deploymentRollback,
		},
		Response: simple.Response{StatusCode: http.StatusOK},
	}
	err := c.Setup(t).Deployments(ns).Rollback(deploymentRollback)
	defer c.Close()
	c.ValidateCommon(t, err)
}
