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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
)

func getThirdPartyResourceName() string {
	return "thirdpartyresources"
}

func TestListThirdPartyResources(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{
			Method: "GET",
			Path:   testapi.Extensions.ResourcePath(getThirdPartyResourceName(), "", ""),
		},
		Response: simple.Response{StatusCode: 200,
			Body: &extensions.ThirdPartyResourceList{
				Items: []extensions.ThirdPartyResource{
					{
						ObjectMeta: api.ObjectMeta{
							Name: "foo",
							Labels: map[string]string{
								"foo":  "bar",
								"name": "baz",
							},
						},
						Description: "test third party resource",
					},
				},
			},
		},
	}
	receivedDSs, err := c.Setup(t).Extensions().ThirdPartyResources().List(api.ListOptions{})
	defer c.Close()
	c.Validate(t, receivedDSs, err)

}

func TestGetThirdPartyResource(t *testing.T) {
	c := &simple.Client{
		Request: simple.Request{Method: "GET", Path: testapi.Extensions.ResourcePath(getThirdPartyResourceName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.ThirdPartyResource{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Description: "test third party resource",
			},
		},
	}
	receivedThirdPartyResource, err := c.Setup(t).Extensions().ThirdPartyResources().Get("foo")
	defer c.Close()
	c.Validate(t, receivedThirdPartyResource, err)
}

func TestGetThirdPartyResourceWithNoName(t *testing.T) {
	c := &simple.Client{Error: true}
	receivedPod, err := c.Setup(t).Extensions().ThirdPartyResources().Get("")
	defer c.Close()
	if (err != nil) && (err.Error() != simple.NameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", simple.NameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestUpdateThirdPartyResource(t *testing.T) {
	requestThirdPartyResource := &extensions.ThirdPartyResource{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Extensions.ResourcePath(getThirdPartyResourceName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.ThirdPartyResource{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Description: "test third party resource",
			},
		},
	}
	receivedThirdPartyResource, err := c.Setup(t).Extensions().ThirdPartyResources().Update(requestThirdPartyResource)
	defer c.Close()
	c.Validate(t, receivedThirdPartyResource, err)
}

func TestUpdateThirdPartyResourceUpdateStatus(t *testing.T) {
	requestThirdPartyResource := &extensions.ThirdPartyResource{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "PUT", Path: testapi.Extensions.ResourcePath(getThirdPartyResourceName(), "", "foo") + "/status", Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.ThirdPartyResource{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Description: "test third party resource",
			},
		},
	}
	receivedThirdPartyResource, err := c.Setup(t).Extensions().ThirdPartyResources().UpdateStatus(requestThirdPartyResource)
	defer c.Close()
	c.Validate(t, receivedThirdPartyResource, err)
}

func TestDeleteThirdPartyResource(t *testing.T) {
	c := &simple.Client{
		Request:  simple.Request{Method: "DELETE", Path: testapi.Extensions.ResourcePath(getThirdPartyResourceName(), "", "foo"), Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{StatusCode: 200},
	}
	err := c.Setup(t).Extensions().ThirdPartyResources().Delete("foo")
	defer c.Close()
	c.Validate(t, nil, err)
}

func TestCreateThirdPartyResource(t *testing.T) {
	requestThirdPartyResource := &extensions.ThirdPartyResource{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
	}
	c := &simple.Client{
		Request: simple.Request{Method: "POST", Path: testapi.Extensions.ResourcePath(getThirdPartyResourceName(), "", ""), Body: requestThirdPartyResource, Query: simple.BuildQueryValues(nil)},
		Response: simple.Response{
			StatusCode: 200,
			Body: &extensions.ThirdPartyResource{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo":  "bar",
						"name": "baz",
					},
				},
				Description: "test third party resource",
			},
		},
	}
	receivedThirdPartyResource, err := c.Setup(t).Extensions().ThirdPartyResources().Create(requestThirdPartyResource)
	defer c.Close()
	c.Validate(t, receivedThirdPartyResource, err)
}
