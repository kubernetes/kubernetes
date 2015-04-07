/*
Copyright 2015 Google Inc. All rights reserved.

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
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
)

func getNodesResourceName() string {
	if api.PreV1Beta3(testapi.Version()) {
		return "minions"
	}
	return "nodes"
}
func TestListMinions(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getNodesResourceName(), "", ""),
		},
		Response: Response{StatusCode: 200, Body: &api.NodeList{ListMeta: api.ListMeta{ResourceVersion: "1"}}},
	}
	response, err := c.Setup().Nodes().List()
	c.Validate(t, response, err)
}

func TestGetMinion(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "GET",
			Path:   testapi.ResourcePath(getNodesResourceName(), "", "1"),
		},
		Response: Response{StatusCode: 200, Body: &api.Node{ObjectMeta: api.ObjectMeta{Name: "minion-1"}}},
	}
	response, err := c.Setup().Nodes().Get("1")
	c.Validate(t, response, err)
}

func TestGetMinionWithNoName(t *testing.T) {
	c := &testClient{Error: true}
	receivedPod, err := c.Setup().Nodes().Get("")
	if (err != nil) && (err.Error() != nameRequiredError) {
		t.Errorf("Expected error: %v, but got %v", nameRequiredError, err)
	}

	c.Validate(t, receivedPod, err)
}

func TestCreateMinion(t *testing.T) {
	requestMinion := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: "minion-1",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1000m"),
				api.ResourceMemory: resource.MustParse("1Mi"),
			},
		},
		Spec: api.NodeSpec{
			Unschedulable: false,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "POST",
			Path:   testapi.ResourcePath(getNodesResourceName(), "", ""),
			Body:   requestMinion},
		Response: Response{
			StatusCode: 200,
			Body:       requestMinion,
		},
	}
	receivedMinion, err := c.Setup().Nodes().Create(requestMinion)
	c.Validate(t, receivedMinion, err)
}

func TestDeleteMinion(t *testing.T) {
	c := &testClient{
		Request: testRequest{
			Method: "DELETE",
			Path:   testapi.ResourcePath(getNodesResourceName(), "", "foo"),
		},
		Response: Response{StatusCode: 200},
	}
	err := c.Setup().Nodes().Delete("foo")
	c.Validate(t, nil, err)
}

func TestUpdateMinion(t *testing.T) {
	requestMinion := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				api.ResourceCPU:    resource.MustParse("1000m"),
				api.ResourceMemory: resource.MustParse("1Mi"),
			},
		},
		Spec: api.NodeSpec{
			Unschedulable: true,
		},
	}
	c := &testClient{
		Request: testRequest{
			Method: "PUT",
			Path:   testapi.ResourcePath(getNodesResourceName(), "", "foo"),
		},
		Response: Response{StatusCode: 200, Body: requestMinion},
	}
	response, err := c.Setup().Nodes().Update(requestMinion)
	c.Validate(t, response, err)
}
