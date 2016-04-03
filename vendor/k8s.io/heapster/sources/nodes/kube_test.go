// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package nodes

import (
	"fmt"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

func TestCreate(t *testing.T) {
	handler := util.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Default.Version()})
	kubeNodes, err := NewKubeNodes(client)
	require.NoError(t, err)
	_, err = kubeNodes.List()
	require.NoError(t, err)
}

func body(obj runtime.Object) string {
	if obj != nil {
		bs, _ := latest.GroupOrDie("").Codec.Encode(obj)
		body := string(bs)
		return body
	}

	return ""
}

func TestNodes(t *testing.T) {
	// TODO(vishh): Get this test to work.
	t.Skip("skipping watch nodes test.")
	expectedNodeList := &api.NodeList{
		Items: []api.Node{
			{
				ObjectMeta: api.ObjectMeta{
					Name: "test-machine-a",
				},
				Status: api.NodeStatus{
					Addresses: []api.NodeAddress{
						{
							Type:    api.NodeExternalIP,
							Address: "1.2.3.5",
						},
					},
				},
			},
			{
				ObjectMeta: api.ObjectMeta{
					Name: "test-machine-b",
				},
				Status: api.NodeStatus{
					Addresses: []api.NodeAddress{
						{
							Type:    api.NodeExternalIP,
							Address: "1.2.3.4",
						},
					},
				},
			},
		},
	}

	handler := util.FakeHandler{
		StatusCode:   200,
		RequestBody:  "",
		ResponseBody: body(expectedNodeList),
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	kubeClient := client.NewOrDie(&client.Config{Host: server.URL, Version: testapi.Default.Version()})
	kubeNodes, err := NewKubeNodes(kubeClient)
	require.NoError(t, err)
	nodeList, err := kubeNodes.List()
	require.NoError(t, err)
	fmt.Printf("%+v\n", nodeList)
	for _, expectedNode := range expectedNodeList.Items {
		node, ok := nodeList.Items[Host(expectedNode.Name)]
		assert.True(t, ok)
		assert.Equal(t, node.PublicIP, expectedNode.Status.Addresses[0].Address)
	}
}
