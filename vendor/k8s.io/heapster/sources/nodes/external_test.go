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
	"encoding/json"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/heapster/sources/api"
)

const tempHostsFile = "/temp_file"

func writeMarshaledData(f *os.File, v interface{}) error {
	data, err := json.Marshal(&v)
	if err != nil {
		return err
	}
	_, err = f.WriteAt(data, 0)
	return err
}

func externalizeNodes(nodeList *NodeList) *api.ExternalNodeList {
	externalNodeList := api.ExternalNodeList{}
	for host, info := range nodeList.Items {
		externalNodeList.Items = append(externalNodeList.Items, api.ExternalNode{Name: string(host), IP: info.PublicIP})
	}
	return &externalNodeList
}

func TestExternalFile(t *testing.T) {
	f, err := ioutil.TempFile("", "")
	if err != nil {
		t.Error(err)
	}
	defer os.Remove(f.Name())
	nodesApi := externalCadvisorNodes{hostsFile: f.Name(), nodes: nil}

	testData := &NodeList{
		Items: map[Host]Info{
			Host("host1"): {PublicIP: "1.2.3.4", InternalIP: "1.2.3.4"},
			Host("host2"): {PublicIP: "1.2.3.5", InternalIP: "1.2.3.5"},
		},
	}
	require.NoError(t, writeMarshaledData(f, externalizeNodes(testData)))
	res, err := nodesApi.List()
	require.NoError(t, err)
	require.True(t, reflect.DeepEqual(res, testData), "failure. Expected: %+v, got: %+v", res, testData)

	testData.Items[Host("host3")] = Info{PublicIP: "2.2.2.2", InternalIP: "2.2.2.2"}
	require.NoError(t, writeMarshaledData(f, externalizeNodes(testData)))
	res, err = nodesApi.List()
	require.NoError(t, err)
	require.True(t, reflect.DeepEqual(res, testData), "failure. Expected: %+v, got: %+v", res, testData)
}

func TestLocalhostMonitoring(t *testing.T) {
	nodesApi, err := NewExternalNodes(true, "")
	require.NoError(t, err)

	const (
		localhost   = "localhost"
		localhostIP = "127.0.0.1"
	)

	// Should only have the localhost node.
	res, err := nodesApi.List()
	require.NoError(t, err)
	assert.Len(t, res.Items, 1)
	_, ok := res.Items[localhost]
	require.True(t, ok)
	assert.Equal(t, res.Items[localhost].InternalIP, localhostIP)
	assert.Equal(t, res.Items[localhost].PublicIP, localhostIP)
}
