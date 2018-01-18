/*
Copyright 2017 The Kubernetes Authors.

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

package discovery

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/endpoints/request"
)

var (
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	registry             = registered.NewOrDie("")
	scheme               = runtime.NewScheme()
	codecs               = serializer.NewCodecFactory(scheme)
)

func init() {
	// Register Unversioned types under their own special group
	scheme.AddUnversionedTypes(schema.GroupVersion{Group: "", Version: "v1"},
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
}

func decodeResponse(t *testing.T, resp *http.Response, obj interface{}) error {
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	t.Log(string(data))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, obj); err != nil {
		return err
	}
	return nil
}

func getGroupList(t *testing.T, server *httptest.Server) (*metav1.APIGroupList, error) {
	resp, err := http.Get(server.URL)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected server response, expected %d, actual: %d", http.StatusOK, resp.StatusCode)
	}

	groupList := metav1.APIGroupList{}
	err = decodeResponse(t, resp, &groupList)
	return &groupList, err
}

func TestDiscoveryAtAPIS(t *testing.T) {
	mapper := request.NewRequestContextMapper()
	handler := NewRootAPIsHandler(DefaultAddresses{DefaultAddress: "192.168.1.1"}, codecs, mapper)

	server := httptest.NewServer(request.WithRequestContext(handler, mapper))
	groupList, err := getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(t, 0, len(groupList.Groups))

	// Add a Group.
	extensionsGroupName := "extensions"
	extensionsVersions := []metav1.GroupVersionForDiscovery{
		{
			GroupVersion: extensionsGroupName + "/v1",
			Version:      "v1",
		},
	}
	extensionsPreferredVersion := metav1.GroupVersionForDiscovery{
		GroupVersion: extensionsGroupName + "/preferred",
		Version:      "preferred",
	}
	handler.AddGroup(metav1.APIGroup{
		Name:             extensionsGroupName,
		Versions:         extensionsVersions,
		PreferredVersion: extensionsPreferredVersion,
	})

	groupList, err = getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(t, 1, len(groupList.Groups))
	groupListGroup := groupList.Groups[0]
	assert.Equal(t, extensionsGroupName, groupListGroup.Name)
	assert.Equal(t, extensionsVersions, groupListGroup.Versions)
	assert.Equal(t, extensionsPreferredVersion, groupListGroup.PreferredVersion)
	assert.Equal(t, handler.addresses.ServerAddressByClientCIDRs(utilnet.GetClientIP(&http.Request{})), groupListGroup.ServerAddressByClientCIDRs)

	// Remove the group.
	handler.RemoveGroup(extensionsGroupName)
	groupList, err = getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(t, 0, len(groupList.Groups))
}

func TestDiscoveryOrdering(t *testing.T) {
	mapper := request.NewRequestContextMapper()
	handler := NewRootAPIsHandler(DefaultAddresses{DefaultAddress: "192.168.1.1"}, codecs, mapper)

	server := httptest.NewServer(request.WithRequestContext(handler, mapper))
	groupList, err := getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(t, 0, len(groupList.Groups))

	// Register three groups
	handler.AddGroup(metav1.APIGroup{Name: "x"})
	handler.AddGroup(metav1.APIGroup{Name: "y"})
	handler.AddGroup(metav1.APIGroup{Name: "z"})
	// Register three additional groups that come earlier alphabetically
	handler.AddGroup(metav1.APIGroup{Name: "a"})
	handler.AddGroup(metav1.APIGroup{Name: "b"})
	handler.AddGroup(metav1.APIGroup{Name: "c"})
	// Make sure re-adding doesn't double-register or make a group lose its place
	handler.AddGroup(metav1.APIGroup{Name: "x"})

	groupList, err = getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	assert.Equal(t, 6, len(groupList.Groups))
	assert.Equal(t, "x", groupList.Groups[0].Name)
	assert.Equal(t, "y", groupList.Groups[1].Name)
	assert.Equal(t, "z", groupList.Groups[2].Name)
	assert.Equal(t, "a", groupList.Groups[3].Name)
	assert.Equal(t, "b", groupList.Groups[4].Name)
	assert.Equal(t, "c", groupList.Groups[5].Name)

	// Remove a group.
	handler.RemoveGroup("a")
	groupList, err = getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(t, 5, len(groupList.Groups))

	// Re-adding should move to the end.
	handler.AddGroup(metav1.APIGroup{Name: "a"})
	groupList, err = getGroupList(t, server)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(t, 6, len(groupList.Groups))
	assert.Equal(t, "x", groupList.Groups[0].Name)
	assert.Equal(t, "y", groupList.Groups[1].Name)
	assert.Equal(t, "z", groupList.Groups[2].Name)
	assert.Equal(t, "b", groupList.Groups[3].Name)
	assert.Equal(t, "c", groupList.Groups[4].Name)
	assert.Equal(t, "a", groupList.Groups[5].Name)
}
