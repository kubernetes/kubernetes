/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"
	"time"

	"k8s.io/kubernetes/cmd/libs/go2idl/client-gen/test_apis/testgroup.k8s.io/v1"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

var serverIP = fmt.Sprintf("http://localhost:%d", InsecurePort)

var groupVersion = v1.SchemeGroupVersion

var groupVersionForDiscovery = unversioned.GroupVersionForDiscovery{
	GroupVersion: groupVersion.String(),
	Version:      groupVersion.Version,
}

func TestRun(t *testing.T) {
	go func() {
		if err := Run(NewServerRunOptions()); err != nil {
			t.Fatalf("Error in bringing up the server: %v", err)
		}
	}()
	if err := waitForApiserverUp(); err != nil {
		t.Fatalf("%v", err)
	}
	testSwaggerSpec(t)
	testAPIGroupList(t)
	testAPIGroup(t)
	testAPIResourceList(t)
}

func waitForApiserverUp() error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		_, err := http.Get(serverIP)
		if err == nil {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

func readResponse(serverURL string) ([]byte, error) {
	response, err := http.Get(serverURL)
	if err != nil {
		return nil, fmt.Errorf("Error in fetching %s: %v", serverURL, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d for URL: %s, expected status: %d", response.StatusCode, serverURL, http.StatusOK)
	}
	contents, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Error reading response from %s: %v", serverURL, err)
	}
	return contents, nil
}

func testSwaggerSpec(t *testing.T) {
	serverURL := serverIP + "/swaggerapi"
	_, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
}

func testAPIGroupList(t *testing.T) {
	serverURL := serverIP + "/apis"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiGroupList unversioned.APIGroupList
	err = json.Unmarshal(contents, &apiGroupList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, 1, len(apiGroupList.Groups))
	assert.Equal(t, apiGroupList.Groups[0].Name, groupVersion.Group)
	assert.Equal(t, 1, len(apiGroupList.Groups[0].Versions))
	assert.Equal(t, apiGroupList.Groups[0].Versions[0], groupVersionForDiscovery)
	assert.Equal(t, apiGroupList.Groups[0].PreferredVersion, groupVersionForDiscovery)
}

func testAPIGroup(t *testing.T) {
	serverURL := serverIP + "/apis/testgroup.k8s.io"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiGroup unversioned.APIGroup
	err = json.Unmarshal(contents, &apiGroup)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, apiGroup.APIVersion, groupVersion.Version)
	assert.Equal(t, apiGroup.Name, groupVersion.Group)
	assert.Equal(t, 1, len(apiGroup.Versions))
	assert.Equal(t, apiGroup.Versions[0].GroupVersion, groupVersion.String())
	assert.Equal(t, apiGroup.Versions[0].Version, groupVersion.Version)
	assert.Equal(t, apiGroup.Versions[0], apiGroup.PreferredVersion)
}

func testAPIResourceList(t *testing.T) {
	serverURL := serverIP + "/apis/testgroup.k8s.io/v1"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiResourceList unversioned.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, apiResourceList.APIVersion, groupVersion.Version)
	assert.Equal(t, apiResourceList.GroupVersion, groupVersion.String())
	assert.Equal(t, 1, len(apiResourceList.APIResources))
	assert.Equal(t, apiResourceList.APIResources[0].Name, "testtypes")
	assert.True(t, apiResourceList.APIResources[0].Namespaced)
}
