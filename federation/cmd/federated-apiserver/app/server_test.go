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

package app

import (
	"regexp"
	"testing"

	"encoding/json"
	"fmt"
	"github.com/stretchr/testify/assert"
	"io/ioutil"
	fed_v1a1 "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/cmd/federated-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"net"
	"net/http"
	"time"
)

func TestLongRunningRequestRegexp(t *testing.T) {
	regexp := regexp.MustCompile(options.NewAPIServer().LongRunningRequestRE)
	dontMatch := []string{
		"/api/v1/watch-namespace/",
		"/api/v1/namespace-proxy/",
		"/api/v1/namespace-watch",
		"/api/v1/namespace-proxy",
		"/api/v1/namespace-portforward/pods",
		"/api/v1/portforward/pods",
		". anything",
		"/ that",
	}
	doMatch := []string{
		"/api/v1/pods/watch",
		"/api/v1/watch/stuff",
		"/api/v1/default/service/proxy",
		"/api/v1/pods/proxy/path/to/thing",
		"/api/v1/namespaces/myns/pods/mypod/log",
		"/api/v1/namespaces/myns/pods/mypod/logs",
		"/api/v1/namespaces/myns/pods/mypod/portforward",
		"/api/v1/namespaces/myns/pods/mypod/exec",
		"/api/v1/namespaces/myns/pods/mypod/attach",
		"/api/v1/namespaces/myns/pods/mypod/log/",
		"/api/v1/namespaces/myns/pods/mypod/logs/",
		"/api/v1/namespaces/myns/pods/mypod/portforward/",
		"/api/v1/namespaces/myns/pods/mypod/exec/",
		"/api/v1/namespaces/myns/pods/mypod/attach/",
		"/api/v1/watch/namespaces/myns/pods",
	}
	for _, path := range dontMatch {
		if regexp.MatchString(path) {
			t.Errorf("path should not have match regexp but did: %s", path)
		}
	}
	for _, path := range doMatch {
		if !regexp.MatchString(path) {
			t.Errorf("path should have match regexp did not: %s", path)
		}
	}
}

var insecurePort = 8081
var serverIP = fmt.Sprintf("http://localhost:%v", insecurePort)
var groupVersion = fed_v1a1.SchemeGroupVersion

func TestRun(t *testing.T) {
	s := options.NewAPIServer()
	s.InsecurePort = insecurePort
	_, ipNet, _ := net.ParseCIDR("10.10.10.0/24")
	s.ServiceClusterIPRange = *ipNet
	s.EtcdConfig.ServerList = []string{"http://localhost:4001"}
	go func() {
		if err := Run(s); err != nil {
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

func findGroup(groups []unversioned.APIGroup, groupName string) *unversioned.APIGroup {
	for _, group := range groups {
		if group.Name == groupName {
			return &group
		}
	}
	return nil
}

func testAPIGroupList(t *testing.T) {
	var groupVersionForDiscovery = unversioned.GroupVersionForDiscovery{
		GroupVersion: groupVersion.String(),
		Version:      groupVersion.Version,
	}

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

	found := findGroup(apiGroupList.Groups, groupVersion.Group)
	assert.NotNil(t, found)
	assert.Equal(t, found.Name, groupVersion.Group)
	assert.Equal(t, 1, len(found.Versions))
	assert.Equal(t, found.Versions[0], groupVersionForDiscovery)
	assert.Equal(t, found.PreferredVersion, groupVersionForDiscovery)
}

func testAPIGroup(t *testing.T) {
	serverURL := serverIP + "/apis/federation"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiGroup unversioned.APIGroup
	err = json.Unmarshal(contents, &apiGroup)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, apiGroup.APIVersion, "v1")
	assert.Equal(t, apiGroup.Name, groupVersion.Group)
	assert.Equal(t, 1, len(apiGroup.Versions))
	assert.Equal(t, apiGroup.Versions[0].GroupVersion, groupVersion.String())
	assert.Equal(t, apiGroup.Versions[0].Version, groupVersion.Version)
	assert.Equal(t, apiGroup.Versions[0], apiGroup.PreferredVersion)
}

func findResource(resources []unversioned.APIResource, resourceName string) *unversioned.APIResource {
	for _, resource := range resources {
		if resource.Name == resourceName {
			return &resource
		}
	}
	return nil
}

func testAPIResourceList(t *testing.T) {
	serverURL := serverIP + "/apis/federation/v1alpha1"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiResourceList unversioned.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, apiResourceList.APIVersion, "v1")
	assert.Equal(t, apiResourceList.GroupVersion, groupVersion.String())

	found := findResource(apiResourceList.APIResources, "clusters")
	assert.NotNil(t, found)
	assert.False(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "clusters/status")
	assert.NotNil(t, found)
	assert.False(t, found.Namespaced)
}
