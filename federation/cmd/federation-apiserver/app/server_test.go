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

package app

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"regexp"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	fed_v1b1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/federation/cmd/federation-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	ext_v1b1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestLongRunningRequestRegexp(t *testing.T) {
	regexp := regexp.MustCompile(options.NewServerRunOptions().LongRunningRequestRE)
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

var insecurePort = 8082
var serverIP = fmt.Sprintf("http://localhost:%v", insecurePort)
var groupVersions = []unversioned.GroupVersion{
	fed_v1b1.SchemeGroupVersion,
	ext_v1b1.SchemeGroupVersion,
}

func TestRun(t *testing.T) {
	s := options.NewServerRunOptions()
	s.InsecurePort = insecurePort
	_, ipNet, _ := net.ParseCIDR("10.10.10.0/24")
	s.ServiceClusterIPRange = *ipNet
	s.StorageConfig.ServerList = []string{"http://localhost:2379"}
	go func() {
		if err := Run(s); err != nil {
			t.Fatalf("Error in bringing up the server: %v", err)
		}
	}()
	if err := waitForApiserverUp(); err != nil {
		t.Fatalf("%v", err)
	}
	testSwaggerSpec(t)
	testSupport(t)
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

func testSupport(t *testing.T) {
	serverURL := serverIP + "/version"
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
	groupVersionForDiscoveryMap := make(map[string]unversioned.GroupVersionForDiscovery)
	for _, groupVersion := range groupVersions {
		groupVersionForDiscoveryMap[groupVersion.Group] = unversioned.GroupVersionForDiscovery{
			GroupVersion: groupVersion.String(),
			Version:      groupVersion.Version,
		}
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

	for _, groupVersion := range groupVersions {
		found := findGroup(apiGroupList.Groups, groupVersion.Group)
		assert.NotNil(t, found)
		assert.Equal(t, groupVersion.Group, found.Name)
		assert.Equal(t, 1, len(found.Versions))
		groupVersionForDiscovery := groupVersionForDiscoveryMap[groupVersion.Group]
		assert.Equal(t, groupVersionForDiscovery, found.Versions[0])
		assert.Equal(t, groupVersionForDiscovery, found.PreferredVersion)
	}
}

func testAPIGroup(t *testing.T) {
	for _, groupVersion := range groupVersions {
		serverURL := serverIP + "/apis/" + groupVersion.Group
		contents, err := readResponse(serverURL)
		if err != nil {
			t.Fatalf("%v", err)
		}
		var apiGroup unversioned.APIGroup
		err = json.Unmarshal(contents, &apiGroup)
		if err != nil {
			t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
		}
		// empty APIVersion for extensions group
		if groupVersion.Group == "extensions" {
			assert.Equal(t, "", apiGroup.APIVersion)
		} else {
			assert.Equal(t, "v1", apiGroup.APIVersion)
		}
		assert.Equal(t, apiGroup.Name, groupVersion.Group)
		assert.Equal(t, 1, len(apiGroup.Versions))
		assert.Equal(t, groupVersion.String(), apiGroup.Versions[0].GroupVersion)
		assert.Equal(t, groupVersion.Version, apiGroup.Versions[0].Version)
		assert.Equal(t, apiGroup.PreferredVersion, apiGroup.Versions[0])
	}

	testCoreAPIGroup(t)
}

func testCoreAPIGroup(t *testing.T) {
	serverURL := serverIP + "/api"
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiVersions unversioned.APIVersions
	err = json.Unmarshal(contents, &apiVersions)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, 1, len(apiVersions.Versions))
	assert.Equal(t, "v1", apiVersions.Versions[0])
	assert.NotEmpty(t, apiVersions.ServerAddressByClientCIDRs)
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
	testFederationResourceList(t)
	testCoreResourceList(t)
	testExtensionsResourceList(t)
}

func testFederationResourceList(t *testing.T) {
	serverURL := serverIP + "/apis/" + fed_v1b1.SchemeGroupVersion.String()
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiResourceList unversioned.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, "v1", apiResourceList.APIVersion)
	assert.Equal(t, fed_v1b1.SchemeGroupVersion.String(), apiResourceList.GroupVersion)
	// Assert that there are exactly 2 resources.
	assert.Equal(t, 2, len(apiResourceList.APIResources))

	found := findResource(apiResourceList.APIResources, "clusters")
	assert.NotNil(t, found)
	assert.False(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "clusters/status")
	assert.NotNil(t, found)
	assert.False(t, found.Namespaced)
}

func testCoreResourceList(t *testing.T) {
	serverURL := serverIP + "/api/" + v1.SchemeGroupVersion.String()
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiResourceList unversioned.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	assert.Equal(t, "", apiResourceList.APIVersion)
	assert.Equal(t, v1.SchemeGroupVersion.String(), apiResourceList.GroupVersion)
	// Assert that there are exactly 6 resources.
	assert.Equal(t, 6, len(apiResourceList.APIResources))

	// Verify services.
	found := findResource(apiResourceList.APIResources, "services")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "services/status")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)

	// Verify namespaces.
	found = findResource(apiResourceList.APIResources, "namespaces")
	assert.NotNil(t, found)
	assert.False(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "namespaces/status")
	assert.NotNil(t, found)
	assert.False(t, found.Namespaced)

	// Verify events.
	found = findResource(apiResourceList.APIResources, "events")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)

	// Verify secrets.
	found = findResource(apiResourceList.APIResources, "secrets")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)
}

func testExtensionsResourceList(t *testing.T) {
	serverURL := serverIP + "/apis/" + ext_v1b1.SchemeGroupVersion.String()
	contents, err := readResponse(serverURL)
	if err != nil {
		t.Fatalf("%v", err)
	}
	var apiResourceList unversioned.APIResourceList
	err = json.Unmarshal(contents, &apiResourceList)
	if err != nil {
		t.Fatalf("Error in unmarshalling response from server %s: %v", serverURL, err)
	}
	// empty APIVersion for extensions group
	assert.Equal(t, "", apiResourceList.APIVersion)
	assert.Equal(t, ext_v1b1.SchemeGroupVersion.String(), apiResourceList.GroupVersion)
	// Assert that there are exactly 5 resources.
	assert.Equal(t, 5, len(apiResourceList.APIResources))

	// Verify replicasets.
	found := findResource(apiResourceList.APIResources, "replicasets")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "replicasets/status")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "replicasets/scale")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)

	// Verify ingress.
	found = findResource(apiResourceList.APIResources, "ingresses")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)
	found = findResource(apiResourceList.APIResources, "ingresses/status")
	assert.NotNil(t, found)
	assert.True(t, found.Namespaced)
}
