/*
Copyright 2014 The Kubernetes Authors.

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

package aggregated

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/version"
	restclient "k8s.io/client-go/rest"
)

func TestServerVersion(t *testing.T) {
	assert := assert.New(t)
	cacheDir, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(cacheDir)
	expect := version.Info{
		Major:     "foo",
		Minor:     "bar",
		GitCommit: "baz",
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(expect)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client, derr := NewAggregatedDiscoveryClientForConfig(&restclient.Config{Host: server.URL}, cacheDir)
	assert.True(derr == nil, "unexpected error creating AggregatedDiscoveryClient")
	got, err := client.ServerVersion()
	assert.True(err == nil, "unexpected error calling ServerVersion")
	assert.Equal(expect, *got)
}

func TestServerGroups(t *testing.T) {
	assert := assert.New(t)
	cacheDir, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(cacheDir)
	groupList := createDiscoveryAPIGroupList()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(*groupList)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client, derr := NewAggregatedDiscoveryClientForConfig(&restclient.Config{Host: server.URL}, cacheDir)
	assert.True(derr == nil, "unexpected error creating AggregatedDiscoveryClient")
	actualGroups, err := client.ServerGroups()
	assert.True(err == nil, "unexpected error calling ServerVersion")
	assert.True(len(actualGroups.Groups) == 1, "expected 1 APIGroup, got %d", len(actualGroups.Groups))
}

func TestServerResourcesForGroupVersion(t *testing.T) {
	assert := assert.New(t)
	cacheDir, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(cacheDir)
	groupList := createDiscoveryAPIGroupList()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(*groupList)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client, derr := NewAggregatedDiscoveryClientForConfig(&restclient.Config{Host: server.URL}, cacheDir)
	assert.True(derr == nil, "unexpected error creating AggregatedDiscoveryClient")
	actualResources, err := client.ServerResourcesForGroupVersion("apps/v1")
	assert.True(err == nil, "unexpected error calling ServerResourcesForGroupVersion")
	assert.True(actualResources.GroupVersion == "apps/v1", "expected GroupVersion (apps/v1), got (%s)", actualResources.GroupVersion)
	assert.True(len(actualResources.APIResources) == 1, "expected 1 APIGroup, got %d", len(actualResources.APIResources))
}

func TestServerGroupsAndResources(t *testing.T) {
	assert := assert.New(t)
	cacheDir, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(cacheDir)
	groupList := createDiscoveryAPIGroupList()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(*groupList)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client, derr := NewAggregatedDiscoveryClientForConfig(&restclient.Config{Host: server.URL}, cacheDir)
	assert.True(derr == nil, "unexpected error creating AggregatedDiscoveryClient")
	actualGroups, actualResources, err := client.ServerGroupsAndResources()
	assert.True(err == nil, "unexpected error calling ServerGroupAndResources")
	assert.True(len(actualGroups) == 1, "expected 1 APIGroup, got %d", len(actualGroups))
	assert.True(len(actualResources) == 1, "expected 1 APIResourceList, got %d", len(actualResources))
}

func TestServerPreferredResources(t *testing.T) {
	assert := assert.New(t)
	cacheDir, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(cacheDir)
	groupList := createDiscoveryAPIGroupList()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(*groupList)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client, derr := NewAggregatedDiscoveryClientForConfig(&restclient.Config{Host: server.URL}, cacheDir)
	assert.True(derr == nil, "unexpected error creating AggregatedDiscoveryClient")
	actualResources, err := client.ServerPreferredResources()
	assert.True(err == nil, "unexpected error calling ServerPreferredResources")
	assert.True(len(actualResources) == 1, "expected 1 APIResourceList, got %d", len(actualResources))
}

func TestServerPreferredNamespacedResources(t *testing.T) {
	assert := assert.New(t)
	cacheDir, err := ioutil.TempDir("", "")
	assert.NoError(err)
	defer os.RemoveAll(cacheDir)
	groupList := createDiscoveryAPIGroupList()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		output, err := json.Marshal(*groupList)
		if err != nil {
			t.Errorf("unexpected encoding error: %v", err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write(output)
	}))
	defer server.Close()
	client, derr := NewAggregatedDiscoveryClientForConfig(&restclient.Config{Host: server.URL}, cacheDir)
	assert.True(derr == nil, "unexpected error creating AggregatedDiscoveryClient")
	actualResources, err := client.ServerPreferredNamespacedResources()
	assert.True(err == nil, "unexpected error calling ServerPreferredNamespacedResources")
	assert.True(len(actualResources) == 1, "expected 1 APIResourceList, got %d", len(actualResources))
}

func createDiscoveryAPIGroupList() *metav1.DiscoveryAPIGroupList {
	gl := &metav1.DiscoveryAPIGroupList{}
	group1 := metav1.DiscoveryAPIGroup{}
	group1.Name = "apps"
	version := metav1.DiscoveryGroupVersion{}
	version.Version = "v1"
	resource := metav1.DiscoveryAPIResource{}
	resource.Name = "cronjobs"
	resource.SingularName = "cronjob"
	resource.Namespaced = true
	resource.Group = "apps"
	resource.Version = "v1"
	resource.Kind = "Cronjob"
	version.APIResources = append(version.APIResources, resource)
	group1.Versions = append(group1.Versions, version)
	gl.Groups = append(gl.Groups, group1)

	return gl
}
