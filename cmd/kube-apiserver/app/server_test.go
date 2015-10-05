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
	"reflect"
	"regexp"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/storage"
)

func TestLongRunningRequestRegexp(t *testing.T) {
	regexp := regexp.MustCompile(defaultLongRunningRequestRE)
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

func TestGenerateStorageVersionMap(t *testing.T) {
	testCases := []struct {
		legacyVersion   string
		storageVersions string
		expectedMap     map[string]string
	}{
		{
			legacyVersion:   "v1",
			storageVersions: "v1,experimental/v1alpha1",
			expectedMap: map[string]string{
				"":             "v1",
				"experimental": "experimental/v1alpha1",
			},
		},
		{
			legacyVersion:   "",
			storageVersions: "experimental/v1alpha1,v1",
			expectedMap: map[string]string{
				"":             "v1",
				"experimental": "experimental/v1alpha1",
			},
		},
		{
			legacyVersion:   "",
			storageVersions: "",
			expectedMap:     map[string]string{},
		},
	}
	for _, test := range testCases {
		output := generateStorageVersionMap(test.legacyVersion, test.storageVersions)
		if !reflect.DeepEqual(test.expectedMap, output) {
			t.Errorf("unexpected error. expect: %v, got: %v", test.expectedMap, output)
		}
	}
}

func TestUpdateEtcdOverrides(t *testing.T) {
	storageVersions := generateStorageVersionMap("", "v1,experimental/v1alpha1")

	testCases := []struct {
		apigroup string
		resource string
		servers  []string
	}{
		{
			apigroup: "",
			resource: "resource",
			servers:  []string{"http://127.0.0.1:10000"},
		},
		{
			apigroup: "",
			resource: "resource",
			servers:  []string{"http://127.0.0.1:10000", "http://127.0.0.1:20000"},
		},
		{
			apigroup: "experimental",
			resource: "resource",
			servers:  []string{"http://127.0.0.1:10000"},
		},
	}

	for _, test := range testCases {
		newEtcd := func(_ string, serverList []string, _ meta.VersionInterfacesFunc, _, _ string) (storage.Interface, error) {
			if !reflect.DeepEqual(test.servers, serverList) {
				t.Errorf("unexpected server list, expected: %#v, got: %#v", test.servers, serverList)
			}
			return nil, nil
		}
		storageDestinations := master.NewStorageDestinations()
		override := test.apigroup + "/" + test.resource + "#" + strings.Join(test.servers, ";")
		updateEtcdOverrides([]string{override}, storageVersions, "", &storageDestinations, newEtcd)
		apigroup, ok := storageDestinations.APIGroups[test.apigroup]
		if !ok {
			t.Errorf("apigroup: %s not created", test.apigroup)
			continue
		}
		if apigroup.Overrides == nil {
			t.Errorf("Overrides not created for: %s", test.apigroup)
			continue
		}
		if _, ok := apigroup.Overrides[test.resource]; !ok {
			t.Errorf("override not created for: %s", test.resource)
			continue
		}
	}
}
