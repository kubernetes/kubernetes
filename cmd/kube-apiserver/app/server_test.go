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

	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/genericapiserver"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
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

func TestUpdateEtcdOverrides(t *testing.T) {
	storageVersions := map[string]string{
		"":           "v1",
		"extensions": "extensions/v1beta1",
	}

	testCases := []struct {
		apigroup string
		resource string
		servers  []string
	}{
		{
			apigroup: api.GroupName,
			resource: "resource",
			servers:  []string{"http://127.0.0.1:10000"},
		},
		{
			apigroup: api.GroupName,
			resource: "resource",
			servers:  []string{"http://127.0.0.1:10000", "http://127.0.0.1:20000"},
		},
		{
			apigroup: extensions.GroupName,
			resource: "resource",
			servers:  []string{"http://127.0.0.1:10000"},
		},
	}

	for _, test := range testCases {
		newEtcd := func(_ runtime.NegotiatedSerializer, _, _ string, etcdConfig etcdstorage.EtcdConfig) (storage.Interface, error) {
			if !reflect.DeepEqual(test.servers, etcdConfig.ServerList) {
				t.Errorf("unexpected server list, expected: %#v, got: %#v", test.servers, etcdConfig.ServerList)
			}
			return nil, nil
		}
		storageDestinations := genericapiserver.NewStorageDestinations()
		override := test.apigroup + "/" + test.resource + "#" + strings.Join(test.servers, ";")
		defaultEtcdConfig := etcdstorage.EtcdConfig{
			Prefix:     genericapiserver.DefaultEtcdPathPrefix,
			ServerList: []string{"http://127.0.0.1"},
		}
		updateEtcdOverrides([]string{override}, storageVersions, defaultEtcdConfig, &storageDestinations, newEtcd)
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

func TestParseRuntimeConfig(t *testing.T) {
	testCases := []struct {
		runtimeConfig            map[string]string
		apiGroupVersionOverrides map[string]genericapiserver.APIGroupVersionOverride
		err                      bool
	}{
		{
			runtimeConfig:            map[string]string{},
			apiGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{},
			err: false,
		},
		{
			// Cannot override v1 resources.
			runtimeConfig: map[string]string{
				"api/v1/pods": "false",
			},
			apiGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{},
			err: true,
		},
		{
			// Disable v1.
			runtimeConfig: map[string]string{
				"api/v1": "false",
			},
			apiGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{
				"api/v1": {
					Disable: true,
				},
			},
			err: false,
		},
		{
			// Disable extensions.
			runtimeConfig: map[string]string{
				"extensions/v1beta1": "false",
			},
			apiGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{
				"extensions/v1beta1": {
					Disable: true,
				},
			},
			err: false,
		},
		{
			// Disable deployments.
			runtimeConfig: map[string]string{
				"extensions/v1beta1/deployments": "false",
			},
			apiGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{
				"extensions/v1beta1": {
					ResourceOverrides: map[string]bool{
						"deployments": false,
					},
				},
			},
			err: false,
		},
		{
			// Enable deployments and disable jobs.
			runtimeConfig: map[string]string{
				"extensions/v1beta1/deployments": "true",
				"extensions/v1beta1/jobs":        "false",
			},
			apiGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{
				"extensions/v1beta1": {
					ResourceOverrides: map[string]bool{
						"deployments": true,
						"jobs":        false,
					},
				},
			},
			err: false,
		},
	}
	for _, test := range testCases {
		s := &options.APIServer{
			RuntimeConfig: test.runtimeConfig,
		}
		apiGroupVersionOverrides, err := parseRuntimeConfig(s)

		if err == nil && test.err {
			t.Fatalf("expected error for test: %q", test)
		} else if err != nil && !test.err {
			t.Fatalf("unexpected error: %s, for test: %q", err, test)
		}

		if err == nil && !reflect.DeepEqual(apiGroupVersionOverrides, test.apiGroupVersionOverrides) {
			t.Fatalf("unexpected apiGroupVersionOverrides. Actual: %q, expected: %q", apiGroupVersionOverrides, test.apiGroupVersionOverrides)
		}
	}

}
