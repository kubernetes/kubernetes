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

package fake_test

import (
	"errors"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	kubetesting "k8s.io/client-go/testing"
)

func TestFakingServerVersion(t *testing.T) {
	client := fakeclientset.NewSimpleClientset()
	fakeDiscovery, ok := client.Discovery().(*fakediscovery.FakeDiscovery)
	if !ok {
		t.Fatalf("couldn't convert Discovery() to *FakeDiscovery")
	}

	testGitCommit := "v1.0.0"
	fakeDiscovery.FakedServerVersion = &version.Info{
		GitCommit: testGitCommit,
	}

	sv, err := client.Discovery().ServerVersion()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if sv.GitCommit != testGitCommit {
		t.Fatalf("unexpected faked discovery return value: %q", sv.GitCommit)
	}
}

func TestGetErrorResponseFromClient(t *testing.T) {
	expErr := errors.New("error override")

	tests := map[string]struct {
		call     func(client discovery.DiscoveryInterface) error
		resource string
	}{
		"ServerResourcesForGroupVersion": {
			call: func(client discovery.DiscoveryInterface) error {
				_, err := client.ServerResourcesForGroupVersion("")
				return err
			},
			resource: "resource",
		},
		"ServerGroupsAndResources": {
			call: func(client discovery.DiscoveryInterface) error {
				_, _, err := client.ServerGroupsAndResources()
				return err
			},
			resource: "resource",
		},
		"ServerGroups": {
			call: func(client discovery.DiscoveryInterface) error {
				_, err := client.ServerGroups()
				return err
			},
			resource: "group",
		},
		"ServerVersion": {
			call: func(client discovery.DiscoveryInterface) error {
				_, err := client.ServerVersion()
				return err
			},
			resource: "version",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			reactor := func(action kubetesting.Action) (handled bool, ret runtime.Object, err error) {
				return true, nil, expErr
			}

			client := fakeclientset.NewSimpleClientset()
			client.PrependReactor("get", test.resource, reactor)

			err := test.call(client.Discovery())
			if !reflect.DeepEqual(err, expErr) {
				t.Errorf("unexpected error response, exp=%v got=%v",
					expErr, err)
			}
		})
	}
}
