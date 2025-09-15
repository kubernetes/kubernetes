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
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/version"
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

func TestFakingServerVersionWithError(t *testing.T) {
	expectedError := errors.New("an error occurred")
	fakeClient := fakeclientset.NewSimpleClientset()
	fakeClient.Discovery().(*fakediscovery.FakeDiscovery).PrependReactor("*", "*", func(action kubetesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, expectedError
	})

	_, err := fakeClient.Discovery().ServerVersion()
	if err == nil {
		t.Fatal("ServerVersion should return error, returned nil instead")
	}
	if err != expectedError {
		t.Fatal("ServerVersion should return expected error, returned different error instead")
	}
}

func TestFakingServerResourcesForGroupVersionWithError(t *testing.T) {
	expectedError := errors.New("an error occurred")
	fakeClient := fakeclientset.NewClientset()
	fakeClient.Discovery().(*fakediscovery.FakeDiscovery).PrependReactor("*", "*", func(action kubetesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, expectedError
	})

	result, err := fakeClient.Discovery().ServerResourcesForGroupVersion("dummy.group.io/v1beta2")
	if result != nil {
		t.Errorf(`expect result to be nil but got "%v" instead`, result)
	}
	if !errors.Is(err, expectedError) {
		t.Errorf(`expect error to be "%v" but got "%v" instead`, expectedError, err)
	}
}

func TestFakingServerGroupsWithError(t *testing.T) {
	expectedError := errors.New("an error occurred")
	fakeClient := fakeclientset.NewClientset()
	fakeClient.Discovery().(*fakediscovery.FakeDiscovery).PrependReactor("*", "*", func(action kubetesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, expectedError
	})

	result, err := fakeClient.Discovery().ServerGroups()
	if result != nil {
		t.Errorf(`expect result to be nil but got "%v" instead`, result)
	}
	if !errors.Is(err, expectedError) {
		t.Errorf(`expect error to be "%v" but got "%v" instead`, expectedError, err)
	}
}

func TestFakingServerGroupsAndResourcesWithError(t *testing.T) {
	expectedError := errors.New("an error occurred")
	fakeClient := fakeclientset.NewClientset()
	fakeClient.Discovery().(*fakediscovery.FakeDiscovery).PrependReactor("get", "resource", func(action kubetesting.Action) (handled bool, ret runtime.Object, err error) {
		return true, nil, expectedError
	})

	_, _, err := fakeClient.Discovery().ServerGroupsAndResources()
	if !errors.Is(err, expectedError) {
		t.Errorf(`expect error to be "%v" but got "%v" instead`, expectedError, err)
	}
}
