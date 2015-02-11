/*
Copyright 2014 Google Inc. All rights reserved.

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

package pod

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type fakeCache struct {
	requestedNamespace string
	requestedName      string
	clearedNamespace   string
	clearedName        string

	statusToReturn *api.PodStatus
	errorToReturn  error
}

func (f *fakeCache) GetPodStatus(namespace, name string) (*api.PodStatus, error) {
	f.requestedNamespace = namespace
	f.requestedName = name
	return f.statusToReturn, f.errorToReturn
}

func (f *fakeCache) ClearPodStatus(namespace, name string) {
	f.clearedNamespace = namespace
	f.clearedName = name
}

func TestPodStatusDecorator(t *testing.T) {
	cache := &fakeCache{statusToReturn: &api.PodStatus{Phase: api.PodRunning}}
	pod := &api.Pod{}
	if err := PodStatusDecorator(cache)(pod); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pod.Status.Phase != api.PodRunning {
		t.Errorf("unexpected pod: %#v", pod)
	}
	pod = &api.Pod{
		Status: api.PodStatus{
			Host: "foo",
		},
	}
	if err := PodStatusDecorator(cache)(pod); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pod.Status.Phase != api.PodRunning || pod.Status.Host != "foo" {
		t.Errorf("unexpected pod: %#v", pod)
	}
}

func TestPodStatusDecoratorError(t *testing.T) {
	cache := &fakeCache{errorToReturn: fmt.Errorf("test error")}
	pod := &api.Pod{}
	if err := PodStatusDecorator(cache)(pod); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pod.Status.Phase != api.PodUnknown {
		t.Errorf("unexpected pod: %#v", pod)
	}
}
