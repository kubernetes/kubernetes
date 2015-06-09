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

package volume

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"strings"
)

func TestScrubberSuccess(t *testing.T) {
	client := &mockScrubberClient{}
	scrubber := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "scrubber-test",
			Namespace: api.NamespaceDefault,
		},
		Status: api.PodStatus{
			Phase: api.PodSucceeded,
		},
	}

	err := internalScrubPodVolumeAndWatchUntilCompletion(scrubber, client)
	if err != nil {
		t.Errorf("Unexpected error watching scrubber pod: %+v", err)
	}
	if !client.deletedCalled {
		t.Errorf("Expected deferred client.Delete to be called on scrubber pod")
	}
}

func TestScrubberFailure(t *testing.T) {
	client := &mockScrubberClient{}
	scrubber := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "scrubber-test",
			Namespace: api.NamespaceDefault,
		},
		Status: api.PodStatus{
			Phase:   api.PodFailed,
			Message: "foo",
		},
	}

	err := internalScrubPodVolumeAndWatchUntilCompletion(scrubber, client)
	if err == nil {
		t.Fatalf("Expected pod failure but got nil error returned")
	}
	if err != nil {
		if !strings.Contains(err.Error(), "foo") {
			t.Errorf("Expected pod.Status.Message %s but got %s", scrubber.Status.Message, err)
		}
	}
	if !client.deletedCalled {
		t.Errorf("Expected deferred client.Delete to be called on scrubber pod")
	}
}

type mockScrubberClient struct {
	pod           *api.Pod
	deletedCalled bool
}

func (c *mockScrubberClient) CreatePod(pod *api.Pod) (*api.Pod, error) {
	c.pod = pod
	return c.pod, nil
}

func (c *mockScrubberClient) GetPod(name, namespace string) (*api.Pod, error) {
	if c.pod != nil {
		return c.pod, nil
	} else {
		return nil, fmt.Errorf("pod does not exist")
	}
}

func (c *mockScrubberClient) DeletePod(name, namespace string) error {
	c.deletedCalled = true
	return nil
}

func (c *mockScrubberClient) WatchPod(name, namespace, resourceVersion string, stopChannel chan struct{}) func() *api.Pod {
	return func() *api.Pod {
		return c.pod
	}
}
