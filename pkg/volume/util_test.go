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
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

func TestRecyclerSuccess(t *testing.T) {
	client := &mockRecyclerClient{}
	recycler := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "recycler-test",
			Namespace: api.NamespaceDefault,
		},
		Status: api.PodStatus{
			Phase: api.PodSucceeded,
		},
	}

	err := internalRecycleVolumeByWatchingPodUntilCompletion(recycler, client)
	if err != nil {
		t.Errorf("Unexpected error watching recycler pod: %+v", err)
	}
	if !client.deletedCalled {
		t.Errorf("Expected deferred client.Delete to be called on recycler pod")
	}
}

func TestRecyclerFailure(t *testing.T) {
	client := &mockRecyclerClient{}
	recycler := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "recycler-test",
			Namespace: api.NamespaceDefault,
		},
		Status: api.PodStatus{
			Phase:   api.PodFailed,
			Message: "foo",
		},
	}

	err := internalRecycleVolumeByWatchingPodUntilCompletion(recycler, client)
	if err == nil {
		t.Fatalf("Expected pod failure but got nil error returned")
	}
	if err != nil {
		if !strings.Contains(err.Error(), "foo") {
			t.Errorf("Expected pod.Status.Message %s but got %s", recycler.Status.Message, err)
		}
	}
	if !client.deletedCalled {
		t.Errorf("Expected deferred client.Delete to be called on recycler pod")
	}
}

type mockRecyclerClient struct {
	pod           *api.Pod
	deletedCalled bool
}

func (c *mockRecyclerClient) CreatePod(pod *api.Pod) (*api.Pod, error) {
	c.pod = pod
	return c.pod, nil
}

func (c *mockRecyclerClient) GetPod(name, namespace string) (*api.Pod, error) {
	if c.pod != nil {
		return c.pod, nil
	} else {
		return nil, fmt.Errorf("pod does not exist")
	}
}

func (c *mockRecyclerClient) DeletePod(name, namespace string) error {
	c.deletedCalled = true
	return nil
}

func (c *mockRecyclerClient) WatchPod(name, namespace, resourceVersion string, stopChannel chan struct{}) func() *api.Pod {
	return func() *api.Pod {
		return c.pod
	}
}

func TestCalculateTimeoutForVolume(t *testing.T) {
	pv := &api.PersistentVolume{
		Spec: api.PersistentVolumeSpec{
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("500M"),
			},
		},
	}

	timeout := CalculateTimeoutForVolume(50, 30, pv)
	if timeout != 50 {
		t.Errorf("Expected 50 for timeout but got %v", timeout)
	}

	pv.Spec.Capacity[api.ResourceStorage] = resource.MustParse("2Gi")
	timeout = CalculateTimeoutForVolume(50, 30, pv)
	if timeout != 60 {
		t.Errorf("Expected 60 for timeout but got %v", timeout)
	}

	pv.Spec.Capacity[api.ResourceStorage] = resource.MustParse("150Gi")
	timeout = CalculateTimeoutForVolume(50, 30, pv)
	if timeout != 4500 {
		t.Errorf("Expected 4500 for timeout but got %v", timeout)
	}
}
