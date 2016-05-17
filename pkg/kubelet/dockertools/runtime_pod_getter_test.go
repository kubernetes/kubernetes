/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"reflect"
	"sort"
	"testing"

	dockertypes "github.com/docker/engine-api/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

func TestFindContainersByPod(t *testing.T) {
	tests := []struct {
		runningContainerList []dockertypes.Container
		exitedContainerList  []dockertypes.Container
		all                  bool
		expectedPods         []*kubecontainer.Pod
	}{

		{
			[]dockertypes.Container{
				{
					ID:    "foobar",
					Names: []string{"/k8s_foobar.1234_qux_ns_1234_42"},
				},
				{
					ID:    "barbar",
					Names: []string{"/k8s_barbar.1234_qux_ns_2343_42"},
				},
				{
					ID:    "baz",
					Names: []string{"/k8s_baz.1234_qux_ns_1234_42"},
				},
			},
			[]dockertypes.Container{
				{
					ID:    "barfoo",
					Names: []string{"/k8s_barfoo.1234_qux_ns_1234_42"},
				},
				{
					ID:    "bazbaz",
					Names: []string{"/k8s_bazbaz.1234_qux_ns_5678_42"},
				},
			},
			false,
			[]*kubecontainer.Pod{
				{
					ID:        "1234",
					Name:      "qux",
					Namespace: "ns",
					Containers: []*kubecontainer.Container{
						{
							ID:    kubecontainer.DockerID("foobar").ContainerID(),
							Name:  "foobar",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
						{
							ID:    kubecontainer.DockerID("baz").ContainerID(),
							Name:  "baz",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
					},
				},
				{
					ID:        "2343",
					Name:      "qux",
					Namespace: "ns",
					Containers: []*kubecontainer.Container{
						{
							ID:    kubecontainer.DockerID("barbar").ContainerID(),
							Name:  "barbar",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
					},
				},
			},
		},
		{
			[]dockertypes.Container{
				{
					ID:    "foobar",
					Names: []string{"/k8s_foobar.1234_qux_ns_1234_42"},
				},
				{
					ID:    "barbar",
					Names: []string{"/k8s_barbar.1234_qux_ns_2343_42"},
				},
				{
					ID:    "baz",
					Names: []string{"/k8s_baz.1234_qux_ns_1234_42"},
				},
			},
			[]dockertypes.Container{
				{
					ID:    "barfoo",
					Names: []string{"/k8s_barfoo.1234_qux_ns_1234_42"},
				},
				{
					ID:    "bazbaz",
					Names: []string{"/k8s_bazbaz.1234_qux_ns_5678_42"},
				},
			},
			true,
			[]*kubecontainer.Pod{
				{
					ID:        "1234",
					Name:      "qux",
					Namespace: "ns",
					Containers: []*kubecontainer.Container{
						{
							ID:    kubecontainer.DockerID("foobar").ContainerID(),
							Name:  "foobar",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
						{
							ID:    kubecontainer.DockerID("barfoo").ContainerID(),
							Name:  "barfoo",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
						{
							ID:    kubecontainer.DockerID("baz").ContainerID(),
							Name:  "baz",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
					},
				},
				{
					ID:        "2343",
					Name:      "qux",
					Namespace: "ns",
					Containers: []*kubecontainer.Container{
						{
							ID:    kubecontainer.DockerID("barbar").ContainerID(),
							Name:  "barbar",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
					},
				},
				{
					ID:        "5678",
					Name:      "qux",
					Namespace: "ns",
					Containers: []*kubecontainer.Container{
						{
							ID:    kubecontainer.DockerID("bazbaz").ContainerID(),
							Name:  "bazbaz",
							Hash:  0x1234,
							State: kubecontainer.ContainerStateUnknown,
						},
					},
				},
			},
		},
		{
			[]dockertypes.Container{},
			[]dockertypes.Container{},
			true,
			nil,
		},
	}
	fakeClient := NewFakeDockerClient()
	runtimePodGetter := NewRuntimePodGetter(fakeClient)
	// image back-off is set to nil, this test should not pull images
	for i, test := range tests {
		fakeClient.RunningContainerList = test.runningContainerList
		fakeClient.ExitedContainerList = test.exitedContainerList

		result, _ := runtimePodGetter.GetPods(test.all)
		for i := range result {
			sort.Sort(containersByID(result[i].Containers))
		}
		for i := range test.expectedPods {
			sort.Sort(containersByID(test.expectedPods[i].Containers))
		}
		sort.Sort(podsByID(result))
		sort.Sort(podsByID(test.expectedPods))
		if !reflect.DeepEqual(test.expectedPods, result) {
			t.Errorf("%d: expected: %#v, saw: %#v", i, test.expectedPods, result)
		}
	}
}

func TestGetPods(t *testing.T) {
	fakeClient := NewFakeDockerClient()
	runtimePodGetter := NewRuntimePodGetter(fakeClient)
	dockerContainers := []dockertypes.Container{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
		{
			ID:    "3333",
			Names: []string{"/k8s_bar_jlk_wen_5678_42"},
		},
	}

	// Convert the docker containers. This does not affect the test coverage
	// because the conversion is tested separately in convert_test.go
	containers := make([]*kubecontainer.Container, len(dockerContainers))
	for i, cont := range dockerContainers {
		c, err := toRuntimeContainer(&cont)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		containers[i] = c
	}

	expected := []*kubecontainer.Pod{
		{
			ID:         types.UID("1234"),
			Name:       "qux",
			Namespace:  "new",
			Containers: []*kubecontainer.Container{containers[0], containers[1]},
		},
		{
			ID:         types.UID("5678"),
			Name:       "jlk",
			Namespace:  "wen",
			Containers: []*kubecontainer.Container{containers[2]},
		},
	}
	fakeClient.RunningContainerList = dockerContainers
	actual, err := runtimePodGetter.GetPods(false)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !verifyPods(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}
