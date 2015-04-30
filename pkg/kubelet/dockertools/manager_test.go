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

package dockertools

import (
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/fsouza/go-dockerclient"
)

func NewFakeDockerManager() (*DockerManager, *FakeDockerClient) {
	fakeDocker := &FakeDockerClient{Errors: make(map[string]error), RemovedImages: util.StringSet{}}
	fakeRecorder := &record.FakeRecorder{}
	readinessManager := kubecontainer.NewReadinessManager()
	containerRefManager := kubecontainer.NewRefManager()
	networkPlugin, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))

	dockerManager := NewDockerManager(
		fakeDocker,
		fakeRecorder,
		readinessManager,
		containerRefManager,
		PodInfraContainerImage,
		0, 0, "",
		kubecontainer.FakeOS{},
		networkPlugin,
		nil)

	return dockerManager, fakeDocker
}

func TestSetEntrypointAndCommand(t *testing.T) {
	cases := []struct {
		name      string
		container *api.Container
		expected  *docker.CreateContainerOptions
	}{
		{
			name:      "none",
			container: &api.Container{},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{},
			},
		},
		{
			name: "command",
			container: &api.Container{
				Command: []string{"foo", "bar"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo", "bar"},
				},
			},
		},
		{
			name: "args",
			container: &api.Container{
				Args: []string{"foo", "bar"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Cmd: []string{"foo", "bar"},
				},
			},
		},
		{
			name: "both",
			container: &api.Container{
				Command: []string{"foo"},
				Args:    []string{"bar", "baz"},
			},
			expected: &docker.CreateContainerOptions{
				Config: &docker.Config{
					Entrypoint: []string{"foo"},
					Cmd:        []string{"bar", "baz"},
				},
			},
		},
	}

	for _, tc := range cases {
		actualOpts := &docker.CreateContainerOptions{
			Config: &docker.Config{},
		}
		setEntrypointAndCommand(tc.container, actualOpts)

		if e, a := tc.expected.Config.Entrypoint, actualOpts.Config.Entrypoint; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected entrypoint: expected %v, got %v", tc.name, e, a)
		}
		if e, a := tc.expected.Config.Cmd, actualOpts.Config.Cmd; !api.Semantic.DeepEqual(e, a) {
			t.Errorf("%v: unexpected command: expected %v, got %v", tc.name, e, a)
		}
	}
}

// verifyPods returns true if the two pod slices are equal.
func verifyPods(a, b []*kubecontainer.Pod) bool {
	if len(a) != len(b) {
		return false
	}

	// Sort the containers within a pod.
	for i := range a {
		sort.Sort(containersByID(a[i].Containers))
	}
	for i := range b {
		sort.Sort(containersByID(b[i].Containers))
	}

	// Sort the pods by UID.
	sort.Sort(podsByID(a))
	sort.Sort(podsByID(b))

	return reflect.DeepEqual(a, b)
}

func TestGetPods(t *testing.T) {
	manager, fakeDocker := NewFakeDockerManager()
	dockerContainers := []docker.APIContainers{
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
	for i := range containers {
		c, err := toRuntimeContainer(&dockerContainers[i])
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

	fakeDocker.ContainerList = dockerContainers
	actual, err := manager.GetPods(false)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if !verifyPods(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

func TestListImages(t *testing.T) {
	manager, fakeDocker := NewFakeDockerManager()
	dockerImages := []docker.APIImages{{ID: "1111"}, {ID: "2222"}, {ID: "3333"}}
	expected := util.NewStringSet([]string{"1111", "2222", "3333"}...)

	fakeDocker.Images = dockerImages
	actualImages, err := manager.ListImages()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	actual := util.NewStringSet()
	for _, i := range actualImages {
		actual.Insert(i.ID)
	}
	// We can compare the two sets directly because util.StringSet.List()
	// returns a "sorted" list.
	if !reflect.DeepEqual(expected.List(), actual.List()) {
		t.Errorf("expected %#v, got %#v", expected.List(), actual.List())
	}
}

func apiContainerToContainer(c docker.APIContainers) kubecontainer.Container {
	dockerName, hash, err := ParseDockerName(c.Names[0])
	if err != nil {
		return kubecontainer.Container{}
	}
	return kubecontainer.Container{
		ID:   types.UID(c.ID),
		Name: dockerName.ContainerName,
		Hash: hash,
	}
}

func dockerContainersToPod(containers DockerContainers) kubecontainer.Pod {
	var pod kubecontainer.Pod
	for _, c := range containers {
		dockerName, hash, err := ParseDockerName(c.Names[0])
		if err != nil {
			continue
		}
		pod.Containers = append(pod.Containers, &kubecontainer.Container{
			ID:    types.UID(c.ID),
			Name:  dockerName.ContainerName,
			Hash:  hash,
			Image: c.Image,
		})
		// TODO(yifan): Only one evaluation is enough.
		pod.ID = dockerName.PodUID
		name, namespace, _ := kubecontainer.ParsePodFullName(dockerName.PodFullName)
		pod.Name = name
		pod.Namespace = namespace
	}
	return pod
}

func TestKillContainerInPod(t *testing.T) {
	manager, fakeDocker := NewFakeDockerManager()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "qux",
			Namespace: "new",
		},
		Spec: api.PodSpec{Containers: []api.Container{{Name: "foo"}, {Name: "bar"}}},
	}
	containers := []docker.APIContainers{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
	}
	containerToKill := &containers[0]
	containerToSpare := &containers[1]
	fakeDocker.ContainerList = containers
	// Set all containers to ready.
	for _, c := range fakeDocker.ContainerList {
		manager.readinessManager.SetReadiness(c.ID, true)
	}

	if err := manager.KillContainerInPod(pod.Spec.Containers[0], pod); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Assert the container has been stopped.
	if err := fakeDocker.AssertStopped([]string{containerToKill.ID}); err != nil {
		t.Errorf("container was not stopped correctly: %v", err)
	}

	// Verify that the readiness has been removed for the stopped container.
	if ready := manager.readinessManager.GetReadiness(containerToKill.ID); ready {
		t.Errorf("exepcted container entry ID '%v' to not be found. states: %+v", containerToKill.ID, ready)
	}
	if ready := manager.readinessManager.GetReadiness(containerToSpare.ID); !ready {
		t.Errorf("exepcted container entry ID '%v' to be found. states: %+v", containerToSpare.ID, ready)
	}
}

func TestKillContainerInPodWithError(t *testing.T) {
	manager, fakeDocker := NewFakeDockerManager()

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "qux",
			Namespace: "new",
		},
		Spec: api.PodSpec{Containers: []api.Container{{Name: "foo"}, {Name: "bar"}}},
	}
	containers := []docker.APIContainers{
		{
			ID:    "1111",
			Names: []string{"/k8s_foo_qux_new_1234_42"},
		},
		{
			ID:    "2222",
			Names: []string{"/k8s_bar_qux_new_1234_42"},
		},
	}
	containerToKill := &containers[0]
	containerToSpare := &containers[1]
	fakeDocker.ContainerList = containers
	fakeDocker.Errors["stop"] = fmt.Errorf("sample error")

	// Set all containers to ready.
	for _, c := range fakeDocker.ContainerList {
		manager.readinessManager.SetReadiness(c.ID, true)
	}

	if err := manager.KillContainerInPod(pod.Spec.Containers[0], pod); err == nil {
		t.Errorf("expected error, found nil")
	}

	// Verify that the readiness has been removed even though the stop failed.
	if ready := manager.readinessManager.GetReadiness(containerToKill.ID); ready {
		t.Errorf("exepcted container entry ID '%v' to not be found. states: %+v", containerToKill.ID, ready)
	}
	if ready := manager.readinessManager.GetReadiness(containerToSpare.ID); !ready {
		t.Errorf("exepcted container entry ID '%v' to be found. states: %+v", containerToSpare.ID, ready)
	}
}
