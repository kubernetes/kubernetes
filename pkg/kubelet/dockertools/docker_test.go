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
	"hash/adler32"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/fsouza/go-dockerclient"
)

func verifyCalls(t *testing.T, fakeDocker *FakeDockerClient, calls []string) {
	fakeDocker.Lock()
	defer fakeDocker.Unlock()
	verifyStringArrayEquals(t, fakeDocker.called, calls)
}

func verifyStringArrayEquals(t *testing.T, actual, expected []string) {
	invalid := len(actual) != len(expected)
	if !invalid {
		for ix, value := range actual {
			if expected[ix] != value {
				invalid = true
			}
		}
	}
	if invalid {
		t.Errorf("Expected: %#v, Actual: %#v", expected, actual)
	}
}

func TestGetContainerID(t *testing.T) {
	fakeDocker := &FakeDockerClient{}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "foobar",
			Names: []string{"/k8s--foo--qux--1234"},
		},
		{
			ID:    "barbar",
			Names: []string{"/k8s--bar--qux--2565"},
		},
	}
	fakeDocker.Container = &docker.Container{
		ID: "foobar",
	}

	dockerContainers, err := GetKubeletDockerContainers(fakeDocker)
	if err != nil {
		t.Errorf("Expected no error, Got %#v", err)
	}
	if len(dockerContainers) != 2 {
		t.Errorf("Expected %#v, Got %#v", fakeDocker.ContainerList, dockerContainers)
	}
	verifyCalls(t, fakeDocker, []string{"list"})
	dockerContainer, found, _ := dockerContainers.FindPodContainer("qux", "", "foo")
	if dockerContainer == nil || !found {
		t.Errorf("Failed to find container %#v", dockerContainer)
	}

	fakeDocker.clearCalls()
	dockerContainer, found, _ = dockerContainers.FindPodContainer("foobar", "", "foo")
	verifyCalls(t, fakeDocker, []string{})
	if dockerContainer != nil || found {
		t.Errorf("Should not have found container %#v", dockerContainer)
	}
}

func verifyPackUnpack(t *testing.T, podNamespace, podName, containerName string) {
	container := &api.Container{Name: containerName}
	hasher := adler32.New()
	data := fmt.Sprintf("%#v", *container)
	hasher.Write([]byte(data))
	computedHash := uint64(hasher.Sum32())
	podFullName := fmt.Sprintf("%s.%s", podName, podNamespace)
	name := BuildDockerName("", podFullName, container)
	returnedPodFullName, _, returnedContainerName, hash := ParseDockerName(name)
	if podFullName != returnedPodFullName || containerName != returnedContainerName || computedHash != hash {
		t.Errorf("For (%s, %s, %d), unpacked (%s, %s, %d)", podFullName, containerName, computedHash, returnedPodFullName, returnedContainerName, hash)
	}
}

func TestContainerManifestNaming(t *testing.T) {
	verifyPackUnpack(t, "file", "manifest1234", "container5678")
	verifyPackUnpack(t, "file", "manifest--", "container__")
	verifyPackUnpack(t, "file", "--manifest", "__container")
	verifyPackUnpack(t, "", "m___anifest_", "container-_-")
	verifyPackUnpack(t, "other", "_m___anifest", "-_-container")

	container := &api.Container{Name: "container"}
	podName := "foo"
	podNamespace := "test"
	name := fmt.Sprintf("k8s--%s--%s.%s--12345", container.Name, podName, podNamespace)

	podFullName := fmt.Sprintf("%s.%s", podName, podNamespace)
	returnedPodFullName, _, returnedContainerName, hash := ParseDockerName(name)
	if returnedPodFullName != podFullName || returnedContainerName != container.Name || hash != 0 {
		t.Errorf("unexpected parse: %s %s %d", returnedPodFullName, returnedContainerName, hash)
	}
}

func TestDockerContainerCommand(t *testing.T) {
	runner := dockerContainerCommandRunner{}
	containerID := "1234"
	command := []string{"ls"}
	cmd, _ := runner.getRunInContainerCommand(containerID, command)
	if cmd.Dir != "/var/lib/docker/execdriver/native/"+containerID {
		t.Errorf("unexpected command CWD: %s", cmd.Dir)
	}
	if !reflect.DeepEqual(cmd.Args, []string{"/usr/sbin/nsinit", "exec", "ls"}) {
		t.Errorf("unexpectd command args: %s", cmd.Args)
	}
}

var parseImageNameTests = []struct {
	imageName string
	name      string
	tag       string
}{
	{"ubuntu", "ubuntu", ""},
	{"ubuntu:2342", "ubuntu", "2342"},
	{"ubuntu:latest", "ubuntu", "latest"},
	{"foo/bar:445566", "foo/bar", "445566"},
	{"registry.example.com:5000/foobar", "registry.example.com:5000/foobar", ""},
	{"registry.example.com:5000/foobar:5342", "registry.example.com:5000/foobar", "5342"},
	{"registry.example.com:5000/foobar:latest", "registry.example.com:5000/foobar", "latest"},
}

func TestParseImageName(t *testing.T) {
	for _, tt := range parseImageNameTests {
		name, tag := parseImageName(tt.imageName)
		if name != tt.name || tag != tt.tag {
			t.Errorf("Expected name/tag: %s/%s, got %s/%s", tt.name, tt.tag, name, tag)
		}
	}
}
