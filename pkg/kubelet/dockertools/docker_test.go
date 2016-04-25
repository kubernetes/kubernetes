/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"hash/adler32"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/docker/docker/pkg/jsonmessage"
	dockertypes "github.com/docker/engine-api/types"
	dockernat "github.com/docker/go-connections/nat"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/types"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
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

func findPodContainer(dockerContainers []*dockertypes.Container, podFullName string, uid types.UID, containerName string) (*dockertypes.Container, bool, uint64) {
	for _, dockerContainer := range dockerContainers {
		if len(dockerContainer.Names) == 0 {
			continue
		}
		dockerName, hash, err := ParseDockerName(dockerContainer.Names[0])
		if err != nil {
			continue
		}
		if dockerName.PodFullName == podFullName &&
			(uid == "" || dockerName.PodUID == uid) &&
			dockerName.ContainerName == containerName {
			return dockerContainer, true, hash
		}
	}
	return nil, false, 0
}

func TestGetContainerID(t *testing.T) {
	fakeDocker := NewFakeDockerClient()
	fakeDocker.SetFakeRunningContainers([]*FakeContainer{
		{
			ID:   "foobar",
			Name: "/k8s_foo_qux_ns_1234_42",
		},
		{
			ID:   "barbar",
			Name: "/k8s_bar_qux_ns_2565_42",
		},
	})

	dockerContainers, err := GetKubeletDockerContainers(fakeDocker, false)
	if err != nil {
		t.Errorf("Expected no error, Got %#v", err)
	}
	if len(dockerContainers) != 2 {
		t.Errorf("Expected %#v, Got %#v", fakeDocker.RunningContainerList, dockerContainers)
	}
	verifyCalls(t, fakeDocker, []string{"list"})

	dockerContainer, found, _ := findPodContainer(dockerContainers, "qux_ns", "", "foo")
	if dockerContainer == nil || !found {
		t.Errorf("Failed to find container %#v", dockerContainer)
	}

	fakeDocker.ClearCalls()
	dockerContainer, found, _ = findPodContainer(dockerContainers, "foobar", "", "foo")
	verifyCalls(t, fakeDocker, []string{})
	if dockerContainer != nil || found {
		t.Errorf("Should not have found container %#v", dockerContainer)
	}
}

func verifyPackUnpack(t *testing.T, podNamespace, podUID, podName, containerName string) {
	container := &api.Container{Name: containerName}
	hasher := adler32.New()
	hashutil.DeepHashObject(hasher, *container)
	computedHash := uint64(hasher.Sum32())
	podFullName := fmt.Sprintf("%s_%s", podName, podNamespace)
	_, name, _ := BuildDockerName(KubeletContainerName{podFullName, types.UID(podUID), container.Name}, container)
	returned, hash, err := ParseDockerName(name)
	if err != nil {
		t.Errorf("Failed to parse Docker container name %q: %v", name, err)
	}
	if podFullName != returned.PodFullName || podUID != string(returned.PodUID) || containerName != returned.ContainerName || computedHash != hash {
		t.Errorf("For (%s, %s, %s, %d), unpacked (%s, %s, %s, %d)", podFullName, podUID, containerName, computedHash, returned.PodFullName, returned.PodUID, returned.ContainerName, hash)
	}
}

func TestContainerNaming(t *testing.T) {
	podUID := "12345678"
	verifyPackUnpack(t, "file", podUID, "name", "container")
	verifyPackUnpack(t, "file", podUID, "name-with-dashes", "container")
	// UID is same as pod name
	verifyPackUnpack(t, "file", podUID, podUID, "container")
	// No Container name
	verifyPackUnpack(t, "other", podUID, "name", "")

	container := &api.Container{Name: "container"}
	podName := "foo"
	podNamespace := "test"
	name := fmt.Sprintf("k8s_%s_%s_%s_%s_42", container.Name, podName, podNamespace, podUID)
	podFullName := fmt.Sprintf("%s_%s", podName, podNamespace)

	returned, hash, err := ParseDockerName(name)
	if err != nil {
		t.Errorf("Failed to parse Docker container name %q: %v", name, err)
	}
	if returned.PodFullName != podFullName || string(returned.PodUID) != podUID || returned.ContainerName != container.Name || hash != 0 {
		t.Errorf("unexpected parse: %s %s %s %d", returned.PodFullName, returned.PodUID, returned.ContainerName, hash)
	}
}

func TestApplyDefaultImageTag(t *testing.T) {
	for _, testCase := range []struct {
		Input  string
		Output string
	}{
		{Input: "root", Output: "root:latest"},
		{Input: "root:tag", Output: "root:tag"},
		{Input: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", Output: "root@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
	} {
		image, err := applyDefaultImageTag(testCase.Input)
		if err != nil {
			t.Errorf("applyDefaultTag(%s) failed: %v", testCase.Input, err)
		} else if image != testCase.Output {
			t.Errorf("Expected image reference: %q, got %q", testCase.Output, image)
		}
	}
}

func TestPullWithNoSecrets(t *testing.T) {
	tests := []struct {
		imageName     string
		expectedImage string
	}{
		{"ubuntu", "ubuntu:latest using {}"},
		{"ubuntu:2342", "ubuntu:2342 using {}"},
		{"ubuntu:latest", "ubuntu:latest using {}"},
		{"foo/bar:445566", "foo/bar:445566 using {}"},
		{"registry.example.com:5000/foobar", "registry.example.com:5000/foobar:latest using {}"},
		{"registry.example.com:5000/foobar:5342", "registry.example.com:5000/foobar:5342 using {}"},
		{"registry.example.com:5000/foobar:latest", "registry.example.com:5000/foobar:latest using {}"},
	}
	for _, test := range tests {
		fakeKeyring := &credentialprovider.FakeKeyring{}
		fakeClient := NewFakeDockerClient()

		dp := dockerPuller{
			client:  fakeClient,
			keyring: fakeKeyring,
		}

		err := dp.Pull(test.imageName, []api.Secret{})
		if err != nil {
			t.Errorf("unexpected non-nil err: %s", err)
			continue
		}

		if e, a := 1, len(fakeClient.pulled); e != a {
			t.Errorf("%s: expected 1 pulled image, got %d: %v", test.imageName, a, fakeClient.pulled)
			continue
		}

		if e, a := test.expectedImage, fakeClient.pulled[0]; e != a {
			t.Errorf("%s: expected pull of %q, but got %q", test.imageName, e, a)
		}
	}
}

func TestPullWithJSONError(t *testing.T) {
	tests := map[string]struct {
		imageName     string
		err           error
		expectedError string
	}{
		"Json error": {
			"ubuntu",
			&jsonmessage.JSONError{Code: 50, Message: "Json error"},
			"Json error",
		},
		"Bad gateway": {
			"ubuntu",
			&jsonmessage.JSONError{Code: 502, Message: "<!doctype html>\n<html class=\"no-js\" lang=\"\">\n    <head>\n  </head>\n    <body>\n   <h1>Oops, there was an error!</h1>\n        <p>We have been contacted of this error, feel free to check out <a href=\"http://status.docker.com/\">status.docker.com</a>\n           to see if there is a bigger issue.</p>\n\n    </body>\n</html>"},
			kubecontainer.RegistryUnavailable.Error(),
		},
	}
	for i, test := range tests {
		fakeKeyring := &credentialprovider.FakeKeyring{}
		fakeClient := NewFakeDockerClient()
		fakeClient.InjectError("pull", test.err)

		puller := &dockerPuller{
			client:  fakeClient,
			keyring: fakeKeyring,
		}
		err := puller.Pull(test.imageName, []api.Secret{})
		if err == nil || !strings.Contains(err.Error(), test.expectedError) {
			t.Errorf("%s: expect error %s, got : %s", i, test.expectedError, err)
			continue
		}
	}
}

func TestPullWithSecrets(t *testing.T) {
	// auth value is equivalent to: "username":"passed-user","password":"passed-password"
	dockerCfg := map[string]map[string]string{"index.docker.io/v1/": {"email": "passed-email", "auth": "cGFzc2VkLXVzZXI6cGFzc2VkLXBhc3N3b3Jk"}}
	dockercfgContent, err := json.Marshal(dockerCfg)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	dockerConfigJson := map[string]map[string]map[string]string{"auths": dockerCfg}
	dockerConfigJsonContent, err := json.Marshal(dockerConfigJson)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := map[string]struct {
		imageName           string
		passedSecrets       []api.Secret
		builtInDockerConfig credentialprovider.DockerConfig
		expectedPulls       []string
	}{
		"no matching secrets": {
			"ubuntu",
			[]api.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{}),
			[]string{"ubuntu:latest using {}"},
		},
		"default keyring secrets": {
			"ubuntu",
			[]api.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{"index.docker.io/v1/": {"built-in", "password", "email", nil}}),
			[]string{`ubuntu:latest using {"username":"built-in","password":"password","email":"email"}`},
		},
		"default keyring secrets unused": {
			"ubuntu",
			[]api.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{"extraneous": {"built-in", "password", "email", nil}}),
			[]string{`ubuntu:latest using {}`},
		},
		"builtin keyring secrets, but use passed": {
			"ubuntu",
			[]api.Secret{{Type: api.SecretTypeDockercfg, Data: map[string][]byte{api.DockerConfigKey: dockercfgContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{"index.docker.io/v1/": {"built-in", "password", "email", nil}}),
			[]string{`ubuntu:latest using {"username":"passed-user","password":"passed-password","email":"passed-email"}`},
		},
		"builtin keyring secrets, but use passed with new docker config": {
			"ubuntu",
			[]api.Secret{{Type: api.SecretTypeDockerConfigJson, Data: map[string][]byte{api.DockerConfigJsonKey: dockerConfigJsonContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{"index.docker.io/v1/": {"built-in", "password", "email", nil}}),
			[]string{`ubuntu:latest using {"username":"passed-user","password":"passed-password","email":"passed-email"}`},
		},
	}
	for i, test := range tests {
		builtInKeyRing := &credentialprovider.BasicDockerKeyring{}
		builtInKeyRing.Add(test.builtInDockerConfig)

		fakeClient := NewFakeDockerClient()

		dp := dockerPuller{
			client:  fakeClient,
			keyring: builtInKeyRing,
		}

		err := dp.Pull(test.imageName, test.passedSecrets)
		if err != nil {
			t.Errorf("%s: unexpected non-nil err: %s", i, err)
			continue
		}

		if e, a := 1, len(fakeClient.pulled); e != a {
			t.Errorf("%s: expected 1 pulled image, got %d: %v", i, a, fakeClient.pulled)
			continue
		}

		if e, a := test.expectedPulls, fakeClient.pulled; !reflect.DeepEqual(e, a) {
			t.Errorf("%s: expected pull of %v, but got %v", i, e, a)
		}
	}
}

func TestDockerKeyringLookupFails(t *testing.T) {
	fakeKeyring := &credentialprovider.FakeKeyring{}
	fakeClient := NewFakeDockerClient()
	fakeClient.InjectError("pull", fmt.Errorf("test error"))

	dp := dockerPuller{
		client:  fakeClient,
		keyring: fakeKeyring,
	}

	err := dp.Pull("host/repository/image:version", []api.Secret{})
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	msg := "image pull failed for host/repository/image:version, this may be because there are no credentials on this request.  details: (test error)"
	if err.Error() != msg {
		t.Errorf("expected: %s, saw: %s", msg, err.Error())
	}
}

func TestDockerKeyringLookup(t *testing.T) {
	ada := credentialprovider.LazyAuthConfiguration{
		AuthConfig: dockertypes.AuthConfig{
			Username: "ada",
			Password: "smash",
			Email:    "ada@example.com",
		},
	}

	grace := credentialprovider.LazyAuthConfiguration{
		AuthConfig: dockertypes.AuthConfig{
			Username: "grace",
			Password: "squash",
			Email:    "grace@example.com",
		},
	}

	dk := &credentialprovider.BasicDockerKeyring{}
	dk.Add(credentialprovider.DockerConfig{
		"bar.example.com/pong": credentialprovider.DockerConfigEntry{
			Username: grace.Username,
			Password: grace.Password,
			Email:    grace.Email,
		},
		"bar.example.com": credentialprovider.DockerConfigEntry{
			Username: ada.Username,
			Password: ada.Password,
			Email:    ada.Email,
		},
	})

	tests := []struct {
		image string
		match []credentialprovider.LazyAuthConfiguration
		ok    bool
	}{
		// direct match
		{"bar.example.com", []credentialprovider.LazyAuthConfiguration{ada}, true},

		// direct match deeper than other possible matches
		{"bar.example.com/pong", []credentialprovider.LazyAuthConfiguration{grace, ada}, true},

		// no direct match, deeper path ignored
		{"bar.example.com/ping", []credentialprovider.LazyAuthConfiguration{ada}, true},

		// match first part of path token
		{"bar.example.com/pongz", []credentialprovider.LazyAuthConfiguration{grace, ada}, true},

		// match regardless of sub-path
		{"bar.example.com/pong/pang", []credentialprovider.LazyAuthConfiguration{grace, ada}, true},

		// no host match
		{"example.com", []credentialprovider.LazyAuthConfiguration{}, false},
		{"foo.example.com", []credentialprovider.LazyAuthConfiguration{}, false},
	}

	for i, tt := range tests {
		match, ok := dk.Lookup(tt.image)
		if tt.ok != ok {
			t.Errorf("case %d: expected ok=%t, got %t", i, tt.ok, ok)
		}

		if !reflect.DeepEqual(tt.match, match) {
			t.Errorf("case %d: expected match=%#v, got %#v", i, tt.match, match)
		}
	}
}

// This validates that dockercfg entries with a scheme and url path are properly matched
// by images that only match the hostname.
// NOTE: the above covers the case of a more specific match trumping just hostname.
func TestIssue3797(t *testing.T) {
	rex := credentialprovider.LazyAuthConfiguration{
		AuthConfig: dockertypes.AuthConfig{
			Username: "rex",
			Password: "tiny arms",
			Email:    "rex@example.com",
		},
	}

	dk := &credentialprovider.BasicDockerKeyring{}
	dk.Add(credentialprovider.DockerConfig{
		"https://quay.io/v1/": credentialprovider.DockerConfigEntry{
			Username: rex.Username,
			Password: rex.Password,
			Email:    rex.Email,
		},
	})

	tests := []struct {
		image string
		match []credentialprovider.LazyAuthConfiguration
		ok    bool
	}{
		// direct match
		{"quay.io", []credentialprovider.LazyAuthConfiguration{rex}, true},

		// partial matches
		{"quay.io/foo", []credentialprovider.LazyAuthConfiguration{rex}, true},
		{"quay.io/foo/bar", []credentialprovider.LazyAuthConfiguration{rex}, true},
	}

	for i, tt := range tests {
		match, ok := dk.Lookup(tt.image)
		if tt.ok != ok {
			t.Errorf("case %d: expected ok=%t, got %t", i, tt.ok, ok)
		}

		if !reflect.DeepEqual(tt.match, match) {
			t.Errorf("case %d: expected match=%#v, got %#v", i, tt.match, match)
		}
	}
}

type imageTrackingDockerClient struct {
	*FakeDockerClient
	imageName string
}

func (f *imageTrackingDockerClient) InspectImage(name string) (image *dockertypes.ImageInspect, err error) {
	image, err = f.FakeDockerClient.InspectImage(name)
	f.imageName = name
	return
}

func TestIsImagePresent(t *testing.T) {
	cl := &imageTrackingDockerClient{NewFakeDockerClient(), ""}
	puller := &dockerPuller{
		client: cl,
	}
	_, _ = puller.IsImagePresent("abc:123")
	if cl.imageName != "abc:123" {
		t.Errorf("expected inspection of image abc:123, instead inspected image %v", cl.imageName)
	}
}

type podsByID []*kubecontainer.Pod

func (b podsByID) Len() int           { return len(b) }
func (b podsByID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b podsByID) Less(i, j int) bool { return b[i].ID < b[j].ID }

type containersByID []*kubecontainer.Container

func (b containersByID) Len() int           { return len(b) }
func (b containersByID) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b containersByID) Less(i, j int) bool { return b[i].ID.ID < b[j].ID.ID }

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
	np, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", nettest.NewFakeHost(nil))
	// image back-off is set to nil, this test should not pull images
	containerManager := NewFakeDockerManager(fakeClient, &record.FakeRecorder{}, nil, nil, &cadvisorapi.MachineInfo{}, options.GetDefaultPodInfraContainerImage(), 0, 0, "", containertest.FakeOS{}, np, nil, nil, nil)
	for i, test := range tests {
		fakeClient.RunningContainerList = test.runningContainerList
		fakeClient.ExitedContainerList = test.exitedContainerList

		result, _ := containerManager.GetPods(test.all)
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

func TestMakePortsAndBindings(t *testing.T) {
	ports := []kubecontainer.PortMapping{
		{
			ContainerPort: 80,
			HostPort:      8080,
			HostIP:        "127.0.0.1",
		},
		{
			ContainerPort: 443,
			HostPort:      443,
			Protocol:      "tcp",
		},
		{
			ContainerPort: 444,
			HostPort:      444,
			Protocol:      "udp",
		},
		{
			ContainerPort: 445,
			HostPort:      445,
			Protocol:      "foobar",
		},
		{
			ContainerPort: 443,
			HostPort:      446,
			Protocol:      "tcp",
		},
		{
			ContainerPort: 443,
			HostPort:      446,
			Protocol:      "udp",
		},
	}

	exposedPorts, bindings := makePortsAndBindings(ports)

	// Count the expected exposed ports and bindings
	expectedExposedPorts := map[string]struct{}{}

	for _, binding := range ports {
		dockerKey := strconv.Itoa(binding.ContainerPort) + "/" + string(binding.Protocol)
		expectedExposedPorts[dockerKey] = struct{}{}
	}

	// Should expose right ports in docker
	if len(expectedExposedPorts) != len(exposedPorts) {
		t.Errorf("Unexpected ports and bindings, %#v %#v %#v", ports, exposedPorts, bindings)
	}

	// Construct expected bindings
	expectPortBindings := map[string][]dockernat.PortBinding{
		"80/tcp": {
			dockernat.PortBinding{
				HostPort: "8080",
				HostIP:   "127.0.0.1",
			},
		},
		"443/tcp": {
			dockernat.PortBinding{
				HostPort: "443",
				HostIP:   "",
			},
			dockernat.PortBinding{
				HostPort: "446",
				HostIP:   "",
			},
		},
		"443/udp": {
			dockernat.PortBinding{
				HostPort: "446",
				HostIP:   "",
			},
		},
		"444/udp": {
			dockernat.PortBinding{
				HostPort: "444",
				HostIP:   "",
			},
		},
		"445/tcp": {
			dockernat.PortBinding{
				HostPort: "445",
				HostIP:   "",
			},
		},
	}

	// interate the bindings by dockerPort, and check its portBindings
	for dockerPort, portBindings := range bindings {
		switch dockerPort {
		case "80/tcp", "443/tcp", "443/udp", "444/udp", "445/tcp":
			if !reflect.DeepEqual(expectPortBindings[string(dockerPort)], portBindings) {
				t.Errorf("Unexpected portbindings for %#v, expected: %#v, but got: %#v",
					dockerPort, expectPortBindings[string(dockerPort)], portBindings)
			}
		default:
			t.Errorf("Unexpected docker port: %#v with portbindings: %#v", dockerPort, portBindings)
		}
	}
}

func TestMilliCPUToQuota(t *testing.T) {
	testCases := []struct {
		input  int64
		quota  int64
		period int64
	}{
		{
			input:  int64(0),
			quota:  int64(0),
			period: int64(0),
		},
		{
			input:  int64(5),
			quota:  int64(1000),
			period: int64(100000),
		},
		{
			input:  int64(9),
			quota:  int64(1000),
			period: int64(100000),
		},
		{
			input:  int64(10),
			quota:  int64(1000),
			period: int64(100000),
		},
		{
			input:  int64(200),
			quota:  int64(20000),
			period: int64(100000),
		},
		{
			input:  int64(500),
			quota:  int64(50000),
			period: int64(100000),
		},
		{
			input:  int64(1000),
			quota:  int64(100000),
			period: int64(100000),
		},
		{
			input:  int64(1500),
			quota:  int64(150000),
			period: int64(100000),
		},
	}
	for _, testCase := range testCases {
		quota, period := milliCPUToQuota(testCase.input)
		if quota != testCase.quota || period != testCase.period {
			t.Errorf("Input %v, expected quota %v period %v, but got quota %v period %v", testCase.input, testCase.quota, testCase.period, quota, period)
		}
	}
}
