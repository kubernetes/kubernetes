/*
Copyright 2014 The Kubernetes Authors.

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
	"math/rand"
	"path"
	"reflect"
	"strings"
	"testing"

	"github.com/docker/docker/pkg/jsonmessage"
	dockertypes "github.com/docker/engine-api/types"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/kubelet/images"
	hashutil "k8s.io/kubernetes/pkg/util/hash"
)

func verifyCalls(t *testing.T, fakeDocker *FakeDockerClient, calls []string) {
	assert.New(t).NoError(fakeDocker.AssertCalls(calls))
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
	container := &v1.Container{Name: containerName}
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

	container := &v1.Container{Name: "container"}
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

func TestMatchImageTagOrSHA(t *testing.T) {
	for i, testCase := range []struct {
		Inspected dockertypes.ImageInspect
		Image     string
		Output    bool
	}{
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"ubuntu:latest"}},
			Image:     "ubuntu",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"ubuntu:14.04"}},
			Image:     "ubuntu:latest",
			Output:    false,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"colemickens/hyperkube-amd64:217.9beff63"}},
			Image:     "colemickens/hyperkube-amd64:217.9beff63",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"colemickens/hyperkube-amd64:217.9beff63"}},
			Image:     "docker.io/colemickens/hyperkube-amd64:217.9beff63",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"docker.io/kubernetes/pause:latest"}},
			Image:     "kubernetes/pause:latest",
			Output:    true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005",
			Output: false,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208",
			Output: false,
		},
		{
			// mismatched ID is ignored
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:0000f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: false,
		},
		{
			// invalid digest is ignored
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:unparseable",
			},
			Image:  "myimage@sha256:unparseable",
			Output: false,
		},
		{
			// v1 schema images can be pulled in one format and returned in another
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// RepoDigest match is is required
			Inspected: dockertypes.ImageInspect{
				ID:          "",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:000084acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: false,
		},
		{
			// RepoDigest match is allowed
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// RepoDigest and ID are checked
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// unparseable RepoDigests are skipped
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{
					"centos/ruby-23-centos7@sha256:unparseable",
					"docker.io/centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
				},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: true,
		},
		{
			// unparseable RepoDigest is ignored
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:unparseable"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: false,
		},
		{
			// unparseable image digest is ignored
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:unparseable"},
			},
			Image:  "centos/ruby-23-centos7@sha256:unparseable",
			Output: false,
		},
		{
			// prefix match is rejected for ID and RepoDigest
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:unparseable",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:unparseable"},
			},
			Image:  "sha256:unparseable",
			Output: false,
		},
		{
			// possible SHA prefix match is rejected for ID and RepoDigest because it is not in the named format
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:0000f247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"docker.io/centos/ruby-23-centos7@sha256:0000f247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227"},
			},
			Image:  "sha256:0000",
			Output: false,
		},
	} {
		match := matchImageTagOrSHA(testCase.Inspected, testCase.Image)
		assert.Equal(t, testCase.Output, match, testCase.Image+fmt.Sprintf(" is not a match (%d)", i))
	}
}

func TestMatchImageIDOnly(t *testing.T) {
	for i, testCase := range []struct {
		Inspected dockertypes.ImageInspect
		Image     string
		Output    bool
	}{
		// shouldn't match names or tagged names
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"ubuntu:latest"}},
			Image:     "ubuntu",
			Output:    false,
		},
		{
			Inspected: dockertypes.ImageInspect{RepoTags: []string{"colemickens/hyperkube-amd64:217.9beff63"}},
			Image:     "colemickens/hyperkube-amd64:217.9beff63",
			Output:    false,
		},
		// should match name@digest refs if they refer to the image ID (but only the full ID)
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: true,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208f7a29005",
			Output: false,
		},
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:2208",
			Output: false,
		},
		// should match when the IDs are literally the same
		{
			Inspected: dockertypes.ImageInspect{
				ID: "foobar",
			},
			Image:  "foobar",
			Output: true,
		},
		// shouldn't match mismatched IDs
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:2208f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			},
			Image:  "myimage@sha256:0000f7a29005d226d1ee33a63e33af1f47af6156c740d7d23c7948e8d282d53d",
			Output: false,
		},
		// shouldn't match invalid IDs or refs
		{
			Inspected: dockertypes.ImageInspect{
				ID: "sha256:unparseable",
			},
			Image:  "myimage@sha256:unparseable",
			Output: false,
		},
		// shouldn't match against repo digests
		{
			Inspected: dockertypes.ImageInspect{
				ID:          "sha256:9bbdf247c91345f0789c10f50a57e36a667af1189687ad1de88a6243d05a2227",
				RepoDigests: []string{"centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf"},
			},
			Image:  "centos/ruby-23-centos7@sha256:940584acbbfb0347272112d2eb95574625c0c60b4e2fdadb139de5859cf754bf",
			Output: false,
		},
	} {
		match := matchImageIDOnly(testCase.Inspected, testCase.Image)
		assert.Equal(t, testCase.Output, match, fmt.Sprintf("%s is not a match (%d)", testCase.Image, i))
	}

}

func TestPullWithNoSecrets(t *testing.T) {
	tests := []struct {
		imageName     string
		expectedImage string
	}{
		{"ubuntu", "ubuntu using {}"},
		{"ubuntu:2342", "ubuntu:2342 using {}"},
		{"ubuntu:latest", "ubuntu:latest using {}"},
		{"foo/bar:445566", "foo/bar:445566 using {}"},
		{"registry.example.com:5000/foobar", "registry.example.com:5000/foobar using {}"},
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

		err := dp.Pull(test.imageName, []v1.Secret{})
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
			images.RegistryUnavailable.Error(),
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
		err := puller.Pull(test.imageName, []v1.Secret{})
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
		passedSecrets       []v1.Secret
		builtInDockerConfig credentialprovider.DockerConfig
		expectedPulls       []string
	}{
		"no matching secrets": {
			"ubuntu",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{}),
			[]string{"ubuntu using {}"},
		},
		"default keyring secrets": {
			"ubuntu",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Email: "email", Provider: nil},
			}),
			[]string{`ubuntu using {"username":"built-in","password":"password","email":"email"}`},
		},
		"default keyring secrets unused": {
			"ubuntu",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"extraneous": {Username: "built-in", Password: "password", Email: "email", Provider: nil},
			}),
			[]string{`ubuntu using {}`},
		},
		"builtin keyring secrets, but use passed": {
			"ubuntu",
			[]v1.Secret{{Type: v1.SecretTypeDockercfg, Data: map[string][]byte{v1.DockerConfigKey: dockercfgContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Email: "email", Provider: nil},
			}),
			[]string{`ubuntu using {"username":"passed-user","password":"passed-password","email":"passed-email"}`},
		},
		"builtin keyring secrets, but use passed with new docker config": {
			"ubuntu",
			[]v1.Secret{{Type: v1.SecretTypeDockerConfigJson, Data: map[string][]byte{v1.DockerConfigJsonKey: dockerConfigJsonContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Email: "email", Provider: nil},
			}),
			[]string{`ubuntu using {"username":"passed-user","password":"passed-password","email":"passed-email"}`},
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

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func randStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return string(b)
}

func TestLogSymLink(t *testing.T) {
	as := assert.New(t)
	containerLogsDir := "/foo/bar"
	podFullName := randStringBytes(128)
	containerName := randStringBytes(70)
	dockerId := randStringBytes(80)
	// The file name cannot exceed 255 characters. Since .log suffix is required, the prefix cannot exceed 251 characters.
	expectedPath := path.Join(containerLogsDir, fmt.Sprintf("%s_%s-%s", podFullName, containerName, dockerId)[:251]+".log")
	as.Equal(expectedPath, LogSymlink(containerLogsDir, podFullName, containerName, dockerId))
}
