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
	"math/rand"
	"path"
	"strings"
	"testing"

	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/images"
)

// TODO: Examine the tests and see if they can be migrated to kuberuntime.
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
		fakeClient := libdocker.NewFakeDockerClient()

		dp := dockerPuller{
			client:  fakeClient,
			keyring: fakeKeyring,
		}

		err := dp.Pull(test.imageName, []v1.Secret{})
		if err != nil {
			t.Errorf("unexpected non-nil err: %s", err)
			continue
		}

		if err := fakeClient.AssertImagesPulled([]string{test.imageName}); err != nil {
			t.Errorf("images pulled do not match the expected: %v", err)
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
		fakeClient := libdocker.NewFakeDockerClient()
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

		fakeClient := libdocker.NewFakeDockerClient()

		dp := dockerPuller{
			client:  fakeClient,
			keyring: builtInKeyRing,
		}

		err := dp.Pull(test.imageName, test.passedSecrets)
		if err != nil {
			t.Errorf("%s: unexpected non-nil err: %s", i, err)
			continue
		}
		if err := fakeClient.AssertImagesPulledMsgs(test.expectedPulls); err != nil {
			t.Errorf("images pulled do not match the expected: %v", err)
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
