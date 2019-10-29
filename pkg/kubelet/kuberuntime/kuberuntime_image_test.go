/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"encoding/json"
	"path"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/utils/pointer"
)

func TestPullImage(t *testing.T) {
	_, _, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	imageRef, err := fakeManager.PullImage(kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "busybox", imageRef)

	images, err := fakeManager.ListImages()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(images))
	assert.Equal(t, images[0].RepoTags, []string{"busybox"})
}

func TestPullImageRuntimeHandler(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClass, true)()
	_, fakeImageService, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	rcm := runtimeclass.NewManager(rctest.NewPopulatedClient())
	defer rctest.StartManagerSync(rcm)()

	fakeManager.runtimeClassManager = rcm

	tests := map[string]struct {
		podNs           string
		podName         string
		rcn             *string
		expectedHandler string
		expectError     bool
	}{
		"Valid":   {podNs: "hello", podName: "world", rcn: pointer.StringPtr(rctest.SandboxRuntimeClass), expectedHandler: rctest.SandboxRuntimeHandler, expectError: false},
		"Empty":   {podNs: "hello1", podName: "world", rcn: nil, expectedHandler: "", expectError: false},
		"Invalid": {podNs: "hello2", podName: "world", rcn: pointer.StringPtr("invalid"), expectedHandler: "invalid", expectError: true},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			fakeImageService.Called = []string{}
			pod := newNamedTestPod(test.podNs, test.podName)
			pod.Spec.RuntimeClassName = test.rcn

			podSandboxConfig, err := fakeManager.generatePodSandboxConfig(pod, 1)
			if test.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)

				imageRef, err := fakeManager.PullImage(kubecontainer.ImageSpec{Image: "busybox"}, nil, podSandboxConfig)
				assert.NoError(t, err)
				assert.Contains(t, fakeImageService.Called, "PullImage")

				assert.Equal(t, "busybox", imageRef)

				m := fakeImageService.ListSandboxes()
				assert.Equal(t, test.expectedHandler, m[path.Join(test.podNs, test.podName)])
			}
		})
	}
}

func TestListImages(t *testing.T) {
	_, fakeImageService, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	images := []string{"1111", "2222", "3333"}
	expected := sets.NewString(images...)
	fakeImageService.SetFakeImages(images)

	actualImages, err := fakeManager.ListImages()
	assert.NoError(t, err)
	actual := sets.NewString()
	for _, i := range actualImages {
		actual.Insert(i.ID)
	}

	assert.Equal(t, expected.List(), actual.List())
}

func TestGetImageRef(t *testing.T) {
	_, fakeImageService, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	image := "busybox"
	fakeImageService.SetFakeImages([]string{image})
	imageRef, err := fakeManager.GetImageRef(kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)
	assert.Equal(t, image, imageRef)
}

func TestRemoveImage(t *testing.T) {
	_, fakeImageService, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	_, err = fakeManager.PullImage(kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(fakeImageService.Images))

	err = fakeManager.RemoveImage(kubecontainer.ImageSpec{Image: "busybox"})
	assert.NoError(t, err)
	assert.Equal(t, 0, len(fakeImageService.Images))
}

func TestImageStats(t *testing.T) {
	_, fakeImageService, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	const imageSize = 64
	fakeImageService.SetFakeImageSize(imageSize)
	images := []string{"1111", "2222", "3333"}
	fakeImageService.SetFakeImages(images)

	actualStats, err := fakeManager.ImageStats()
	assert.NoError(t, err)
	expectedStats := &kubecontainer.ImageStats{TotalStorageBytes: imageSize * uint64(len(images))}
	assert.Equal(t, expectedStats, actualStats)
}

func TestPullWithSecrets(t *testing.T) {
	// auth value is equivalent to: "username":"passed-user","password":"passed-password"
	dockerCfg := map[string]map[string]string{"index.docker.io/v1/": {"email": "passed-email", "auth": "cGFzc2VkLXVzZXI6cGFzc2VkLXBhc3N3b3Jk"}}
	dockercfgContent, err := json.Marshal(dockerCfg)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	dockerConfigJSON := map[string]map[string]map[string]string{"auths": dockerCfg}
	dockerConfigJSONContent, err := json.Marshal(dockerConfigJSON)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	tests := map[string]struct {
		imageName           string
		passedSecrets       []v1.Secret
		builtInDockerConfig credentialprovider.DockerConfig
		expectedAuth        *runtimeapi.AuthConfig
	}{
		"no matching secrets": {
			"ubuntu",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{}),
			nil,
		},
		"default keyring secrets": {
			"ubuntu",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "built-in", Password: "password"},
		},
		"default keyring secrets unused": {
			"ubuntu",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"extraneous": {Username: "built-in", Password: "password", Provider: nil},
			}),
			nil,
		},
		"builtin keyring secrets, but use passed": {
			"ubuntu",
			[]v1.Secret{{Type: v1.SecretTypeDockercfg, Data: map[string][]byte{v1.DockerConfigKey: dockercfgContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "passed-user", Password: "passed-password"},
		},
		"builtin keyring secrets, but use passed with new docker config": {
			"ubuntu",
			[]v1.Secret{{Type: v1.SecretTypeDockerConfigJson, Data: map[string][]byte{v1.DockerConfigJsonKey: dockerConfigJSONContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "passed-user", Password: "passed-password"},
		},
	}
	for description, test := range tests {
		builtInKeyRing := &credentialprovider.BasicDockerKeyring{}
		builtInKeyRing.Add(test.builtInDockerConfig)
		_, fakeImageService, fakeManager, err := customTestRuntimeManager(builtInKeyRing)
		require.NoError(t, err)

		_, err = fakeManager.PullImage(kubecontainer.ImageSpec{Image: test.imageName}, test.passedSecrets, nil)
		require.NoError(t, err)
		fakeImageService.AssertImagePulledWithAuth(t, &runtimeapi.ImageSpec{Image: test.imageName}, test.expectedAuth, description)
	}
}

func newNamedTestPod(ns string, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Namespace: ns,
			Name:      name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}
}
