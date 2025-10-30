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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	imagepullmanager "k8s.io/kubernetes/pkg/kubelet/images/pullmanager"
	"k8s.io/kubernetes/testutils/ktesting"
	"k8s.io/utils/ptr"
)

func TestPullImage(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	imageRef, creds, err := fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Equal(t, "busybox", imageRef)
	assert.Nil(t, creds) // as this was an anonymous pull

	images, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	assert.Len(t, images, 1)
	assert.Equal(t, []string{"busybox"}, images[0].RepoTags)
}

func TestPullImageWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	fakeImageService.InjectError("PullImage", fmt.Errorf("test-error"))
	imageRef, creds, err := fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.Error(t, err)
	assert.Equal(t, "", imageRef)
	assert.Nil(t, creds)

	images, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	assert.Empty(t, images)
}

func TestListImages(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	images := []string{"1111", "2222", "3333"}
	expected := sets.New[string](images...)
	fakeImageService.SetFakeImages(images)

	actualImages, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	actual := sets.New[string]()
	for _, i := range actualImages {
		actual.Insert(i.ID)
	}

	assert.Equal(t, sets.List(expected), sets.List(actual))
}

func TestListImagesPinnedField(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	imagesPinned := map[string]bool{
		"1111": false,
		"2222": true,
		"3333": false,
	}
	imageList := []string{}
	for image, pinned := range imagesPinned {
		fakeImageService.SetFakeImagePinned(image, pinned)
		imageList = append(imageList, image)
	}
	fakeImageService.SetFakeImages(imageList)

	actualImages, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	for _, image := range actualImages {
		assert.Equal(t, imagesPinned[image.ID], image.Pinned)
	}
}

func TestListImagesWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	fakeImageService.InjectError("ListImages", fmt.Errorf("test-failure"))

	actualImages, err := fakeManager.ListImages(tCtx)
	assert.Error(t, err)
	assert.Nil(t, actualImages)
}

func TestGetImageRef(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	image := "busybox"
	fakeImageService.SetFakeImages([]string{image})
	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)
	assert.Equal(t, image, imageRef)
}

func TestImageSize(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	const imageSize = uint64(64)
	fakeImageService.SetFakeImageSize(imageSize)
	image := "busybox"
	fakeImageService.SetFakeImages([]string{image})
	actualSize, err := fakeManager.GetImageSize(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)
	assert.Equal(t, imageSize, actualSize)
}

func TestGetImageRefImageNotAvailableLocally(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	image := "busybox"

	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.NoError(t, err)

	imageNotAvailableLocallyRef := ""
	assert.Equal(t, imageNotAvailableLocallyRef, imageRef)
}

func TestGetImageRefWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	image := "busybox"

	fakeImageService.InjectError("ImageStatus", fmt.Errorf("test-error"))

	imageRef, err := fakeManager.GetImageRef(tCtx, kubecontainer.ImageSpec{Image: image})
	assert.Error(t, err)
	assert.Equal(t, "", imageRef)
}

func TestRemoveImage(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	_, _, err = fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Len(t, fakeImageService.Images, 1)

	err = fakeManager.RemoveImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"})
	assert.NoError(t, err)
	assert.Empty(t, fakeImageService.Images)
}

func TestRemoveImageNoOpIfImageNotLocal(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	err = fakeManager.RemoveImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"})
	assert.NoError(t, err)
}

func TestRemoveImageWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	_, _, err = fakeManager.PullImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"}, nil, nil)
	assert.NoError(t, err)
	assert.Len(t, fakeImageService.Images, 1)

	fakeImageService.InjectError("RemoveImage", fmt.Errorf("test-failure"))

	err = fakeManager.RemoveImage(tCtx, kubecontainer.ImageSpec{Image: "busybox"})
	assert.Error(t, err)
	assert.Len(t, fakeImageService.Images, 1)
}

func TestImageStats(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	const imageSize = 64
	fakeImageService.SetFakeImageSize(imageSize)
	images := []string{"1111", "2222", "3333"}
	fakeImageService.SetFakeImages(images)

	actualStats, err := fakeManager.ImageStats(tCtx)
	assert.NoError(t, err)
	expectedStats := &kubecontainer.ImageStats{TotalStorageBytes: imageSize * uint64(len(images))}
	assert.Equal(t, expectedStats, actualStats)
}

func TestImageStatsWithError(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	fakeImageService.InjectError("ListImages", fmt.Errorf("test-failure"))

	actualImageStats, err := fakeManager.ImageStats(tCtx)
	assert.Error(t, err)
	assert.Nil(t, actualImageStats)
}

func TestPullWithSecrets(t *testing.T) {
	tCtx := ktesting.Init(t)
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

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      "testpod",
			Namespace: "testpod-ns",
			Uid:       "testpod-uid",
		},
	}

	tests := map[string]struct {
		imageName           string
		passedSecrets       []v1.Secret
		builtInDockerConfig credentialprovider.DockerConfig
		expectedAuth        *runtimeapi.AuthConfig
	}{
		"no matching secrets": {
			"ubuntu:latest",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{}),
			nil,
		},
		"default keyring secrets": {
			"ubuntu:latest",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "built-in", Password: "password"},
		},
		"default keyring secrets unused": {
			"ubuntu:latest",
			[]v1.Secret{},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"extraneous": {Username: "built-in", Password: "password", Provider: nil},
			}),
			nil,
		},
		"builtin keyring secrets, but use passed": {
			"ubuntu:latest",
			[]v1.Secret{{Type: v1.SecretTypeDockercfg, Data: map[string][]byte{v1.DockerConfigKey: dockercfgContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "passed-user", Password: "passed-password"},
		},
		"builtin keyring secrets, but use passed with new docker config": {
			"ubuntu:latest",
			[]v1.Secret{{Type: v1.SecretTypeDockerConfigJson, Data: map[string][]byte{v1.DockerConfigJsonKey: dockerConfigJSONContent}}},
			credentialprovider.DockerConfig(map[string]credentialprovider.DockerConfigEntry{
				"index.docker.io/v1/": {Username: "built-in", Password: "password", Provider: nil},
			}),
			&runtimeapi.AuthConfig{Username: "passed-user", Password: "passed-password"},
		},
	}
	for description, test := range tests {
		builtInKeyRing := &credentialprovider.BasicDockerKeyring{}
		builtInKeyRing.Add(nil, test.builtInDockerConfig)

		_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
		require.NoError(t, err)

		fsRecordAccessor, err := imagepullmanager.NewFSPullRecordsAccessor(t.TempDir())
		if err != nil {
			t.Fatal("failed to setup an file pull records accessor")
		}

		const intentsRecordsCacheSize, pulledRecordsCacheSize, stripedLocksSetSize = 50, 100, 10
		memCacheRecordsAccessor := imagepullmanager.NewCachedPullRecordsAccessor(
			fsRecordAccessor,
			intentsRecordsCacheSize,
			pulledRecordsCacheSize,
			stripedLocksSetSize,
		)

		imagePullManager, err := imagepullmanager.NewImagePullManager(tCtx, memCacheRecordsAccessor, imagepullmanager.AlwaysVerifyImagePullPolicy(), fakeManager, 10)
		if err != nil {
			t.Fatal("failed to setup an image pull manager")
		}

		fakeManager.imagePuller = images.NewImageManager(
			fakeManager.recorder,
			builtInKeyRing,
			fakeManager,
			imagePullManager,
			flowcontrol.NewBackOff(time.Second, 300*time.Second),
			false,
			ptr.To[int32](0), // No limit on max parallel image pulls,
			0,                // Disable image pull throttling by setting QPS to 0,
			0,
			&fakePodPullingTimeRecorder{},
		)

		_, _, err = fakeManager.imagePuller.EnsureImageExists(tCtx, nil, makeTestPod("testpod", "testpod-ns", "testpod-uid", []v1.Container{}), test.imageName, test.passedSecrets, podSandboxConfig, "", v1.PullAlways)
		require.NoError(t, err)
		fakeImageService.AssertImagePulledWithAuth(t, &runtimeapi.ImageSpec{Image: test.imageName, Annotations: make(map[string]string)}, test.expectedAuth, description)
	}
}

func TestPullWithSecretsWithError(t *testing.T) {
	tCtx := ktesting.Init(t)

	dockerCfg := map[string]map[string]map[string]string{
		"auths": {
			"index.docker.io/v1/": {
				"email": "passed-email",
				"auth":  "cGFzc2VkLXVzZXI6cGFzc2VkLXBhc3N3b3Jk",
			},
		},
	}

	dockerConfigJSON, err := json.Marshal(dockerCfg)
	if err != nil {
		t.Fatal(err)
	}

	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      "testpod",
			Namespace: "testpod-ns",
			Uid:       "testpod-uid",
		},
	}

	for _, test := range []struct {
		name              string
		imageName         string
		passedSecrets     []v1.Secret
		shouldInjectError bool
	}{
		{
			name:          "invalid docker secret",
			imageName:     "ubuntu:latest",
			passedSecrets: []v1.Secret{{Type: v1.SecretTypeDockercfg, Data: map[string][]byte{v1.DockerConfigKey: []byte("invalid")}}},
		},
		{
			name:      "secret provided, pull failed",
			imageName: "ubuntu:latest",
			passedSecrets: []v1.Secret{
				{Type: v1.SecretTypeDockerConfigJson, Data: map[string][]byte{v1.DockerConfigKey: dockerConfigJSON}},
			},
			shouldInjectError: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, fakeImageService, fakeManager, err := createTestRuntimeManager(tCtx)
			assert.NoError(t, err)

			if test.shouldInjectError {
				fakeImageService.InjectError("PullImage", fmt.Errorf("test-error"))
			}

			fsRecordAccessor, err := imagepullmanager.NewFSPullRecordsAccessor(t.TempDir())
			if err != nil {
				t.Fatal("failed to setup an file pull records accessor")
			}

			const intentsRecordsCacheSize, pulledRecordsCacheSize, stripedLocksSetSize = 50, 100, 10
			memCacheRecordsAccessor := imagepullmanager.NewCachedPullRecordsAccessor(
				fsRecordAccessor,
				intentsRecordsCacheSize,
				pulledRecordsCacheSize,
				stripedLocksSetSize,
			)

			imagePullManager, err := imagepullmanager.NewImagePullManager(tCtx, memCacheRecordsAccessor, imagepullmanager.AlwaysVerifyImagePullPolicy(), fakeManager, 10)
			if err != nil {
				t.Fatal("failed to setup an image pull manager")
			}

			fakeManager.imagePuller = images.NewImageManager(
				fakeManager.recorder,
				&credentialprovider.BasicDockerKeyring{},
				fakeManager,
				imagePullManager,
				flowcontrol.NewBackOff(time.Second, 300*time.Second),
				false,
				ptr.To[int32](0), // No limit on max parallel image pulls,
				0,                // Disable image pull throttling by setting QPS to 0,
				0,
				&fakePodPullingTimeRecorder{},
			)

			imageRef, _, err := fakeManager.imagePuller.EnsureImageExists(tCtx, nil, makeTestPod("testpod", "testpod-ns", "testpod-uid", []v1.Container{}), test.imageName, test.passedSecrets, podSandboxConfig, "", v1.PullAlways)
			assert.Error(t, err)
			assert.Equal(t, "", imageRef)

			images, err := fakeManager.ListImages(tCtx)
			assert.NoError(t, err)
			assert.Empty(t, images)
		})
	}
}

func TestPullThenListWithAnnotations(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeManager, err := createTestRuntimeManager(tCtx)
	assert.NoError(t, err)

	imageSpec := kubecontainer.ImageSpec{
		Image: "12345",
		Annotations: []kubecontainer.Annotation{
			{Name: "kubernetes.io/runtimehandler", Value: "handler_name"},
		},
	}

	_, _, err = fakeManager.PullImage(tCtx, imageSpec, nil, nil)
	assert.NoError(t, err)

	images, err := fakeManager.ListImages(tCtx)
	assert.NoError(t, err)
	assert.Len(t, images, 1)
	assert.Equal(t, images[0].Spec, imageSpec)
}
