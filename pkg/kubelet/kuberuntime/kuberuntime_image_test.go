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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestPullImage(t *testing.T) {
	_, _, fakeManager, err := createTestRuntimeManager()
	assert.NoError(t, err)

	imageRef, err := fakeManager.PullImage(kubecontainer.ImageSpec{Image: "busybox"}, nil)
	assert.NoError(t, err)
	assert.Equal(t, "busybox", imageRef)

	images, err := fakeManager.ListImages()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(images))
	assert.Equal(t, images[0].RepoTags, []string{"busybox"})
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

	_, err = fakeManager.PullImage(kubecontainer.ImageSpec{Image: "busybox"}, nil)
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
