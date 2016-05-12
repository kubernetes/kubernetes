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

package images

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	runtime "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

func getApiPod() *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:            "test_pod",
			Namespace:       "test-ns",
			UID:             "bar",
			ResourceVersion: "42",
			SelfLink:        "/api/v1/pods/foo",
		}}
}

func getRuntimeImage(image string) runtime.Image {
	return runtime.Image{
		RepoTags: []string{image},
	}
}

func TestPullMissingImage(t *testing.T) {
	as := assert.New(t)
	mockRuntimeImages := new(ctest.MockRuntimeImages)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := util.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRecorder := &record.FakeRecorder{}
	containerImage := "missing_image"
	mockRuntimeImages.On("ListImages").Return([]runtime.Image{}, nil)
	mockRuntimeImages.On("PullImage", runtime.ImageSpec{Image: containerImage}, []api.Secret(nil)).Return(nil)
	manager, err := NewImageManager(fakeRecorder, mockRuntimeImages, []runtime.Pod{}, backOff, false /*parallel*/)
	as.Nil(err, "image manager creation failed")

	container := &api.Container{
		Name:            "container_name",
		Image:           containerImage,
		ImagePullPolicy: api.PullIfNotPresent,
	}
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Nil(err, "Unexpected error. Pulling images failed")
}

func TestDontPullExistingImage(t *testing.T) {
	as := assert.New(t)
	mockRuntimeImages := new(ctest.MockRuntimeImages)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := util.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRecorder := &record.FakeRecorder{}
	containerImage := "existing_image"
	container := &api.Container{
		Name:            "container_name",
		Image:           containerImage,
		ImagePullPolicy: api.PullIfNotPresent,
	}

	mockRuntimeImages.On("ListImages").Once().Return([]runtime.Image{getRuntimeImage(containerImage)}, nil)
	mockRuntimeImages.AssertNotCalled(t, "PullImage")

	manager, err := NewImageManager(fakeRecorder, mockRuntimeImages, []runtime.Pod{}, backOff, false /*parallel*/)
	as.Nil(err, "image manager creation failed")

	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Nil(err, "Unexpected error. Pulling images failed")
}

func TestPullExistingImage(t *testing.T) {
	as := assert.New(t)
	mockRuntimeImages := new(ctest.MockRuntimeImages)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := util.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRecorder := &record.FakeRecorder{}
	containerImage := "existing_image"
	container := &api.Container{
		Name:            "container_name",
		Image:           containerImage,
		ImagePullPolicy: api.PullAlways,
	}

	mockRuntimeImages.On("ListImages").Once().Return([]runtime.Image{getRuntimeImage(containerImage)}, nil)
	mockRuntimeImages.On("PullImage", runtime.ImageSpec{Image: containerImage}, []api.Secret(nil)).Return(nil)
	manager, err := NewImageManager(fakeRecorder, mockRuntimeImages, []runtime.Pod{}, backOff, false /*parallel*/)
	as.Nil(err, "image manager creation failed")

	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Nil(err, "Unexpected error. Pulling images failed")
}

func TestDontPullImagePullNeverPolicy(t *testing.T) {
	as := assert.New(t)
	mockRuntimeImages := new(ctest.MockRuntimeImages)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := util.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRecorder := &record.FakeRecorder{}
	containerImage := "missing_image"
	container := &api.Container{
		Name:            "container_name",
		Image:           containerImage,
		ImagePullPolicy: api.PullNever,
	}

	mockRuntimeImages.On("ListImages").Once().Return([]runtime.Image{}, nil)
	mockRuntimeImages.AssertNotCalled(t, "PullImage")
	manager, err := NewImageManager(fakeRecorder, mockRuntimeImages, []runtime.Pod{}, backOff, false /*parallel*/)
	as.Nil(err, "image manager creation failed")

	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImageNeverPull, "expected error")
}

func TestPullFailure(t *testing.T) {
	as := assert.New(t)
	mockRuntimeImages := new(ctest.MockRuntimeImages)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := util.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRecorder := &record.FakeRecorder{}
	containerImage := "typo_image"
	container := &api.Container{
		Name:            "container_name",
		Image:           containerImage,
		ImagePullPolicy: api.PullIfNotPresent,
	}

	mockRuntimeImages.On("ListImages").Once().Return([]runtime.Image{}, nil)
	mockRuntimeImages.On("PullImage", runtime.ImageSpec{Image: containerImage}, []api.Secret(nil)).Return(errors.New("404"))
	manager, err := NewImageManager(fakeRecorder, mockRuntimeImages, []runtime.Pod{}, backOff, false /*parallel*/)
	as.Nil(err, "image manager creation failed")

	// Attempt 1 - Fails to pull image
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImagePull, "expected error")
	fakeClock.Step(time.Second)
	// Attempt 2 - Pull backoff
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImagePull, "expected error")
	fakeClock.Step(time.Second)
	// Attempt 3 - Pull backoff
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImagePullBackOff, "expected error")
	fakeClock.Step(time.Second)
	// Attempt 4 - Fails to pull image
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImagePull, "expected error")
	fakeClock.Step(time.Second)
	// Attempt 5 - Pull backoff
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImagePullBackOff, "expected error")
	fakeClock.Step(time.Second)
	// Attempt 6 - Pull backoff
	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Equal(err, ErrImagePullBackOff, "expected error")
}

func TestPullGCedExistingImage(t *testing.T) {
	as := assert.New(t)
	mockRuntimeImages := new(ctest.MockRuntimeImages)
	backOff := flowcontrol.NewBackOff(time.Second, time.Minute)
	fakeClock := util.NewFakeClock(time.Now())
	backOff.Clock = fakeClock
	fakeRecorder := &record.FakeRecorder{}
	containerImage := "existing_image"
	container := &api.Container{
		Name:            "container_name",
		Image:           containerImage,
		ImagePullPolicy: api.PullIfNotPresent,
	}

	mockRuntimeImages.On("ListImages").Once().Return([]runtime.Image{getRuntimeImage(containerImage)}, nil)
	mockRuntimeImages.On("PullImage", runtime.ImageSpec{Image: containerImage}, []api.Secret(nil)).Once().Return(nil)

	manager, err := NewImageManager(fakeRecorder, mockRuntimeImages, []runtime.Pod{}, backOff, false /*parallel*/)
	as.Nil(err, "image manager creation failed")

	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Nil(err, "Unexpected error. Pulling images failed")

	manager.DeleteUnusedImages()

	err, _ = manager.EnsureImageExists(getApiPod(), container, nil)
	as.Nil(err, "Unexpected error. Pulling images failed")
}
