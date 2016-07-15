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

package images

import (
	"errors"
	"time"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

var (
	// Container image pull failed, kubelet is backing off image pull
	ErrImagePullBackOff = errors.New("ImagePullBackOff")

	// Unable to inspect image
	ErrImageInspect = errors.New("ImageInspectError")

	// General image pull error
	ErrImagePull = errors.New("ErrImagePull")

	// Required Image is absent on host and PullPolicy is NeverPullImage
	ErrImageNeverPull = errors.New("ErrImageNeverPull")

	// Get http error when pulling image from registry
	RegistryUnavailable = errors.New("RegistryUnavailable")
)

// ImageManager provides an interface to manage the lifecycle of images.
// Implementations of this interface are expected to deal with pulling (downloading),
// managing, and deleting container images.
// Implementations are expected to abstract the underlying runtimes.
// Implementations are expected to be thread safe.
type ImageManager interface {
	// EnsureImageExists ensures that image specified in `container` exists.
	EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string)

	// Applies the garbage collection policy. Errors include being unable to free
	// enough space as per the garbage collection policy.
	GarbageCollect() error

	// Start async garbage collection of images.
	Start() error

	// GetImageList returns the images that's available
	GetImageList() ([]kubecontainer.Image, error)

	// DeleteUnusedImages deletes all unused images and returns the number of bytes freed. The number of bytes freed is always returned.
	DeleteUnusedImages() (int64, error)
}

// A policy for garbage collecting images. Policy defines an allowed band in
// which garbage collection will be run.
type ImageGCPolicy struct {
	// Any usage above this threshold will always trigger garbage collection.
	// This is the highest usage we will allow.
	HighThresholdPercent int

	// Any usage below this threshold will never trigger garbage collection.
	// This is the lowest threshold we will try to garbage collect to.
	LowThresholdPercent int

	// Minimum age at which a image can be garbage collected.
	MinAge time.Duration
}
