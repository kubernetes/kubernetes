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

	"k8s.io/api/core/v1"
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

	// Unable to parse the image name.
	ErrInvalidImageName = errors.New("InvalidImageName")
)

// ImageManager provides an interface to manage the lifecycle of images.
// Implementations of this interface are expected to deal with pulling (downloading),
// managing, and deleting container images.
// Implementations are expected to abstract the underlying runtimes.
// Implementations are expected to be thread safe.
type ImageManager interface {
	// EnsureImageExists ensures that image specified in `container` exists.
	EnsureImageExists(pod *v1.Pod, container *v1.Container, pullSecrets []v1.Secret) (string, string, error)

	// TODO(ronl): consolidating image managing and deleting operation in this interface
}
