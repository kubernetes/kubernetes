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

package images

import "k8s.io/kubernetes/pkg/api"

// ImageManager provides an interface to manage the lifecycle of images.
// Implementations of this interface are expected to deal with pulling (downloading),
// managing, and deleting container images.
// Implementations are expected to abstract the underlying runtimes.
// Implementations are expected to be thread safe.
type ImageManager interface {
	// EnsureImageExists will ensure that image specified in `container` exists.
	// TODO: Consider failing when conflicting PullPolicy is provided for the same image.
	EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) error
	// DecImageUsage decreases the usage count of the image specified in `imageSpec`.
	// This is useful for reference counting images.
	// Returns an error if the image is not found.
	DecImageUsage(container *api.Container) error
	// DeleteImages attempts to free up disk space by deleting images that are currently unused.
	// It will return the number of bytes freed on success, an appropriate error otherwise.
	DeleteUnusedImages()
}

// ImageSpec is an internal representation of an image.
type ImageSpec struct {
	Image       string
	pullSecrets []api.Secret
}
