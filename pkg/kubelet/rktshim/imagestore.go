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

package rktshim

import (
	"errors"

	"k8s.io/kubernetes/pkg/kubelet/container"
)

// TODO(tmrts): Move these errors to the container API for code re-use.
var (
	ErrImageNotFound = errors.New("rktshim: image not found")
)

var _ container.ImageService = (*ImageStore)(nil)

// ImageStore supports CRUD operations for images.
type ImageStore struct{}

// TODO(tmrts): fill the image store configuration fields.
type ImageStoreConfig struct{}

// NewImageStore creates an image storage that allows CRUD operations for images.
func NewImageStore(ImageStoreConfig) (container.ImageService, error) {
	return &ImageStore{}, nil
}

// List lists the images residing in the image store.
func (*ImageStore) List() ([]container.Image, error) {
	panic("not implemented")
}

// Pull pulls an image into the image store and uses the given authentication method.
func (*ImageStore) Pull(container.ImageSpec, container.AuthConfig, *container.PodSandboxConfig) error {
	panic("not implemented")
}

// Remove removes the image from the image store.
func (*ImageStore) Remove(container.ImageSpec) error {
	panic("not implemented")
}

// Status returns the status of the image.
func (*ImageStore) Status(container.ImageSpec) (container.Image, error) {
	panic("not implemented")
}
