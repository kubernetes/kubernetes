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
	"fmt"
	"strconv"
	"strings"
	"time"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"

	"k8s.io/kubernetes/pkg/kubelet/container"
)

// TODO(tmrts): Move these errors to the container API for code re-use.
var (
	ErrImageNotFound = errors.New("rktshim: image not found")
)

// var _ kubeletApi.ImageManagerService = (*ImageStore)(nil)

// ImageStore supports CRUD operations for images.
type ImageStore struct {
	rkt            CLI
	requestTimeout time.Duration
}

// TODO(tmrts): fill the image store configuration fields.
type ImageStoreConfig struct {
	CLI            CLI
	RequestTimeout time.Duration
}

// NewImageStore creates an image storage that allows CRUD operations for images.
func NewImageStore(cfg ImageStoreConfig) (*ImageStore, error) {
	return &ImageStore{rkt: cfg.CLI, requestTimeout: cfg.RequestTimeout}, nil
}

// List lists the images residing in the image store.
func (s *ImageStore) List() ([]container.Image, error) {
	list, err := s.rkt.RunCommand("image", "list",
		"--no-legend",
		"--fields=id,name,size",
		"--sort=importtime",
	)
	if err != nil {
		return nil, fmt.Errorf("couldn't list images: %v", err)
	}

	images := make([]container.Image, len(list))
	for i, image := range list {
		tokens := strings.Fields(image)

		id, name := tokens[0], tokens[1]

		size, err := strconv.ParseInt(tokens[2], 10, 0)
		if err != nil {
			return nil, fmt.Errorf("invalid image size format: %v", err)
		}

		images[i] = container.Image{
			ID:       id,
			RepoTags: []string{name},
			Size:     size,
		}
	}

	return images, nil
}

// Pull pulls an image into the image store and uses the given authentication method.
func (s *ImageStore) Pull(container.ImageSpec, runtimeApi.AuthConfig, *runtimeApi.PodSandboxConfig) error {
	panic("not implemented yet!")
}

// Remove removes the image from the image store.
func (s *ImageStore) Remove(imgSpec container.ImageSpec) error {
	img, err := s.Status(imgSpec)
	if err != nil {
		return err
	}

	if _, err := s.rkt.RunCommand("image", "rm", img.ID); err != nil {
		return fmt.Errorf("failed to remove the image: %v", err)
	}

	return nil
}

// Status returns the status of the image.
func (s *ImageStore) Status(spec container.ImageSpec) (container.Image, error) {
	images, err := s.List()
	if err != nil {
		return container.Image{}, err
	}

	for _, img := range images {
		for _, tag := range img.RepoTags {
			if tag == spec.Image {
				return img, nil
			}
		}
	}

	return container.Image{}, fmt.Errorf("couldn't to find the image %v", spec.Image)
}
