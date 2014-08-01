/*
Copyright 2014 Google Inc. All rights reserved.

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

package image

import (
	"fmt"

	"code.google.com/p/go-uuid/uuid"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/image/imageapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// ImageRegistryStorage is an implementation of RESTStorage for the api server.
type ImageRegistryStorage struct {
	registry ImageRegistry
}

func NewImageRegistryStorage(registry ImageRegistry) apiserver.RESTStorage {
	return &ImageRegistryStorage{
		registry: registry,
	}
}

// List obtains a list of Images that match selector.
func (s *ImageRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := imageapi.ImageList{}
	images, err := s.registry.ListImages(selector)
	if err == nil {
		result.Items = images
	}
	return result, err
}

// Get obtains the Image specified by its id.
func (s *ImageRegistryStorage) Get(id string) (interface{}, error) {
	image, err := s.registry.GetImage(id)
	if err != nil {
		return nil, err
	}
	return image, err
}

// Delete asynchronously deletes the Image specified by its id.
func (s *ImageRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, s.registry.DeleteImage(id)
	}), nil
}

// Extract deserializes user provided data into an imageapi.Image.
func (s *ImageRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := imageapi.Image{}
	err := api.DecodeInto(body, &result)
	return result, err
}

// Create registers a given new Image instance to s.registry.
func (s *ImageRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	image, ok := obj.(imageapi.Image)
	if !ok {
		return nil, fmt.Errorf("not an image: %#v", obj)
	}
	if len(image.ID) == 0 {
		image.ID = uuid.NewUUID().String()
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := s.registry.CreateImage(image)
		if err != nil {
			return nil, err
		}
		return s.registry.GetImage(image.ID)
	}), nil
}

// Update replaces a given Image instance with an existing instance in s.registry.
func (s *ImageRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	return s.Create(obj)
}
