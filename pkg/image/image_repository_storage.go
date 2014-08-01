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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/image/imageapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// ImageRepositoryRegistryStorage is an implementation of RESTStorage for the api server.
type ImageRepositoryRegistryStorage struct {
	registry      ImageRepositoryRegistry
	imageRegistry ImageRegistry
}

func NewImageRepositoryRegistryStorage(registry ImageRepositoryRegistry, imageRegistry ImageRegistry) apiserver.RESTStorage {
	return &ImageRepositoryRegistryStorage{
		registry:      registry,
		imageRegistry: imageRegistry,
	}
}

// List obtains a list of ImageRepositorys that match selector.
func (s *ImageRepositoryRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	result := imageapi.ImageRepositoryList{}
	images, err := s.registry.ListImageRepositories(selector)
	if err == nil {
		result.Items = images
	}
	return result, err
}

// Get obtains the ImageRepository specified by its id.
func (s *ImageRepositoryRegistryStorage) Get(id string) (interface{}, error) {
	image, err := s.registry.GetImageRepository(id)
	if err != nil {
		return nil, err
	}
	return image, err
}

// Delete asynchronously deletes the ImageRepository specified by its id.
func (s *ImageRepositoryRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	return apiserver.MakeAsync(func() (interface{}, error) {
		return api.Status{Status: api.StatusSuccess}, s.registry.DeleteImageRepository(id)
	}), nil
}

// Extract deserializes user provided data into an imageapi.ImageRepository.
func (s *ImageRepositoryRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := imageapi.ImageRepository{}
	err := api.DecodeInto(body, &result)
	return result, err
}

// Create registers a given new ImageRepository instance to the registry.
func (s *ImageRepositoryRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	repository, ok := obj.(imageapi.ImageRepository)
	if !ok {
		return nil, fmt.Errorf("not an image repository: %#v", obj)
	}
	repository.ID = repository.Name
	if repository.ID == "" {
		return nil, fmt.Errorf("image repository must have a name: %#v", obj)
	}
	for tag, imageID := range repository.Tags {
		if _, err := s.imageRegistry.GetImage(imageID); err != nil {
			return nil, fmt.Errorf("unable to set tag '%s' to image '%s': %v", tag, imageID, err)
		}
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		if err := s.registry.CreateImageRepository(repository); err != nil {
			return nil, err
		}
		if err := s.associateTags(repository); err != nil {
			return nil, err
		}
		return s.registry.GetImageRepository(repository.ID)
	}), nil
}

// Update replaces a given ImageRepository instance with an existing instance in the registry.
func (s *ImageRepositoryRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	repository, ok := obj.(imageapi.ImageRepository)
	if !ok {
		return nil, fmt.Errorf("not an image repository: %#v", obj)
	}
	if len(repository.ID) == 0 {
		return nil, fmt.Errorf("ID should not be empty: %#v", repository)
	}
	for tag, imageID := range repository.Tags {
		if _, err := s.imageRegistry.GetImage(imageID); err != nil {
			return nil, fmt.Errorf("unable to set tag '%s' to image '%s': %v", tag, imageID, err)
		}
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		if err := s.registry.UpdateImageRepository(repository); err != nil {
			return nil, err
		}
		if err := s.associateTags(repository); err != nil {
			return nil, err
		}
		return s.registry.GetImageRepository(repository.ID)
	}), nil
}

// associateTags adds each of the tagged images to the registry.
func (s *ImageRepositoryRegistryStorage) associateTags(repository imageapi.ImageRepository) error {
	errors := []error{}
	for tag, imageID := range repository.Tags {
		if err := s.registry.AddImageToRepository(repository.ID, imageID); err != nil {
			errors = append(errors, fmt.Errorf("unable to set tag '%s' to image '%s': %v", tag, imageID, err))
		}
	}
	if len(errors) > 0 {
		return fmt.Errorf("tags not applied: %v", errors)
	}
	return nil
}
