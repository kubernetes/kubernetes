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
	"errors"
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/image/imageapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// ImagesByRepositoryRegistryStorage is an implementation of RESTStorage for the api server.
type ImagesByRepositoryRegistryStorage struct {
	registry      ImageRepositoryRegistry
	imageRegistry ImageRegistry
}

func NewImagesByRepositoryRegistryStorage(registry ImageRepositoryRegistry, imageRegistry ImageRegistry) apiserver.RESTStorage {
	return &ImagesByRepositoryRegistryStorage{
		registry:      registry,
		imageRegistry: imageRegistry,
	}
}

func mappingFromID(id string) (*imageapi.ImageRepositoryMapping, error) {
	segments := strings.SplitN(id, "!", 2)
	if len(segments) < 2 || segments[0] == "" || segments[1] == "" {
		return nil, errors.New("mapping ID is invalid")
	}
	return &imageapi.ImageRepositoryMapping{
		JSONBase:       api.JSONBase{ID: id},
		RepositoryName: segments[0],
		Image:          imageapi.Image{JSONBase: api.JSONBase{ID: segments[1]}},
	}, nil
}

func mappingToID(mapping *imageapi.ImageRepositoryMapping) string {
	if mapping.RepositoryName != "" && mapping.Image.ID != "" {
		return ""
	}
	return fmt.Sprintf("%s!%s", mapping.RepositoryName, mapping.Image.ID)
}

// List obtains a list of ImageRepositorys that match selector.
func (s *ImagesByRepositoryRegistryStorage) List(selector labels.Selector) (interface{}, error) {
	return nil, errors.New("not supported")
}

// Get returns the Images in the ImageRepository specified by its id.
func (s *ImagesByRepositoryRegistryStorage) Get(id string) (interface{}, error) {
	result := imageapi.ImageList{}
	imageIDs, err := s.registry.ListImagesFromRepository(id, labels.Everything())
	if err != nil {
		return result, err
	}
	for _, imageID := range imageIDs {
		if image, err := s.imageRegistry.GetImage(imageID); err == nil {
			result.Items = append(result.Items, *image)
		}
	}
	return result, nil
}

// Delete asynchronously deletes the ImageRepository specified by its id.
func (s *ImagesByRepositoryRegistryStorage) Delete(id string) (<-chan interface{}, error) {
	mapping, err := mappingFromID(id)
	if err != nil {
		return nil, err
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		return &api.Status{Status: api.StatusSuccess}, s.registry.RemoveImageFromRepository(mapping.RepositoryName, mapping.Image.ID)
	}), nil
}

// Extract deserializes user provided data into an imageapi.ImageRepository.
func (s *ImagesByRepositoryRegistryStorage) Extract(body []byte) (interface{}, error) {
	result := imageapi.ImageRepositoryMapping{}
	err := api.DecodeInto(body, &result)
	return result, err
}

// Create binds a new or existing image to an image repository
func (s *ImagesByRepositoryRegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	mapping, ok := obj.(imageapi.ImageRepositoryMapping)
	if !ok {
		return nil, fmt.Errorf("not an image repository mapping: %#v", obj)
	}

	if mapping.Image.ID == "" {
		return nil, fmt.Errorf("no image ID defined: %#v", mapping)
	}

	return apiserver.MakeAsync(func() (interface{}, error) {
		_, err := s.registry.GetImageRepository(mapping.RepositoryName)
		if err != nil {
			return nil, err
		}

		if err := s.imageRegistry.CreateImage(mapping.Image); err != nil && !apiserver.IsAlreadyExists(err) {
			return nil, err
		}

		if err := s.registry.AddImageToRepository(mapping.RepositoryName, mapping.Image.ID); err != nil {
			return nil, err
		}
		return mapping, nil
	}), nil
}

// Update replaces a given ImageRepository instance with an existing instance in the registry.
func (s *ImagesByRepositoryRegistryStorage) Update(obj interface{}) (<-chan interface{}, error) {
	return s.Create(obj)
}
