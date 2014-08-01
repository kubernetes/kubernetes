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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/image/imageapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

// An implementation of ImageRegistry and ImageRepositoryRegistry that is backed by memory
// Mainly used for testing.
type MemoryRegistry struct {
	imageData           map[string]imageapi.Image
	imageRepositoryData map[string]imageapi.ImageRepository
	repositoryImages    map[string][]string
}

func MakeMemoryRegistry() *MemoryRegistry {
	return &MemoryRegistry{
		imageData:           map[string]imageapi.Image{},
		imageRepositoryData: map[string]imageapi.ImageRepository{},
		repositoryImages:    map[string][]string{},
	}
}

func (r *MemoryRegistry) ListImages(selector labels.Selector) ([]imageapi.Image, error) {
	result := []imageapi.Image{}
	for _, value := range r.imageData {
		if selector.Matches(labels.Set(value.Labels)) {
			result = append(result, value)
		}
	}
	return result, nil
}

func (r *MemoryRegistry) GetImage(imageID string) (*imageapi.Image, error) {
	image, found := r.imageData[imageID]
	if found {
		return &image, nil
	} else {
		return nil, apiserver.NewNotFoundErr("image", imageID)
	}
}

func (r *MemoryRegistry) CreateImage(image imageapi.Image) error {
	if _, found := r.imageData[image.ID]; found {
		return apiserver.NewAlreadyExistsErr("image", image.ID)
	}
	r.imageData[image.ID] = image
	return nil
}

func (r *MemoryRegistry) DeleteImage(imageID string) error {
	if _, ok := r.imageData[imageID]; !ok {
		return apiserver.NewNotFoundErr("image", imageID)
	}
	delete(r.imageData, imageID)
	return nil
}

func (r *MemoryRegistry) ListImageRepositories(selector labels.Selector) ([]imageapi.ImageRepository, error) {
	result := []imageapi.ImageRepository{}
	for _, value := range r.imageRepositoryData {
		if selector.Matches(labels.Set(value.Labels)) {
			result = append(result, value)
		}
	}
	return result, nil
}

func (r *MemoryRegistry) ListImagesFromRepository(repositoryID string, selector labels.Selector) ([]string, error) {
	_, err := r.GetImageRepository(repositoryID)
	if err != nil {
		return []string{}, err
	}
	imageIDs, found := r.repositoryImages[repositoryID]
	if !found {
		return []string{}, err
	}
	return imageIDs, nil
}

func (r *MemoryRegistry) GetImageRepository(repositoryID string) (*imageapi.ImageRepository, error) {
	repository, found := r.imageRepositoryData[repositoryID]
	if !found {
		return nil, apiserver.NewNotFoundErr("imageRepository", repositoryID)
	}
	return &repository, nil
}

func (r *MemoryRegistry) CreateImageRepository(repository imageapi.ImageRepository) error {
	if _, found := r.imageRepositoryData[repository.ID]; found {
		return apiserver.NewAlreadyExistsErr("imageRepository", repository.ID)
	}
	delete(r.repositoryImages, repository.ID)
	r.imageRepositoryData[repository.ID] = repository
	return nil
}

func (r *MemoryRegistry) UpdateImageRepository(repository imageapi.ImageRepository) error {
	if _, ok := r.imageRepositoryData[repository.ID]; !ok {
		return apiserver.NewNotFoundErr("imageRepository", repository.ID)
	}
	r.imageRepositoryData[repository.ID] = repository
	return nil
}

func (r *MemoryRegistry) DeleteImageRepository(repositoryID string) error {
	if _, ok := r.imageRepositoryData[repositoryID]; !ok {
		return apiserver.NewNotFoundErr("imageRepository", repositoryID)
	}
	delete(r.repositoryImages, repositoryID)
	delete(r.imageRepositoryData, repositoryID)
	return nil
}

func (r *MemoryRegistry) AddImageToRepository(repositoryID string, imageID string) error {
	glog.Infof("adding %s/%s to %#v", repositoryID, imageID, r.repositoryImages)
	list, found := r.repositoryImages[repositoryID]
	if !found {
		list = []string{}
	}
	list = append(list, imageID)
	r.repositoryImages[repositoryID] = list
	return nil
}

func (r *MemoryRegistry) RemoveImageFromRepository(repositoryID string, imageID string) error {
	list, found := r.repositoryImages[repositoryID]
	if !found {
		list = []string{}
	}
	filtered := []string{}
	for _, id := range list {
		if id != imageID {
			filtered = append(filtered, id)
		}
	}
	r.repositoryImages[repositoryID] = list
	return nil
}
