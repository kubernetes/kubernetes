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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/image/imageapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// ImageRegistry is an interface implemented by things that know how to store Image objects.
type ImageRegistry interface {
	ListImages(selector labels.Selector) ([]imageapi.Image, error)
	GetImage(imageID string) (*imageapi.Image, error)
	CreateImage(image imageapi.Image) error
	DeleteImage(imageID string) error
}

// ImageRepositoryRegistry is an interface for things that know how to store ImageRepository objects.
type ImageRepositoryRegistry interface {
	ListImageRepositories(selector labels.Selector) ([]imageapi.ImageRepository, error)
	ListImagesFromRepository(repositoryID string, selector labels.Selector) ([]string, error)
	GetImageRepository(repositoryID string) (*imageapi.ImageRepository, error)
	CreateImageRepository(repository imageapi.ImageRepository) error
	UpdateImageRepository(repository imageapi.ImageRepository) error
	DeleteImageRepository(repositoryID string) error
	AddImageToRepository(repositoryID string, imageID string) error
	RemoveImageFromRepository(repositoryID string, imageID string) error
}
