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

package imageapi

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/fsouza/go-dockerclient"
)

type Image struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Labels       map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	Metadata     docker.Image      `json:"metadata,omitempty" yaml:"metadata,omitempty"`
	Reference    string            `json:"name,omitempty" yaml:"name,omitempty"`
}

type ImageList struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Items        []Image `json:"items,omitempty" yaml:"items,omitempty"`
}

type ImageRepository struct {
	api.JSONBase     `json:",inline" yaml:",inline"`
	Name             string            `json:"name,omitempty" yaml:"name,omitempty"`
	Labels           map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
	OverrideMetadata *docker.Image     `json:"overrideMetadata,omitempty" yaml:"overrideMetadata,omitempty"`
	Tags             map[string]string `json:"tags,omitempty" yaml:"tags,omitempty"`
}

type ImageRepositoryList struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Items        []ImageRepository `json:"items,omitempty" yaml:"items,omitempty"`
}

type ImageRepositoryMapping struct {
	api.JSONBase   `json:",inline" yaml:",inline"`
	Image          Image  `json:"image,omitempty" yaml:"image,omitempty"`
	RepositoryName string `json:"repositoryName,omitempty" yaml:"repositoryName,omitempty"`
}

func init() {
	api.AddKnownTypes("",
		Image{},
		ImageList{},
		ImageRepository{},
		ImageRepositoryList{},
		ImageRepositoryMapping{},
	)
	api.AddKnownTypes("v1beta1",
		Image{},
		ImageList{},
		ImageRepository{},
		ImageRepositoryList{},
		ImageRepositoryMapping{},
	)
}
