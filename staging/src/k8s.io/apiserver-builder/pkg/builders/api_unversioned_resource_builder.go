/*
Copyright 2017 The Kubernetes Authors.

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

package builders

import (
	"k8s.io/apimachinery/pkg/runtime"
)

// NewInternalResource creates a new strategy for a resource
// name - name of the resource - e.g. "deployments"
// new - function for creating new empty UNVERSIONED instances - e.g. func() runtime.Object { return &Deployment{} }
// newList - function for creating an empty list of UNVERSIONED instances - e.g. func() runtime.Object { return &DeploymentList{} }
func NewInternalResource(name string, new, newList func() runtime.Object) UnversionedResourceBuilder {
	return NewBuilder(name, "", new, newList, true)
}

// NewInternalResourceStatus returns a new strategy for the status subresource of an object
// name - name of the resource - e.g. "deployments"
// new - function for creating new empty UNVERSIONED instances - e.g. func() runtime.Object { return &Deployment{} }
// newList - function for creating an empty list of UNVERSIONED instances - e.g. func() runtime.Object { return &DeploymentList{} }
func NewInternalResourceStatus(name string, new, newList func() runtime.Object) UnversionedResourceBuilder {
	return NewBuilder(
		name,
		"status",
		new, newList,
		true)
}

// NewInternalSubresource returns a new strategy for a subresource
// name - name of the resource - e.g. "deployments"
// path - path to the subresource - e.g. "scale"
// new - function for creating new empty UNVERSIONED instances - e.g. func() runtime.Object { return &Deployment{} }
func NewInternalSubresource(name, path string, new func() runtime.Object) UnversionedResourceBuilder {
	return NewBuilder(
		name,
		path,
		new,
		nil,   // Don't provide a list function
		false, // Don't create a full storage rest interface, just use the provide methods
	)
}

func NewBuilder(
	name, path string,
	new, newList func() runtime.Object,
	useRegistryStore bool) UnversionedResourceBuilder {

	return &UnversionedResourceBuilderImpl{
		path,
		name,
		new,
		newList,
		useRegistryStore,
	}
}

type WithList interface {
	NewList() runtime.Object
}

type UnversionedResourceBuilder interface {
	WithList
	New() runtime.Object

	GetPath() string
	GetName() string
	ShouldUseRegistryStore() bool
}

type UnversionedResourceBuilderImpl struct {
	Path             string
	Name             string
	NewFunc          func() runtime.Object
	NewListFunc      func() runtime.Object
	UseRegistryStore bool
}

func (b *UnversionedResourceBuilderImpl) GetPath() string {
	return b.Path
}

func (b *UnversionedResourceBuilderImpl) GetName() string {
	return b.Name
}

func (b *UnversionedResourceBuilderImpl) ShouldUseRegistryStore() bool {
	return b.UseRegistryStore
}

func (b *UnversionedResourceBuilderImpl) New() runtime.Object {
	if b.NewFunc == nil {
		return nil
	}
	return b.NewFunc()
}

func (b *UnversionedResourceBuilderImpl) NewList() runtime.Object {
	if b.NewListFunc == nil {
		return nil
	}
	return b.NewListFunc()
}
