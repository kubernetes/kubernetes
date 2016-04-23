/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package genericapiserver

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/runtime"
)

type ResourceEncodingConfig interface {
	// StorageEncoding returns the serialization format for the resource.
	// TODO this should actually return a GroupVersionKind since you can logically have multiple "matching" Kinds
	// For now, it returns just the GroupVersion for consistency with old behavior
	StorageEncodingFor(unversioned.GroupResource) (unversioned.GroupVersion, error)

	// InMemoryEncodingFor returns the groupVersion for the in memory representation the storage should convert to.
	InMemoryEncodingFor(unversioned.GroupResource) (unversioned.GroupVersion, error)
}

type DefaultResourceEncodingConfig struct {
	Groups map[string]*GroupResourceEncodingConfig
}

type GroupResourceEncodingConfig struct {
	DefaultExternalEncoding   unversioned.GroupVersion
	ExternalResourceEncodings map[string]unversioned.GroupVersion

	DefaultInternalEncoding   unversioned.GroupVersion
	InternalResourceEncodings map[string]unversioned.GroupVersion
}

var _ ResourceEncodingConfig = &DefaultResourceEncodingConfig{}

func NewDefaultResourceEncodingConfig() *DefaultResourceEncodingConfig {
	return &DefaultResourceEncodingConfig{Groups: map[string]*GroupResourceEncodingConfig{}}
}

func newGroupResourceEncodingConfig(defaultEncoding, defaultInternalVersion unversioned.GroupVersion) *GroupResourceEncodingConfig {
	return &GroupResourceEncodingConfig{
		DefaultExternalEncoding: defaultEncoding, ExternalResourceEncodings: map[string]unversioned.GroupVersion{},
		DefaultInternalEncoding: defaultInternalVersion, InternalResourceEncodings: map[string]unversioned.GroupVersion{},
	}
}

func (o *DefaultResourceEncodingConfig) SetVersionEncoding(group string, externalEncodingVersion, internalVersion unversioned.GroupVersion) {
	_, groupExists := o.Groups[group]
	if !groupExists {
		o.Groups[group] = newGroupResourceEncodingConfig(externalEncodingVersion, internalVersion)
	}

	o.Groups[group].DefaultExternalEncoding = externalEncodingVersion
	o.Groups[group].DefaultInternalEncoding = internalVersion
}

func (o *DefaultResourceEncodingConfig) SetResourceEncoding(resourceBeingStored unversioned.GroupResource, externalEncodingVersion, internalVersion unversioned.GroupVersion) {
	group := resourceBeingStored.Group
	_, groupExists := o.Groups[group]
	if !groupExists {
		o.Groups[group] = newGroupResourceEncodingConfig(externalEncodingVersion, internalVersion)
	}

	o.Groups[group].ExternalResourceEncodings[resourceBeingStored.Resource] = externalEncodingVersion
	o.Groups[group].InternalResourceEncodings[resourceBeingStored.Resource] = internalVersion
}

func (o *DefaultResourceEncodingConfig) StorageEncodingFor(resource unversioned.GroupResource) (unversioned.GroupVersion, error) {
	groupMeta, err := registered.Group(resource.Group)
	if err != nil {
		return unversioned.GroupVersion{}, err
	}

	groupEncoding, groupExists := o.Groups[resource.Group]

	if !groupExists {
		// return the most preferred external version for the group
		return groupMeta.GroupVersion, nil
	}

	resourceOverride, resourceExists := groupEncoding.ExternalResourceEncodings[resource.Resource]
	if !resourceExists {
		return groupEncoding.DefaultExternalEncoding, nil
	}

	return resourceOverride, nil
}

func (o *DefaultResourceEncodingConfig) InMemoryEncodingFor(resource unversioned.GroupResource) (unversioned.GroupVersion, error) {
	if _, err := registered.Group(resource.Group); err != nil {
		return unversioned.GroupVersion{}, err
	}

	groupEncoding, groupExists := o.Groups[resource.Group]
	if !groupExists {
		return unversioned.GroupVersion{Group: resource.Group, Version: runtime.APIVersionInternal}, nil
	}

	resourceOverride, resourceExists := groupEncoding.InternalResourceEncodings[resource.Resource]
	if !resourceExists {
		return groupEncoding.DefaultInternalEncoding, nil
	}

	return resourceOverride, nil
}
