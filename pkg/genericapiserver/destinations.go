/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// StorageDestinations is a mapping from API group & resource to
// the underlying storage interfaces.
type StorageDestinations struct {
	APIGroups map[string]*StorageDestinationsForAPIGroup
}

type StorageDestinationsForAPIGroup struct {
	Default   storage.Interface
	Overrides map[string]storage.Interface
}

func NewStorageDestinations() StorageDestinations {
	return StorageDestinations{
		APIGroups: map[string]*StorageDestinationsForAPIGroup{},
	}
}

// AddAPIGroup replaces 'group' if it's already registered.
func (s *StorageDestinations) AddAPIGroup(group string, defaultStorage storage.Interface) {
	glog.Infof("Adding storage destination for group %v", group)
	s.APIGroups[group] = &StorageDestinationsForAPIGroup{
		Default:   defaultStorage,
		Overrides: map[string]storage.Interface{},
	}
}

func (s *StorageDestinations) AddStorageOverride(groupResource unversioned.GroupResource, override storage.Interface) {
	group := groupResource.Group
	if _, ok := s.APIGroups[group]; !ok {
		s.AddAPIGroup(group, nil)
	}
	if s.APIGroups[group].Overrides == nil {
		s.APIGroups[group].Overrides = map[string]storage.Interface{}
	}
	s.APIGroups[group].Overrides[groupResource.Resource] = override
}

// Get finds the storage destination for the given group and resource. It will
// return an error if the group has no storage destination configured.
func (s *StorageDestinations) Get(groupResource unversioned.GroupResource) (storage.Interface, error) {
	apigroup, ok := s.APIGroups[groupResource.Group]
	if !ok {
		return nil, &NoDestinationFoundError{GroupResources: []unversioned.GroupResource{groupResource}, AvailableGroups: sets.StringKeySet(s.APIGroups)}
	}

	if client, exists := apigroup.Overrides[groupResource.Resource]; exists {
		return client, nil
	}
	return apigroup.Default, nil
}

// Search is like Get, but can be used to search a list of groups. It tries the
// groups in order and returns an error if none of them exist. The intention is for
// this to be used for resources that move between groups.
func (s *StorageDestinations) Search(resources []unversioned.GroupResource) (storage.Interface, error) {
	for _, resource := range resources {
		storage, err := s.Get(resource)
		if IsNoDestinationFoundError(err) {
			continue
		}
		if err != nil {
			return nil, err
		}
		return storage, nil
	}

	return nil, &NoDestinationFoundError{GroupResources: resources, AvailableGroups: sets.StringKeySet(s.APIGroups)}
}

// Get all backends for all registered storage destinations.
// Used for getting all instances for health validations.
func (s *StorageDestinations) Backends() []string {
	backends := sets.String{}
	for _, group := range s.APIGroups {
		if group.Default != nil {
			for _, backend := range group.Default.Backends(context.TODO()) {
				backends.Insert(backend)
			}
		}
		if group.Overrides != nil {
			for _, storage := range group.Overrides {
				for _, backend := range storage.Backends(context.TODO()) {
					backends.Insert(backend)
				}
			}
		}
	}
	return backends.List()
}

type NoDestinationFoundError struct {
	GroupResources  []unversioned.GroupResource
	AvailableGroups sets.String
}

func (e *NoDestinationFoundError) Error() string {
	return fmt.Sprintf("No storage defined for: '%v'. Defined groups: %v", e.GroupResources, e.AvailableGroups.List())
}

func IsNoDestinationFoundError(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(*NoDestinationFoundError)
	return ok
}
