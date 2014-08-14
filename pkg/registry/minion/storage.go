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

package minion

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// RegistryStorage implements the RESTStorage interface, backed by a MinionRegistry.
type RegistryStorage struct {
	registry Registry
}

// NewRegistryStorage returns a new RegistryStorage.
func NewRegistryStorage(m Registry) apiserver.RESTStorage {
	return &RegistryStorage{
		registry: m,
	}
}

func (rs *RegistryStorage) Create(obj interface{}) (<-chan interface{}, error) {
	minion, ok := obj.(*api.Minion)
	if !ok {
		return nil, fmt.Errorf("not a minion: %#v", obj)
	}
	if minion.ID == "" {
		return nil, fmt.Errorf("ID should not be empty: %#v", minion)
	}

	minion.CreationTimestamp = util.Now()

	return apiserver.MakeAsync(func() (interface{}, error) {
		err := rs.registry.Insert(minion.ID)
		if err != nil {
			return nil, err
		}
		contains, err := rs.registry.Contains(minion.ID)
		if err != nil {
			return nil, err
		}
		if contains {
			return rs.toApiMinion(minion.ID), nil
		}
		return nil, fmt.Errorf("unable to add minion %#v", minion)
	}), nil
}

func (rs *RegistryStorage) Delete(id string) (<-chan interface{}, error) {
	exists, err := rs.registry.Contains(id)
	if !exists {
		return nil, ErrDoesNotExist
	}
	if err != nil {
		return nil, err
	}
	return apiserver.MakeAsync(func() (interface{}, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.Delete(id)
	}), nil
}

func (rs *RegistryStorage) Get(id string) (interface{}, error) {
	exists, err := rs.registry.Contains(id)
	if !exists {
		return nil, ErrDoesNotExist
	}
	return rs.toApiMinion(id), err
}

func (rs *RegistryStorage) List(selector labels.Selector) (interface{}, error) {
	nameList, err := rs.registry.List()
	if err != nil {
		return nil, err
	}
	var list api.MinionList
	for _, name := range nameList {
		list.Items = append(list.Items, rs.toApiMinion(name))
	}
	return list, nil
}

func (rs RegistryStorage) New() interface{} {
	return &api.Minion{}
}

func (rs *RegistryStorage) Update(minion interface{}) (<-chan interface{}, error) {
	return nil, fmt.Errorf("Minions can only be created (inserted) and deleted.")
}

func (rs *RegistryStorage) toApiMinion(name string) api.Minion {
	return api.Minion{JSONBase: api.JSONBase{ID: name}}
}
