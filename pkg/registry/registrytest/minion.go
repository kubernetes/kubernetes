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

package registrytest

import (
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type MinionRegistry struct {
	Err     error
	Minion  string
	Minions api.MinionList
	sync.Mutex
}

func MakeMinionList(minions []string) *api.MinionList {
	list := api.MinionList{
		Items: make([]api.Minion, len(minions)),
	}
	for i := range minions {
		list.Items[i].ID = minions[i]
	}
	return &list
}

func NewMinionRegistry(minions []string) *MinionRegistry {
	return &MinionRegistry{
		Minions: *MakeMinionList(minions),
	}
}

func (r *MinionRegistry) List() (*api.MinionList, error) {
	r.Lock()
	defer r.Unlock()
	return &r.Minions, r.Err
}

func (r *MinionRegistry) Insert(minion string) error {
	r.Lock()
	defer r.Unlock()
	r.Minion = minion
	r.Minions.Items = append(r.Minions.Items, api.Minion{JSONBase: api.JSONBase{ID: minion}})
	return r.Err
}

func (r *MinionRegistry) Contains(nodeID string) (bool, error) {
	r.Lock()
	defer r.Unlock()
	for _, node := range r.Minions.Items {
		if node.ID == nodeID {
			return true, r.Err
		}
	}
	return false, r.Err
}

func (r *MinionRegistry) Delete(minion string) error {
	r.Lock()
	defer r.Unlock()
	var newList []api.Minion
	for _, node := range r.Minions.Items {
		if node.ID != minion {
			newList = append(newList, api.Minion{JSONBase: api.JSONBase{ID: minion}})
		}
	}
	r.Minions.Items = newList
	return r.Err
}
