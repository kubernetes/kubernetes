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

func MakeMinionList(minions []string, nodeResources api.NodeResources) *api.MinionList {
	list := api.MinionList{
		Items: make([]api.Minion, len(minions)),
	}
	for i := range minions {
		list.Items[i].Name = minions[i]
		list.Items[i].NodeResources = nodeResources
	}
	return &list
}

func NewMinionRegistry(minions []string, nodeResources api.NodeResources) *MinionRegistry {
	return &MinionRegistry{
		Minions: *MakeMinionList(minions, nodeResources),
	}
}

func (r *MinionRegistry) ListMinions(ctx api.Context) (*api.MinionList, error) {
	r.Lock()
	defer r.Unlock()
	return &r.Minions, r.Err
}

func (r *MinionRegistry) CreateMinion(ctx api.Context, minion *api.Minion) error {
	r.Lock()
	defer r.Unlock()
	r.Minion = minion.Name
	r.Minions.Items = append(r.Minions.Items, *minion)
	return r.Err
}

func (r *MinionRegistry) GetMinion(ctx api.Context, minionID string) (*api.Minion, error) {
	r.Lock()
	defer r.Unlock()
	for _, node := range r.Minions.Items {
		if node.Name == minionID {
			return &node, r.Err
		}
	}
	return nil, r.Err
}

func (r *MinionRegistry) DeleteMinion(ctx api.Context, minionID string) error {
	r.Lock()
	defer r.Unlock()
	var newList []api.Minion
	for _, node := range r.Minions.Items {
		if node.Name != minionID {
			newList = append(newList, api.Minion{ObjectMeta: api.ObjectMeta{Name: node.Name}})
		}
	}
	r.Minions.Items = newList
	return r.Err
}
