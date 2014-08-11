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

import "sync"

type MinionRegistry struct {
	Err     error
	Minion  string
	Minions []string
	sync.Mutex
}

func NewMinionRegistry(minions []string) *MinionRegistry {
	return &MinionRegistry{
		Minions: minions,
	}
}

func (r *MinionRegistry) List() ([]string, error) {
	r.Lock()
	defer r.Unlock()
	return r.Minions, r.Err
}

func (r *MinionRegistry) Insert(minion string) error {
	r.Lock()
	defer r.Unlock()
	r.Minion = minion
	return r.Err
}

func (r *MinionRegistry) Contains(minion string) (bool, error) {
	r.Lock()
	defer r.Unlock()
	for _, name := range r.Minions {
		if name == minion {
			return true, r.Err
		}
	}
	return false, r.Err
}

func (r *MinionRegistry) Delete(minion string) error {
	r.Lock()
	defer r.Unlock()
	var newList []string
	for _, name := range r.Minions {
		if name != minion {
			newList = append(newList, name)
		}
	}
	r.Minions = newList
	return r.Err
}
