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
	"sort"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var ErrDoesNotExist = fmt.Errorf("The requested resource does not exist.")

// Keep track of a set of minions. Safe for concurrent reading/writing.
type Registry interface {
	List() (currentMinions []string, err error)
	Insert(minion string) error
	Delete(minion string) error
	Contains(minion string) (bool, error)
}

// Initialize a minion registry with a list of minions.
func NewRegistry(minions []string) Registry {
	m := &minionList{
		minions: util.StringSet{},
	}
	for _, minion := range minions {
		m.minions.Insert(minion)
	}
	return m
}

type minionList struct {
	minions util.StringSet
	lock    sync.Mutex
}

func (m *minionList) Contains(minion string) (bool, error) {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.minions.Has(minion), nil
}

func (m *minionList) Delete(minion string) error {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.minions.Delete(minion)
	return nil
}

func (m *minionList) Insert(newMinion string) error {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.minions.Insert(newMinion)
	return nil
}

func (m *minionList) List() (currentMinions []string, err error) {
	m.lock.Lock()
	defer m.lock.Unlock()
	for minion := range m.minions {
		currentMinions = append(currentMinions, minion)
	}
	sort.StringSlice(currentMinions).Sort()
	return
}
