/*
Copyright 2025 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
)

func NewIdentifier(name string, exampleObject runtime.Object) *Identifier {
	id := &Identifier{name: name, itemType: itemType(exampleObject)}
	id.tryMakeUnique()
	return id
}

// Identifier is used to identify of informers and FIFO for metrics and logging purposes.
type Identifier struct {
	// Name is the name of proposed name to reference.
	name string
	// ItemType is the type of item type like "v1.Pod".
	itemType string
}

func (id *Identifier) Name() string {
	if id == nil {
		return ""
	}
	return id.name
}

func (id *Identifier) ItemType() string {
	if id == nil {
		return ""
	}
	return id.itemType
}

func (id *Identifier) WithObjectType(exampleObject runtime.Object) *Identifier {
	if id == nil {
		return nil
	}
	itemType := itemType(exampleObject)
	if id.itemType == itemType {
		return id
	}
	newId := *id
	newId.itemType = itemType
	newId.tryMakeUnique()
	return &newId
}

func itemType(exampleObject runtime.Object) string {
	return reflect.TypeOf(exampleObject).Elem().String()
}

func (id *Identifier) IsUnique() bool {
	if id == nil {
		return false
	}
	if id.name == "" || id.itemType == "" {
		return false
	}
	identityLock.RLock()
	defer identityLock.RUnlock()
	return identityRepresentative[*id] == id
}

func (id *Identifier) tryMakeUnique() {
	if id.name == "" || id.itemType == "" {
		return
	}
	if id.IsUnique() {
		return
	}
	identityLock.Lock()
	defer identityLock.Unlock()
	if identityCounter[*id] > 0 {
		identityCounter[*id]++
		// Try to give suffix to make it unique.
		id.name = fmt.Sprintf("%s-%d", id.name, identityCounter[*id]-1)
		if identityCounter[*id] > 0 {
			// Prevent case where user provided name with the name suffix pattern.
			return
		}
	}
	identityCounter[*id]++
	identityRepresentative[*id] = id
}

var identityLock sync.RWMutex
var identityCounter map[Identifier]int
var identityRepresentative map[Identifier]*Identifier

func init() {
	resetIdentity()
}

func resetIdentity() {
	identityLock.Lock()
	identityCounter = make(map[Identifier]int)
	identityRepresentative = make(map[Identifier]*Identifier)
	identityLock.Unlock()
}
