// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package container

import (
	"fmt"
	"log"
	"strings"
	"sync"
)

type ContainerHandlerFactory interface {
	NewContainerHandler(name string) (ContainerHandler, error)

	// for testability
	String() string
}

type factoryTreeNode struct {
	defaultFactory ContainerHandlerFactory
	children       map[string]*factoryTreeNode
}

func (self *factoryTreeNode) find(elems ...string) ContainerHandlerFactory {
	node := self
	for _, elem := range elems {
		if len(node.children) == 0 {
			break
		}
		if child, ok := node.children[elem]; ok {
			node = child
		} else {
			return node.defaultFactory
		}
	}

	return node.defaultFactory
}

func (self *factoryTreeNode) add(factory ContainerHandlerFactory, elems ...string) {
	node := self
	for _, elem := range elems {
		if node.children == nil {
			node.children = make(map[string]*factoryTreeNode, 16)
		}
		child, ok := self.children[elem]
		if !ok {
			child = &factoryTreeNode{
				defaultFactory: node.defaultFactory,
				children:       make(map[string]*factoryTreeNode, 16),
			}
			node.children[elem] = child
		}
		node = child
	}
	node.defaultFactory = factory
}

type factoryManager struct {
	root *factoryTreeNode
	lock sync.RWMutex
}

func dropEmptyString(elems ...string) []string {
	ret := make([]string, 0, len(elems))
	for _, e := range elems {
		if len(e) > 0 {
			ret = append(ret, e)
		}
	}
	return ret
}

// Must register factory for root container!
func (self *factoryManager) Register(path string, factory ContainerHandlerFactory) {
	self.lock.Lock()
	defer self.lock.Unlock()

	if self.root == nil {
		self.root = &factoryTreeNode{
			defaultFactory: nil,
			children:       make(map[string]*factoryTreeNode, 10),
		}
	}

	elems := dropEmptyString(strings.Split(path, "/")...)
	self.root.add(factory, elems...)
}

func (self *factoryManager) NewContainerHandler(path string) (ContainerHandler, error) {
	self.lock.RLock()
	defer self.lock.RUnlock()

	if self.root == nil {
		err := fmt.Errorf("nil factory for container %v: no factory registered", path)
		return nil, err
	}

	elems := dropEmptyString(strings.Split(path, "/")...)
	factory := self.root.find(elems...)
	if factory == nil {
		err := fmt.Errorf("nil factory for container %v", path)
		return nil, err
	}
	log.Printf("Container handler factory for %v is %v\n", path, factory)
	return factory.NewContainerHandler(path)
}

var globalFactoryManager factoryManager

func RegisterContainerHandlerFactory(path string, factory ContainerHandlerFactory) {
	globalFactoryManager.Register(path, factory)
}

func NewContainerHandler(path string) (ContainerHandler, error) {
	return globalFactoryManager.NewContainerHandler(path)
}
