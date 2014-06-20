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
	"strings"
	"testing"

	"github.com/stretchr/testify/mock"
)

type mockContainerHandlerFactory struct {
	mock.Mock
	Name string
}

func (self *mockContainerHandlerFactory) String() string {
	return self.Name
}

func (self *mockContainerHandlerFactory) NewContainerHandler(name string) (ContainerHandler, error) {
	args := self.Called(name)
	return args.Get(0).(ContainerHandler), args.Error(1)
}

func testExpectedFactory(root *factoryTreeNode, path, expectedFactory string, t *testing.T) {
	elems := dropEmptyString(strings.Split(path, "/")...)
	factory := root.find(elems...)
	if factory.String() != expectedFactory {
		t.Errorf("factory %v should be used to create container %v. but %v is selected",
			expectedFactory,
			path,
			factory)
	}
}

func testAddFactory(root *factoryTreeNode, path string) *factoryTreeNode {
	elems := dropEmptyString(strings.Split(path, "/")...)
	if root == nil {
		root = &factoryTreeNode{
			defaultFactory: nil,
		}
	}
	f := &mockContainerHandlerFactory{
		Name: path,
	}
	root.add(f, elems...)
	return root
}

func TestFactoryTree(t *testing.T) {
	root := testAddFactory(nil, "/")
	root = testAddFactory(root, "/docker")
	root = testAddFactory(root, "/user")
	root = testAddFactory(root, "/user/special/containers")

	testExpectedFactory(root, "/docker/container", "/docker", t)
	testExpectedFactory(root, "/docker", "/docker", t)
	testExpectedFactory(root, "/", "/", t)
	testExpectedFactory(root, "/user/deep/level/container", "/user", t)
	testExpectedFactory(root, "/user/special/containers", "/user/special/containers", t)
	testExpectedFactory(root, "/user/special/containers/container", "/user/special/containers", t)
	testExpectedFactory(root, "/other", "/", t)
}
