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

package container_test

import (
	"testing"

	"github.com/google/cadvisor/container"
	containertest "github.com/google/cadvisor/container/testing"
	"github.com/google/cadvisor/manager/watcher"

	"github.com/stretchr/testify/mock"
)

type mockContainerHandlerFactory struct {
	mock.Mock
	Name           string
	CanHandleValue bool
	CanAcceptValue bool
}

func (self *mockContainerHandlerFactory) String() string {
	return self.Name
}

func (self *mockContainerHandlerFactory) DebugInfo() map[string][]string {
	return map[string][]string{}
}

func (self *mockContainerHandlerFactory) CanHandleAndAccept(name string) (bool, bool, error) {
	return self.CanHandleValue, self.CanAcceptValue, nil
}

func (self *mockContainerHandlerFactory) NewContainerHandler(name string, isHostNamespace bool) (container.ContainerHandler, error) {
	args := self.Called(name)
	return args.Get(0).(container.ContainerHandler), args.Error(1)
}

const testContainerName = "/test"

var mockFactory containertest.FactoryForMockContainerHandler

func TestNewContainerHandler_FirstMatches(t *testing.T) {
	container.ClearContainerHandlerFactories()

	// Register one allways yes factory.
	allwaysYes := &mockContainerHandlerFactory{
		Name:           "yes",
		CanHandleValue: true,
		CanAcceptValue: true,
	}
	container.RegisterContainerHandlerFactory(allwaysYes, []watcher.ContainerWatchSource{watcher.Raw})

	// The yes factory should be asked to create the ContainerHandler.
	mockContainer, err := mockFactory.NewContainerHandler(testContainerName, true)
	if err != nil {
		t.Error(err)
	}
	allwaysYes.On("NewContainerHandler", testContainerName).Return(mockContainer, nil)

	cont, _, err := container.NewContainerHandler(testContainerName, watcher.Raw, true)
	if err != nil {
		t.Error(err)
	}
	if cont == nil {
		t.Error("Expected container to not be nil")
	}
}

func TestNewContainerHandler_SecondMatches(t *testing.T) {
	container.ClearContainerHandlerFactories()

	// Register one allways no and one always yes factory.
	allwaysNo := &mockContainerHandlerFactory{
		Name:           "no",
		CanHandleValue: false,
		CanAcceptValue: true,
	}
	container.RegisterContainerHandlerFactory(allwaysNo, []watcher.ContainerWatchSource{watcher.Raw})
	allwaysYes := &mockContainerHandlerFactory{
		Name:           "yes",
		CanHandleValue: true,
		CanAcceptValue: true,
	}
	container.RegisterContainerHandlerFactory(allwaysYes, []watcher.ContainerWatchSource{watcher.Raw})

	// The yes factory should be asked to create the ContainerHandler.
	mockContainer, err := mockFactory.NewContainerHandler(testContainerName, true)
	if err != nil {
		t.Error(err)
	}
	allwaysYes.On("NewContainerHandler", testContainerName).Return(mockContainer, nil)

	cont, _, err := container.NewContainerHandler(testContainerName, watcher.Raw, true)
	if err != nil {
		t.Error(err)
	}
	if cont == nil {
		t.Error("Expected container to not be nil")
	}
}

func TestNewContainerHandler_NoneMatch(t *testing.T) {
	container.ClearContainerHandlerFactories()

	// Register two allways no factories.
	allwaysNo1 := &mockContainerHandlerFactory{
		Name:           "no",
		CanHandleValue: false,
		CanAcceptValue: true,
	}
	container.RegisterContainerHandlerFactory(allwaysNo1, []watcher.ContainerWatchSource{watcher.Raw})
	allwaysNo2 := &mockContainerHandlerFactory{
		Name:           "no",
		CanHandleValue: false,
		CanAcceptValue: true,
	}
	container.RegisterContainerHandlerFactory(allwaysNo2, []watcher.ContainerWatchSource{watcher.Raw})

	_, _, err := container.NewContainerHandler(testContainerName, watcher.Raw, true)
	if err == nil {
		t.Error("Expected NewContainerHandler to fail")
	}
}

func TestNewContainerHandler_Accept(t *testing.T) {
	container.ClearContainerHandlerFactories()

	// Register handler that can handle the container, but can't accept it.
	cannotHandle := &mockContainerHandlerFactory{
		Name:           "no",
		CanHandleValue: false,
		CanAcceptValue: true,
	}
	container.RegisterContainerHandlerFactory(cannotHandle, []watcher.ContainerWatchSource{watcher.Raw})
	cannotAccept := &mockContainerHandlerFactory{
		Name:           "no",
		CanHandleValue: true,
		CanAcceptValue: false,
	}
	container.RegisterContainerHandlerFactory(cannotAccept, []watcher.ContainerWatchSource{watcher.Raw})

	_, accept, err := container.NewContainerHandler(testContainerName, watcher.Raw, true)
	if err != nil {
		t.Error("Expected NewContainerHandler to succeed")
	}
	if accept == true {
		t.Error("Expected NewContainerHandler to ignore the container.")
	}
}
