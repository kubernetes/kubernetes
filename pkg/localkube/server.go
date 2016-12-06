/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package localkube

import (
	"os"
	"time"
)

// Server represents a component that Kubernetes depends on. It allows for the management of
// the lifecycle of the component.
type Server interface {
	// Start immediately starts the component.
	Start()

	// Stop begins the process of stopping the component.
	Stop()

	// Name returns a unique identifier for the component.
	Name() string
}

// SimpleServer provides a minimal implementation of Server.
type SimpleServer struct {
	ComponentName string
	Interval      time.Duration

	serverRoutine func() error
	stopChannel   chan struct{}
}

func NewSimpleServer(componentName string, msInterval int32, serverRoutine func() error) *SimpleServer {
	return &SimpleServer{
		ComponentName: componentName,
		Interval:      time.Duration(msInterval) * time.Millisecond,

		serverRoutine: serverRoutine,
		stopChannel:   make(chan struct{}),
	}
}

// Start calls startup function.
func (s *SimpleServer) Start() {
	go Until(s.serverRoutine, os.Stdout, s.ComponentName, s.Interval, s.stopChannel)
}

// Stop calls shutdown function.
func (s *SimpleServer) Stop() {
	close(s.stopChannel)
}

// Name returns the name of the service.
func (s SimpleServer) Name() string {
	return s.ComponentName
}
