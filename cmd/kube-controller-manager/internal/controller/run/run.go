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

// Package run contains utility functions for implementing controller wrappers,
// i.e. turning whatever logic into a controller.Controller.
package run

import (
	"context"
	"sync"
)

type Func func(ctx context.Context)

type FuncSlice []Func

func (rx FuncSlice) Run(ctx context.Context) {
	var wg sync.WaitGroup
	wg.Add(len(rx))
	for _, fnc := range rx {
		go func() {
			defer wg.Done()
			fnc(ctx)
		}()
	}
	wg.Wait()
}

// Concurrent returns a Func that wraps the given functions to run concurrently.
func Concurrent(rx ...Func) Func {
	return FuncSlice(rx).Run
}

// ControllerLoop implements the controller.Controller interface. It makes it easy to turn a function into a Controller.
type ControllerLoop struct {
	name string
	run  Func
}

// NewControllerLoop creates a new ControllerLoop using the given Func and name.
func NewControllerLoop(run Func, controllerName string) *ControllerLoop {
	return &ControllerLoop{
		name: controllerName,
		run:  run,
	}
}

// Name implements controller.Controller interface.
func (loop *ControllerLoop) Name() string {
	return loop.name
}

// Run implements controller.Controller interface.
func (loop *ControllerLoop) Run(ctx context.Context) {
	loop.run(ctx)
}
