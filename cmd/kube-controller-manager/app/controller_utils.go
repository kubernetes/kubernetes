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

package app

import (
	"context"
	"sync"
)

// This file contains utility functions for implementing controller wrappers,
// i.e. turning whatever logic into a Controller.

type runFunc func(ctx context.Context)

type runFuncSlice []runFunc

func (rx runFuncSlice) Run(ctx context.Context) {
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

// concurrentRun returns a runFunc that wraps the given functions to run concurrently.
func concurrentRun(rx ...runFunc) runFunc {
	return runFuncSlice(rx).Run
}

// controllerLoop implements the Controller interface. It makes it easy to turn a function into a Controller.
type controllerLoop struct {
	name string
	run  runFunc
}

func newControllerLoop(run runFunc, controllerName string) *controllerLoop {
	return &controllerLoop{
		name: controllerName,
		run:  run,
	}
}

func (loop *controllerLoop) Name() string {
	return loop.name
}

func (loop *controllerLoop) Run(ctx context.Context) {
	loop.run(ctx)
}
