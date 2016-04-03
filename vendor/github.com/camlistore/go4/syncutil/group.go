/*
Copyright 2013 Google Inc.

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

package syncutil

import "sync"

// A Group is like a sync.WaitGroup and coordinates doing
// multiple things at once. Its zero value is ready to use.
type Group struct {
	wg   sync.WaitGroup
	mu   sync.Mutex // guards errs
	errs []error
}

// Go runs fn in its own goroutine, but does not wait for it to complete.
// Call Err or Errs to wait for all the goroutines to complete.
func (g *Group) Go(fn func() error) {
	g.wg.Add(1)
	go func() {
		defer g.wg.Done()
		err := fn()
		if err != nil {
			g.mu.Lock()
			defer g.mu.Unlock()
			g.errs = append(g.errs, err)
		}
	}()
}

// Wait waits for all the previous calls to Go to complete.
func (g *Group) Wait() {
	g.wg.Wait()
}

// Err waits for all previous calls to Go to complete and returns the
// first non-nil error, or nil.
func (g *Group) Err() error {
	g.wg.Wait()
	if len(g.errs) > 0 {
		return g.errs[0]
	}
	return nil
}

// Errs waits for all previous calls to Go to complete and returns
// all non-nil errors.
func (g *Group) Errs() []error {
	g.wg.Wait()
	return g.errs
}
