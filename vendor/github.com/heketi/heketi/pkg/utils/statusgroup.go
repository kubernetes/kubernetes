//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package utils

import (
	"sync"
)

type StatusGroup struct {
	wg      sync.WaitGroup
	results chan error
	err     error
}

// Create a new goroutine error status collector
func NewStatusGroup() *StatusGroup {
	s := &StatusGroup{}
	s.results = make(chan error, 1)

	return s
}

// Adds to the number of goroutines it should wait
func (s *StatusGroup) Add(delta int) {
	s.wg.Add(delta)
}

// Removes the number of pending goroutines by one
func (s *StatusGroup) Done() {
	s.wg.Done()
}

// Goroutine can return an error back to caller
func (s *StatusGroup) Err(err error) {
	s.results <- err
}

// Returns an error if any of the spawned goroutines
// return an error.  Only the last error is saved.
// This function must be called last after the last
// s.Register() function
func (s *StatusGroup) Result() error {

	// This goroutine will wait until all
	// other privously spawned goroutines finish.
	// Once they finish, it will close the channel
	go func() {
		s.wg.Wait()
		close(s.results)
	}()

	// Read from the channel until close
	for err := range s.results {
		// Only save the last one
		if err != nil {
			s.err = err
		}
	}

	return s.err
}
