//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), as published by the Free Software Foundation,
// or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
// http://www.apache.org/licenses/LICENSE-2.0>.
//
// You may not use this file except in compliance with those terms.
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
