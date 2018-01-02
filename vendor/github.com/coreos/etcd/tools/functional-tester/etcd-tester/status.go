// Copyright 2016 The etcd Authors
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

package main

import (
	"sync"
	"time"
)

type Status struct {
	Since      time.Time
	Failures   []string
	RoundLimit int

	Cluster ClusterStatus
	cluster *cluster

	mu    sync.Mutex // guards Round and Case
	Round int
	Case  int
}

func (s *Status) setRound(r int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Round = r
}

func (s *Status) getRound() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Round
}

func (s *Status) setCase(c int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Case = c
}

func (s *Status) getCase() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.Case
}
