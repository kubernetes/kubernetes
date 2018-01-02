// Copyright 2015 The rkt Authors
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

package testutils

import (
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/coreos/gexpect"
)

type GoroutineAssistant struct {
	s  chan error
	wg sync.WaitGroup
	t  *testing.T
}

func NewGoroutineAssistant(t *testing.T) *GoroutineAssistant {
	return &GoroutineAssistant{
		s: make(chan error),
		t: t,
	}
}

func (a *GoroutineAssistant) Fatalf(s string, args ...interface{}) {
	a.s <- fmt.Errorf(s, args...)
	runtime.Goexit()
}

func (a *GoroutineAssistant) Add(n int) {
	a.wg.Add(n)
}

func (a *GoroutineAssistant) Done() {
	a.wg.Done()
}

func (a *GoroutineAssistant) Wait() {
	go func() {
		a.wg.Wait()
		a.s <- nil
	}()
	err := <-a.s
	if err == nil {
		// success
		return
	}
	// Let's mimic testing.Fatalf()'s behavior and fatal with the first received error.
	// All other goroutines will be killed.
	a.t.Logf("Error encountered - shutting down")
	a.t.Fatal(err)
}

func (a *GoroutineAssistant) WaitOrFail(child *gexpect.ExpectSubprocess) {
	err := child.Wait()
	if err != nil {
		a.Fatalf("rkt didn't terminate correctly: %v", err)
	}
}

func (a *GoroutineAssistant) SpawnOrFail(cmd string) *gexpect.ExpectSubprocess {
	a.t.Logf("Command: %v", cmd)
	child, err := gexpect.Spawn(cmd)
	if err != nil {
		a.Fatalf("Cannot exec rkt: %v", err)
	}
	return child
}
