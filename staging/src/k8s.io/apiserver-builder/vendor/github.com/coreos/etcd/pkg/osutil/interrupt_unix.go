// Copyright 2015 The etcd Authors
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

// +build !windows,!plan9

package osutil

import (
	"os"
	"os/signal"
	"sync"
	"syscall"
)

// InterruptHandler is a function that is called on receiving a
// SIGTERM or SIGINT signal.
type InterruptHandler func()

var (
	interruptRegisterMu, interruptExitMu sync.Mutex
	// interruptHandlers holds all registered InterruptHandlers in order
	// they will be executed.
	interruptHandlers = []InterruptHandler{}
)

// RegisterInterruptHandler registers a new InterruptHandler. Handlers registered
// after interrupt handing was initiated will not be executed.
func RegisterInterruptHandler(h InterruptHandler) {
	interruptRegisterMu.Lock()
	defer interruptRegisterMu.Unlock()
	interruptHandlers = append(interruptHandlers, h)
}

// HandleInterrupts calls the handler functions on receiving a SIGINT or SIGTERM.
func HandleInterrupts() {
	notifier := make(chan os.Signal, 1)
	signal.Notify(notifier, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-notifier

		interruptRegisterMu.Lock()
		ihs := make([]InterruptHandler, len(interruptHandlers))
		copy(ihs, interruptHandlers)
		interruptRegisterMu.Unlock()

		interruptExitMu.Lock()

		plog.Noticef("received %v signal, shutting down...", sig)

		for _, h := range ihs {
			h()
		}
		signal.Stop(notifier)
		pid := syscall.Getpid()
		// exit directly if it is the "init" process, since the kernel will not help to kill pid 1.
		if pid == 1 {
			os.Exit(0)
		}
		syscall.Kill(pid, sig.(syscall.Signal))
	}()
}

// Exit relays to os.Exit if no interrupt handlers are running, blocks otherwise.
func Exit(code int) {
	interruptExitMu.Lock()
	os.Exit(code)
}
