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

package term

import (
	"io"
	"os"

	"github.com/docker/docker/pkg/term"
	"k8s.io/kubernetes/pkg/util/interrupt"
)

// SafeFunc is a function to be invoked by TTY.
type SafeFunc func() error

// TTY helps invoke a function and preserve the state of the terminal, even if the
// process is terminated during execution.
type TTY struct {
	// In is a reader to check for a terminal.
	In io.Reader
	// Raw is true if the terminal should be set raw.
	Raw bool
	// TryDev indicates the TTY should try to open /dev/tty if the provided input
	// is not a file descriptor.
	TryDev bool
	// Parent is an optional interrupt handler provided to this function - if provided
	// it will be invoked after the terminal state is restored. If it is not provided,
	// a signal received during the TTY will result in os.Exit(0) being invoked.
	Parent *interrupt.Handler
}

// fd returns a file descriptor for a given object.
type fd interface {
	Fd() uintptr
}

// IsTerminal returns true if the provided input is a terminal. Does not check /dev/tty
// even if TryDev is set.
func (t TTY) IsTerminal() bool {
	return IsTerminal(t.In)
}

// Safe invokes the provided function and will attempt to ensure that when the
// function returns (or a termination signal is sent) that the terminal state
// is reset to the condition it was in prior to the function being invoked. If
// t.Raw is true the terminal will be put into raw mode prior to calling the function.
// If the input file descriptor is not a TTY and TryDev is true, the /dev/tty file
// will be opened (if available).
func (t TTY) Safe(fn SafeFunc) error {
	in := t.In

	var hasFd bool
	var inFd uintptr
	if desc, ok := in.(fd); ok && in != nil {
		inFd = desc.Fd()
		hasFd = true
	}
	if t.TryDev && (!hasFd || !term.IsTerminal(inFd)) {
		if f, err := os.Open("/dev/tty"); err == nil {
			defer f.Close()
			inFd = f.Fd()
			hasFd = true
		}
	}
	if !hasFd || !term.IsTerminal(inFd) {
		return fn()
	}

	var state *term.State
	var err error
	if t.Raw {
		state, err = term.MakeRaw(inFd)
	} else {
		state, err = term.SaveState(inFd)
	}
	if err != nil {
		return err
	}
	return interrupt.Chain(t.Parent, func() { term.RestoreTerminal(inFd, state) }).Run(fn)
}

// IsTerminal returns whether the passed io.Reader is a terminal or not
func IsTerminal(r io.Reader) bool {
	file, ok := r.(fd)
	return ok && term.IsTerminal(file.Fd())
}
