/*
   Copyright The containerd Authors.

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

package console

import (
	"errors"
	"io"
	"os"
)

var ErrNotAConsole = errors.New("provided file is not a console")

type File interface {
	io.ReadWriteCloser

	// Fd returns its file descriptor
	Fd() uintptr
	// Name returns its file name
	Name() string
}

type Console interface {
	File

	// Resize resizes the console to the provided window size
	Resize(WinSize) error
	// ResizeFrom resizes the calling console to the size of the
	// provided console
	ResizeFrom(Console) error
	// SetRaw sets the console in raw mode
	SetRaw() error
	// DisableEcho disables echo on the console
	DisableEcho() error
	// Reset restores the console to its orignal state
	Reset() error
	// Size returns the window size of the console
	Size() (WinSize, error)
}

// WinSize specifies the window size of the console
type WinSize struct {
	// Height of the console
	Height uint16
	// Width of the console
	Width uint16
	x     uint16
	y     uint16
}

// Current returns the current process' console
func Current() (c Console) {
	var err error
	// Usually all three streams (stdin, stdout, and stderr)
	// are open to the same console, but some might be redirected,
	// so try all three.
	for _, s := range []*os.File{os.Stderr, os.Stdout, os.Stdin} {
		if c, err = ConsoleFromFile(s); err == nil {
			return c
		}
	}
	// One of the std streams should always be a console
	// for the design of this function.
	panic(err)
}

// ConsoleFromFile returns a console using the provided file
// nolint:golint
func ConsoleFromFile(f File) (Console, error) {
	if err := checkConsole(f); err != nil {
		return nil, err
	}
	return newMaster(f)
}
