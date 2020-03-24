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

type Console interface {
	io.Reader
	io.Writer
	io.Closer

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
	// Fd returns the console's file descriptor
	Fd() uintptr
	// Name returns the console's file name
	Name() string
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

// Current returns the current processes console
func Current() Console {
	c, err := ConsoleFromFile(os.Stdin)
	if err != nil {
		// stdin should always be a console for the design
		// of this function
		panic(err)
	}
	return c
}

// ConsoleFromFile returns a console using the provided file
func ConsoleFromFile(f *os.File) (Console, error) {
	if err := checkConsole(f); err != nil {
		return nil, err
	}
	return newMaster(f)
}
