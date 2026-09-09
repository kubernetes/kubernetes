/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"

	"github.com/moby/term"
	"k8s.io/apimachinery/pkg/util/runtime"
)

// TerminalSize represents the width and height of a terminal.
// It is the same as staging/src/k8s.io/client-go/tools/remotecommand.TerminalSize.
// Copied to decouple the packages. Terminal-related package should not depend on API client and vice versa.
type TerminalSize struct {
	Width  uint16
	Height uint16
}

// TerminalSizeQueue is capable of returning terminal resize events as they occur.
// It is the same as staging/src/k8s.io/client-go/tools/remotecommand.TerminalSizeQueue.
// Copied to decouple the packages. Terminal-related package should not depend on API client and vice versa.
type TerminalSizeQueue interface {
	// Next returns the new terminal size after the terminal has been resized. It returns nil when
	// monitoring has been stopped.
	Next() *TerminalSize
}

// GetSize returns the current size of the user's terminal. If it isn't a terminal,
// nil is returned.
func (t TTY) GetSize() *TerminalSize {
	outFd, isTerminal := term.GetFdInfo(t.Out)
	if !isTerminal {
		return nil
	}
	return GetSize(outFd)
}

// GetSize returns the current size of the terminal associated with fd.
func GetSize(fd uintptr) *TerminalSize {
	winsize, err := term.GetWinsize(fd)
	if err != nil {
		runtime.HandleError(fmt.Errorf("unable to get terminal size: %v", err))
		return nil
	}

	return &TerminalSize{Width: winsize.Width, Height: winsize.Height}
}

// MonitorSize monitors the terminal's size. It returns a TerminalSizeQueue primed with
// initialSizes, or nil if there's no TTY present.
func (t *TTY) MonitorSize(initialSizes ...*TerminalSize) TerminalSizeQueue {
	outFd, isTerminal := term.GetFdInfo(t.Out)
	if !isTerminal {
		return nil
	}

	t.sizeQueue = &sizeQueue{
		t: *t,
		// make it buffered so we can send the initial terminal sizes without blocking, prior to starting
		// the streaming below
		resizeChan:   make(chan TerminalSize, len(initialSizes)),
		stopResizing: make(chan struct{}),
	}

	t.sizeQueue.monitorSize(outFd, initialSizes...)

	return t.sizeQueue
}

// sizeQueue implements remotecommand.TerminalSizeQueue
type sizeQueue struct {
	t TTY
	// resizeChan receives a Size each time the user's terminal is resized.
	resizeChan   chan TerminalSize
	stopResizing chan struct{}
}

// make sure sizeQueue implements the TerminalSizeQueue interface
var _ TerminalSizeQueue = &sizeQueue{}

// monitorSize primes resizeChan with initialSizes and then monitors for resize events. With each
// new event, it sends the current terminal size to resizeChan.
func (s *sizeQueue) monitorSize(outFd uintptr, initialSizes ...*TerminalSize) {
	// send the initial sizes
	for i := range initialSizes {
		if initialSizes[i] != nil {
			s.resizeChan <- *initialSizes[i]
		}
	}

	resizeEvents := make(chan TerminalSize, 1)

	monitorResizeEvents(outFd, resizeEvents, s.stopResizing)

	// listen for resize events in the background
	go func() {
		defer runtime.HandleCrash()

		for {
			select {
			case size, ok := <-resizeEvents:
				if !ok {
					return
				}

				select {
				// try to send the size to resizeChan, but don't block
				case s.resizeChan <- size:
					// send successful
				default:
					// unable to send / no-op
				}
			case <-s.stopResizing:
				return
			}
		}
	}()
}

// Next returns the new terminal size after the terminal has been resized. It returns nil when
// monitoring has been stopped.
func (s *sizeQueue) Next() *TerminalSize {
	size, ok := <-s.resizeChan
	if !ok {
		return nil
	}
	return &size
}

// stop stops the background goroutine that is monitoring for terminal resizes.
func (s *sizeQueue) stop() {
	close(s.stopResizing)
}
