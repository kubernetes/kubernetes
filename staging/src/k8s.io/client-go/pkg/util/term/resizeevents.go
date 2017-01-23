// +build !windows

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
	"os"
	"os/signal"
	"syscall"

	"k8s.io/apimachinery/pkg/util/runtime"
)

// monitorResizeEvents spawns a goroutine that waits for SIGWINCH signals (these indicate the
// terminal has resized). After receiving a SIGWINCH, this gets the terminal size and tries to send
// it to the resizeEvents channel. The goroutine stops when the stop channel is closed.
func monitorResizeEvents(fd uintptr, resizeEvents chan<- Size, stop chan struct{}) {
	go func() {
		defer runtime.HandleCrash()

		winch := make(chan os.Signal, 1)
		signal.Notify(winch, syscall.SIGWINCH)
		defer signal.Stop(winch)

		for {
			select {
			case <-winch:
				size := GetSize(fd)
				if size == nil {
					return
				}

				// try to send size
				select {
				case resizeEvents <- *size:
					// success
				default:
					// not sent
				}
			case <-stop:
				return
			}
		}
	}()
}
