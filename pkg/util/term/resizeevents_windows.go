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
	"time"

	"k8s.io/apimachinery/pkg/util/runtime"
)

// monitorResizeEvents spawns a goroutine that periodically gets the terminal size and tries to send
// it to the resizeEvents channel if the size has changed. The goroutine stops when the stop channel
// is closed.
func monitorResizeEvents(fd uintptr, resizeEvents chan<- Size, stop chan struct{}) {
	go func() {
		defer runtime.HandleCrash()

		size := GetSize(fd)
		if size == nil {
			return
		}
		lastSize := *size

		for {
			// see if we need to stop running
			select {
			case <-stop:
				return
			default:
			}

			size := GetSize(fd)
			if size == nil {
				return
			}

			if size.Height != lastSize.Height || size.Width != lastSize.Width {
				lastSize.Height = size.Height
				lastSize.Width = size.Width
				resizeEvents <- *size
			}

			// sleep to avoid hot looping
			time.Sleep(250 * time.Millisecond)
		}
	}()
}
