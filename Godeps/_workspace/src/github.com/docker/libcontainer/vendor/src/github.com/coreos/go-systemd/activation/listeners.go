/*
Copyright 2014 CoreOS Inc.

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

package activation

import (
	"fmt"
	"net"
)

// Listeners returns net.Listeners for all socket activated fds passed to this process.
func Listeners(unsetEnv bool) ([]net.Listener, error) {
	files := Files(unsetEnv)
	listeners := make([]net.Listener, len(files))

	for i, f := range files {
		var err error
		listeners[i], err = net.FileListener(f)
		if err != nil {
			return nil, fmt.Errorf("Error setting up FileListener for fd %d: %s", f.Fd(), err.Error())
		}
	}

	return listeners, nil
}
