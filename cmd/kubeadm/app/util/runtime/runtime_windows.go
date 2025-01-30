//go:build windows
// +build windows

/*
Copyright 2019 The Kubernetes Authors.

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

package runtime

import (
	"net/url"

	winio "github.com/Microsoft/go-winio"
)

// isExistingSocket checks if path exists and is domain socket
func isExistingSocket(path string) bool {
	u, err := url.Parse(path)
	if err != nil {
		// should not happen, since we are trying to access known / hardcoded sockets
		return false
	}

	// the dial path must be without "npipe://"
	_, err = winio.DialPipe(u.Path, nil)
	if err != nil {
		return false
	}

	return true
}
