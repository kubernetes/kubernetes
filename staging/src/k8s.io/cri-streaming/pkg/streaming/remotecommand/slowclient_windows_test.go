//go:build windows

/*
Copyright The Kubernetes Authors.

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

package remotecommand

import "net"

// smallReceiveBufferDialer returns a plain dialer on Windows, where the tests
// still check that the fix delivers everything but cannot pin the socket
// buffers to make the old code fail reliably.
func smallReceiveBufferDialer() *net.Dialer {
	return &net.Dialer{}
}
