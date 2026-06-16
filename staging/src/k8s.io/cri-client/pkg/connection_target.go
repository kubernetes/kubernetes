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

package cri

import "strings"

func clientTargetForAddress(addr string) string {
	// Defensive: normalize callers that pass a unix endpoint instead of the
	// parsed socket path.
	if strings.HasPrefix(addr, "unix:///") {
		addr = strings.TrimPrefix(addr, "unix://")
	}

	// grpc defaults to the DNS resolver for bare targets. Use the passthrough
	// resolver for socket-style addresses so the custom dialer gets the raw path.
	if strings.HasPrefix(addr, "/") {
		return "passthrough:///" + addr
	}
	return addr
}
