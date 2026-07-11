// Copyright 2025 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package netutil

import (
	"net"
	"net/url"
	"strings"
)

// urlsHostNormalizedEqual compares two URLs for scheme, normalized host (including IPv6), and path equality.
func urlsHostNormalizedEqual(a, b url.URL) bool {
	return a.Scheme == b.Scheme &&
		normalizeHost(a.Host) == normalizeHost(b.Host) &&
		a.Path == b.Path
}

// normalizeHost returns the canonical string for the host and normalizes IPv6 and IPv4 addresses.
func normalizeHost(host string) string {
	hostOnly, port, err := net.SplitHostPort(host)
	if err != nil {
		hostOnly = host
		port = ""
	}

	// Check if hostOnly is an IPv6 address. It could be with or without brackets.
	ipStr := strings.Trim(hostOnly, "[]")
	if ip := net.ParseIP(ipStr); ip != nil {
		if ip.To4() == nil {
			// For IPv6 address, always use brackets when there is a port.
			return "[" + ip.String() + "]" + normalizePort(port)
		}
		// IPv4 address
		return ip.String() + normalizePort(port)
	}
	return host
}

func normalizePort(port string) string {
	if port == "" {
		return ""
	}
	return ":" + port
}
