/*
Copyright 2022 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"net/url"
	"strings"
)

const (
	// unixProtocol is the only supported protocol for remote KMS provider.
	unixProtocol = "unix"
)

// Parse the endpoint to extract schema, host or path.
func ParseEndpoint(endpoint string) (string, error) {
	if len(endpoint) == 0 {
		return "", fmt.Errorf("remote KMS provider can't use empty string as endpoint")
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return "", fmt.Errorf("invalid endpoint %q for remote KMS provider, error: %v", endpoint, err)
	}

	if u.Scheme != unixProtocol {
		return "", fmt.Errorf("unsupported scheme %q for remote KMS provider", u.Scheme)
	}

	// Linux abstract namespace socket - no physical file required
	// Warning: Linux Abstract sockets have not concept of ACL (unlike traditional file based sockets).
	// However, Linux Abstract sockets are subject to Linux networking namespace, so will only be accessible to
	// containers within the same pod (unless host networking is used).
	if strings.HasPrefix(u.Path, "/@") {
		return strings.TrimPrefix(u.Path, "/"), nil
	}

	return u.Path, nil
}
