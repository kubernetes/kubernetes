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

package internal

import (
	"net/http"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

const (
	maxUserAgentLength      = 1024
	userAgentTruncateSuffix = "...TRUNCATED"
)

// LazyTruncatedUserAgent implements String() string and it will
// return user-agent which may be truncated.
type LazyTruncatedUserAgent struct {
	Req *http.Request
}

func (lazy *LazyTruncatedUserAgent) String() string {
	ua := "unknown"
	if lazy.Req != nil {
		ua = utilnet.GetHTTPClient(lazy.Req)
		if len(ua) > maxUserAgentLength {
			ua = ua[:maxUserAgentLength] + userAgentTruncateSuffix
		}
	}
	return ua
}

// LazyClientIP implements String() string and it will
// calls GetClientIP() lazily only when required.
type LazyClientIP struct {
	Req *http.Request
}

func (lazy *LazyClientIP) String() string {
	if lazy.Req != nil {
		if ip := utilnet.GetClientIP(lazy.Req); ip != nil {
			return ip.String()
		}
	}
	return "unknown"
}
