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

package handlers

import (
	"net/http"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

const (
	maxUserAgentLength      = 1024
	userAgentTruncateSuffix = "...TRUNCATED"
)

// lazyTruncatedUserAgent implements String() string and it will
// return user-agent which may be truncated.
type lazyTruncatedUserAgent struct {
	req *http.Request
}

func (lazy *lazyTruncatedUserAgent) String() string {
	ua := "unknown"
	if lazy.req != nil {
		ua = utilnet.GetHTTPClient(lazy.req)
		if len(ua) > maxUserAgentLength {
			ua = ua[:maxUserAgentLength] + userAgentTruncateSuffix
		}
	}
	return ua
}

// LazyClientIP implements String() string and it will
// calls GetClientIP() lazily only when required.
type lazyClientIP struct {
	req *http.Request
}

func (lazy *lazyClientIP) String() string {
	if lazy.req != nil {
		if ip := utilnet.GetClientIP(lazy.req); ip != nil {
			return ip.String()
		}
	}
	return "unknown"
}

// lazyAccept implements String() string and it will
// calls http.Request Header.Get() lazily only when required.
type lazyAccept struct {
	req *http.Request
}

func (lazy *lazyAccept) String() string {
	if lazy.req != nil {
		accept := lazy.req.Header.Get("Accept")
		return accept
	}

	return "unknown"
}

// lazyAPIGroup implements String() string and it will
// lazily get Group from request info.
type lazyAPIGroup struct {
	req *http.Request
}

func (lazy *lazyAPIGroup) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return requestInfo.APIGroup
		}
	}

	return "unknown"
}

// lazyAPIVersion implements String() string and it will
// lazily get Group from request info.
type lazyAPIVersion struct {
	req *http.Request
}

func (lazy *lazyAPIVersion) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return requestInfo.APIVersion
		}
	}

	return "unknown"
}

// lazyName implements String() string and it will
// lazily get Group from request info.
type lazyName struct {
	req *http.Request
}

func (lazy *lazyName) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return requestInfo.Name
		}
	}

	return "unknown"
}

// lazySubresource implements String() string and it will
// lazily get Group from request info.
type lazySubresource struct {
	req *http.Request
}

func (lazy *lazySubresource) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return requestInfo.Subresource
		}
	}

	return "unknown"
}

// lazyNamespace implements String() string and it will
// lazily get Group from request info.
type lazyNamespace struct {
	req *http.Request
}

func (lazy *lazyNamespace) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return requestInfo.Namespace
		}
	}

	return "unknown"
}

// lazyAuditID implements Stringer interface to lazily retrieve
// the audit ID associated with the request.
type lazyAuditID struct {
	req *http.Request
}

func (lazy *lazyAuditID) String() string {
	if lazy.req != nil {
		return audit.GetAuditIDTruncated(lazy.req.Context())
	}

	return "unknown"
}

// lazyVerb implements String() string and it will
// lazily get normalized Verb
type lazyVerb struct {
	req *http.Request
}

func (lazy *lazyVerb) String() string {
	if lazy.req == nil {
		return "unknown"
	}
	return metrics.NormalizedVerb(lazy.req)
}

// lazyResource implements String() string and it will
// lazily get Resource from request info
type lazyResource struct {
	req *http.Request
}

func (lazy *lazyResource) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return requestInfo.Resource
		}
	}

	return "unknown"
}

// lazyScope implements String() string and it will
// lazily get Scope from request info
type lazyScope struct {
	req *http.Request
}

func (lazy *lazyScope) String() string {
	if lazy.req != nil {
		ctx := lazy.req.Context()
		requestInfo, ok := apirequest.RequestInfoFrom(ctx)
		if ok {
			return metrics.CleanScope(requestInfo)
		}
	}

	return "unknown"
}
