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

package utils

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// InterceptHook is a callback executed when a REST request matches an InterceptionRule.
// For watch requests, the hook is executed for each individual chunked JSON event line
// in the response stream, and eventBytes contains the raw JSON bytes.
// For standard requests, the hook is executed before returning the response, and eventBytes is nil.
type InterceptHook func(req *http.Request, eventBytes []byte)

// InterceptionRule defines structured matching criteria for REST API requests.
type InterceptionRule struct {
	// Method represents the HTTP verb (e.g., "GET", "POST"), or "*" for any method.
	Method string
	// Group represents the targeted resource API group, e.g., "coordination.k8s.io", or empty for the core group, or "*" for any.
	Group string
	// Resource represents the targeted API resource, e.g., "leases", or "*" for any.
	Resource string
	// Namespace represents the targeted namespace, e.g., "kube-node-lease", or "*" for any.
	Namespace string
	// Name represents the target resource instance name, e.g., "di-target-node", or "*" for any.
	Name string
	// IsWatch indicates whether this rule applies only to watch streams.
	IsWatch bool
	// Hook is the callback executed when all matching criteria are met.
	Hook InterceptHook
}

// Matches returns true if the incoming request fits all rule parameters.
func (r *InterceptionRule) Matches(req *http.Request, info *request.RequestInfo) bool {
	if r.Method != "*" && r.Method != req.Method {
		return false
	}
	isWatch := req.URL.Query().Get("watch") == "true"
	if r.IsWatch != isWatch {
		return false
	}
	if info == nil || !info.IsResourceRequest {
		return false
	}
	if r.Group != "*" && r.Group != info.APIGroup {
		return false
	}
	if r.Resource != "*" && r.Resource != info.Resource {
		return false
	}
	if r.Namespace != "*" && r.Namespace != info.Namespace {
		return false
	}
	if r.Name != "*" && r.Name != info.Name {
		return false
	}
	return true
}

// interceptingReader wraps a chunked HTTP response body and executes rules hooks
// on matching line elements.
type interceptingReader struct {
	realBody io.ReadCloser
	reader   *bufio.Reader
	buffer   bytes.Buffer
	rules    []InterceptionRule
	req      *http.Request
	ctx      context.Context
}

func (r *interceptingReader) Read(p []byte) (int, error) {
	if r.buffer.Len() > 0 {
		return r.buffer.Read(p)
	}

	line, err := r.reader.ReadBytes('\n')
	if err != nil {
		return 0, err
	}

	var event metav1.WatchEvent
	if err := json.Unmarshal(line, &event); err == nil {
		var meta metav1.PartialObjectMetadata
		if err := json.Unmarshal(event.Object.Raw, &meta); err == nil {
			for _, rule := range r.rules {
				if rule.Name == "*" || rule.Name == meta.Name {
					if rule.Namespace == "*" || rule.Namespace == meta.Namespace {
						if rule.Hook != nil {
							rule.Hook(r.req, line)
						}
					}
				}
			}
		}
	}

	r.buffer.Write(line)
	return r.buffer.Read(p)
}

func (r *interceptingReader) Close() error {
	return r.realBody.Close()
}

// InterceptingTransport wraps http.RoundTripper and applies InterceptionRules.
type InterceptingTransport struct {
	realTransport http.RoundTripper
	rules         []InterceptionRule
	resolver      *request.RequestInfoFactory
}

func (t *InterceptingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	info, _ := t.resolver.NewRequestInfo(req)
	isWatch := req.URL.Query().Get("watch") == "true"

	if !isWatch {
		for _, rule := range t.rules {
			if rule.Matches(req, info) {
				if rule.Hook != nil {
					rule.Hook(req, nil)
				}
			}
		}
		return t.realTransport.RoundTrip(req)
	}

	resp, err := t.realTransport.RoundTrip(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode == http.StatusOK {
		var activeRules []InterceptionRule
		for _, rule := range t.rules {
			if rule.Matches(req, info) {
				activeRules = append(activeRules, rule)
			}
		}

		if len(activeRules) > 0 {
			resp.Body = &interceptingReader{
				realBody: resp.Body,
				reader:   bufio.NewReader(resp.Body),
				rules:    activeRules,
				req:      req,
				ctx:      req.Context(),
			}
		}
	}

	return resp, nil
}

// NewInterceptingTransport creates a new REST HTTP client intercepting transport.
func NewInterceptingTransport(realTransport http.RoundTripper, rules []InterceptionRule) http.RoundTripper {
	return &InterceptingTransport{
		realTransport: realTransport,
		rules:         rules,
		resolver: &request.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
	}
}
