/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/rest"
)

type CounterMetric interface {
	Inc()
}

// LocationStreamer is a resource that streams the contents of a particular
// location URL.
type LocationStreamer struct {
	Location        *url.URL
	Transport       http.RoundTripper
	ContentType     string
	Flush           bool
	ResponseChecker HttpResponseChecker
	RedirectChecker func(req *http.Request, via []*http.Request) error
	// TLSVerificationErrorCounter is an optional value that will Inc every time a TLS error is encountered.  This can
	// be wired a single prometheus counter instance to get counts overall.
	TLSVerificationErrorCounter CounterMetric
	// DeprecatedTLSVerificationErrorCounter is a temporary field used to rename
	// the kube_apiserver_pod_logs_pods_logs_backend_tls_failure_total metric
	// with a one release deprecation period in 1.27.0.
	DeprecatedTLSVerificationErrorCounter CounterMetric
}

// a LocationStreamer must implement a rest.ResourceStreamer
var _ rest.ResourceStreamer = &LocationStreamer{}

func (obj *LocationStreamer) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}
func (obj *LocationStreamer) DeepCopyObject() runtime.Object {
	panic("rest.LocationStreamer does not implement DeepCopyObject")
}

// InputStream returns a stream with the contents of the URL location. If no location is provided,
// a null stream is returned.
func (s *LocationStreamer) InputStream(ctx context.Context, apiVersion, acceptHeader string) (stream io.ReadCloser, flush bool, contentType string, err error) {
	if s.Location == nil {
		// If no location was provided, return a null stream
		return nil, false, "", nil
	}
	transport := s.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}

	client := &http.Client{
		Transport:     transport,
		CheckRedirect: s.RedirectChecker,
	}
	req, err := http.NewRequest("GET", s.Location.String(), nil)
	if err != nil {
		return nil, false, "", fmt.Errorf("failed to construct request for %s, got %v", s.Location.String(), err)
	}
	// Pass the parent context down to the request to ensure that the resources
	// will be release properly.
	req = req.WithContext(ctx)

	resp, err := client.Do(req)
	if err != nil {
		// TODO prefer segregate TLS errors more reliably, but we do want to increment a count
		if strings.Contains(err.Error(), "x509:") && s.TLSVerificationErrorCounter != nil {
			s.TLSVerificationErrorCounter.Inc()
			if s.DeprecatedTLSVerificationErrorCounter != nil {
				s.DeprecatedTLSVerificationErrorCounter.Inc()
			}
		}
		return nil, false, "", err
	}

	if s.ResponseChecker != nil {
		if err = s.ResponseChecker.Check(resp); err != nil {
			return nil, false, "", err
		}
	}

	contentType = s.ContentType
	if len(contentType) == 0 {
		contentType = resp.Header.Get("Content-Type")
		if len(contentType) > 0 {
			contentType = strings.TrimSpace(strings.SplitN(contentType, ";", 2)[0])
		}
	}
	flush = s.Flush
	stream = resp.Body
	return
}

// PreventRedirects is a redirect checker that prevents the client from following a redirect.
func PreventRedirects(_ *http.Request, _ []*http.Request) error {
	return errors.New("redirects forbidden")
}
