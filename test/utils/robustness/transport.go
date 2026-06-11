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

package robustness

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// VerbWatch is the ClientFacts.Verb for watch streams. Watch rules are evaluated
// per event (Name/Namespace from the event's object); a BlockingFault there
// stalls the reflector, leaving the informer cache stale.
const VerbWatch = "WATCH"

// HTTPStatusError represents a structured API error status that should be converted
// into a valid HTTP response with the given status code and serializable metav1.Status payload.
type HTTPStatusError struct {
	StatusCode int
	Status     metav1.Status
}

func (e HTTPStatusError) Error() string {
	return fmt.Sprintf("HTTP Status Error %d: %s", e.StatusCode, e.Status.Message)
}

// ApplyTransport implements TransportFault by returning a synthetic API error response.
func (e HTTPStatusError) ApplyTransport() *HTTPStatusError {
	return &e
}

// NewHTTPStatusError creates an HTTPStatusError representing common API server failures.
func NewHTTPStatusError(statusCode int, reason metav1.StatusReason, message string) HTTPStatusError {
	return HTTPStatusError{
		StatusCode: statusCode,
		Status: metav1.Status{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Status",
				APIVersion: "v1",
			},
			Status:  metav1.StatusFailure,
			Message: message,
			Reason:  reason,
			Code:    int32(statusCode),
		},
	}
}

// FaultInjectingTransport wraps http.RoundTripper and triggers faults on matching REST API calls.
type FaultInjectingTransport struct {
	realTransport http.RoundTripper
	registry      *FaultRegistry
	activity      *activityTracker // may be nil; records mutating traffic for settle detection
	resolver      *request.RequestInfoFactory
}

// NewFaultInjectingTransport creates a wrapped http.RoundTripper that routes request
// hooks to the registry. activity may be nil when settle detection is not needed.
func NewFaultInjectingTransport(realTransport http.RoundTripper, registry *FaultRegistry, activity *activityTracker) http.RoundTripper {
	return &FaultInjectingTransport{
		realTransport: realTransport,
		registry:      registry,
		activity:      activity,
		resolver: &request.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
	}
}

func (t *FaultInjectingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	info, _ := t.resolver.NewRequestInfo(req)
	if info == nil || !info.IsResourceRequest {
		return t.realTransport.RoundTrip(req)
	}

	// Watches are faulted per event; they are already open when rules register.
	if info.Verb == "watch" {
		return t.roundTripWatch(req, info)
	}

	switch req.Method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		t.activity.recordMutation()
	}

	if httpErr := t.registry.ResolveTransport(req.Context(), ClientFacts{
		Verb:        req.Method,
		Group:       info.APIGroup,
		Resource:    info.Resource,
		Subresource: info.Subresource,
		Namespace:   info.Namespace,
		Name:        info.Name,
	}); httpErr != nil {
		if req.Body != nil {
			_ = req.Body.Close()
		}
		return synthesizeResponse(*httpErr)
	}

	return t.realTransport.RoundTrip(req)
}

func (t *FaultInjectingTransport) roundTripWatch(req *http.Request, info *request.RequestInfo) (*http.Response, error) {
	resp, err := t.realTransport.RoundTrip(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		return resp, err
	}
	// Only newline-delimited JSON streams are intercepted (the fixture pins JSON).
	if !strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		return resp, nil
	}
	resp.Body = &watchEventReader{
		ctx:      req.Context(),
		body:     resp.Body,
		reader:   bufio.NewReader(resp.Body),
		registry: t.registry,
		facts: ClientFacts{
			Verb:        VerbWatch,
			Group:       info.APIGroup,
			Resource:    info.Resource,
			Subresource: info.Subresource,
		},
	}
	return resp, nil
}

// watchEventReader resolves the registry for each JSON watch event before
// passing the bytes through unchanged.
type watchEventReader struct {
	ctx      context.Context
	body     io.ReadCloser
	reader   *bufio.Reader
	buffer   bytes.Buffer
	registry *FaultRegistry
	facts    ClientFacts // per-stream facts; Name and Namespace are filled per event
}

func (r *watchEventReader) Read(p []byte) (int, error) {
	if r.buffer.Len() > 0 {
		return r.buffer.Read(p)
	}
	line, err := r.reader.ReadBytes('\n')
	if len(line) > 0 {
		r.onEvent(line)
		r.buffer.Write(line)
		n, _ := r.buffer.Read(p)
		return n, nil
	}
	return 0, err
}

func (r *watchEventReader) onEvent(line []byte) {
	var event metav1.WatchEvent
	if json.Unmarshal(line, &event) != nil {
		return
	}
	// Bookmarks and errors carry no object name; they must not consume
	// single-shot conditions on wildcard-name rules.
	if event.Type == string(watch.Bookmark) || event.Type == string(watch.Error) {
		return
	}
	var meta metav1.PartialObjectMetadata
	if json.Unmarshal(event.Object.Raw, &meta) != nil {
		return
	}
	facts := r.facts
	facts.Name = meta.Name
	facts.Namespace = meta.Namespace
	r.registry.ResolveWatchEvent(r.ctx, facts)
}

func (r *watchEventReader) Close() error {
	return r.body.Close()
}

func synthesizeResponse(httpErr HTTPStatusError) (*http.Response, error) {
	status := httpErr.Status
	if status.Kind == "" {
		status.Kind = "Status"
	}
	if status.APIVersion == "" {
		status.APIVersion = "v1"
	}
	bodyBytes, jsonErr := json.Marshal(status)
	if jsonErr != nil {
		return nil, fmt.Errorf("failed to marshal injected status error: %w", jsonErr)
	}

	// Set Content-Type so client-go decodes the full metav1.Status payload
	// (Message, Details, RetryAfterSeconds) into *apierrors.StatusError.
	header := make(http.Header)
	header.Set("Content-Type", "application/json")

	return &http.Response{
		StatusCode:    httpErr.StatusCode,
		Status:        fmt.Sprintf("%d %s", httpErr.StatusCode, http.StatusText(httpErr.StatusCode)),
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Body:          io.NopCloser(bytes.NewReader(bodyBytes)),
		ContentLength: int64(len(bodyBytes)),
		Header:        header,
	}, nil
}
