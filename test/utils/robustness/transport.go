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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// HTTPStatusError represents a structured API error status that should be converted
// into a valid HTTP response with the given status code and serializable metav1.Status payload.
type HTTPStatusError struct {
	StatusCode int
	Status     metav1.Status
}

func (e HTTPStatusError) Error() string {
	return fmt.Sprintf("HTTP Status Error %d: %s", e.StatusCode, e.Status.Message)
}

// ApplyTransport implements TransportFault: the request is answered with a
// synthetic API error response instead of being forwarded.
func (e HTTPStatusError) ApplyTransport() TransportVerdict {
	return TransportVerdict{Respond: &e}
}

// ConnectionError is a TransportFault that makes RoundTrip fail at the transport
// level (nil response), simulating a refused or dropped connection.
type ConnectionError struct {
	Err error
}

func (e ConnectionError) Error() string { return e.Err.Error() }

func (e ConnectionError) ApplyTransport() TransportVerdict {
	return TransportVerdict{ConnErr: e.Err}
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

	// Record mutating traffic (the attempt, before any injected failure) so the
	// fixture can detect when the controller has stopped writing (it has settled).
	switch req.Method {
	case http.MethodPost, http.MethodPut, http.MethodPatch, http.MethodDelete:
		t.activity.recordMutation()
	}

	v := t.registry.ResolveTransport(req.Context(), ClientFacts{
		Verb:        req.Method,
		Group:       info.APIGroup,
		Resource:    info.Resource,
		Subresource: info.Subresource,
		Namespace:   info.Namespace,
		Name:        info.Name,
	})
	switch {
	case v.Respond != nil:
		return synthesizeResponse(*v.Respond)
	case v.ConnErr != nil:
		return nil, v.ConnErr
	}

	return t.realTransport.RoundTrip(req)
}

// synthesizeResponse builds an *http.Response carrying a serialized metav1.Status,
// as the real apiserver would return for an error.
func synthesizeResponse(httpErr HTTPStatusError) (*http.Response, error) {
	// Ensure the payload is a well-formed Status object. client-go decodes error
	// bodies (using the response Content-Type) into the returned
	// *apierrors.StatusError. If TypeMeta is missing the body may not round-trip,
	// so backfill it.
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

	// Set Content-Type so client-go decodes the Status body. Note client-go
	// already maps the HTTP status code to a Reason (409 -> Conflict) when it
	// can't decode the body, so IsConflict/IsTooManyRequests still work without
	// this. What the header buys is the actual Status *payload* — Message and
	// Details (e.g. Details.RetryAfterSeconds, used for client backoff via
	// apierrors.SuggestsClientDelay) — so the injected fault behaves like a real
	// apiserver response rather than a generic stub.
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
