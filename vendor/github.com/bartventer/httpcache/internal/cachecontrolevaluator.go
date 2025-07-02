// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"net/http"
	"time"
)

// CacheabilityEvaluator describes the interface implemented by types that can
// evaluate whether a response can be stored in cache, according to RFC 9111 §3.
type CacheabilityEvaluator interface {
	CanStoreResponse(
		resp *http.Response,
		reqCC CCRequestDirectives,
		resCC CCResponseDirectives,
	) bool
}

func canStoreResponse(
	resp *http.Response,
	reqCC CCRequestDirectives,
	resCC CCResponseDirectives,
) bool {
	//  The response status code must be final
	if resp.StatusCode < 200 || resp.StatusCode == http.StatusProcessing || resp.StatusCode >= 600 {
		return false
	}

	// If status is 206 or 304 or must-understand is present, cache must understand the code.
	if (resp.StatusCode == http.StatusPartialContent || resp.StatusCode == http.StatusNotModified || resCC.MustUnderstand()) &&
		!isStatusUnderstood(resp.StatusCode) {
		return false
	}

	// The no-store directive must not be present in the response/request headers.
	if resCC.NoStore() || reqCC.NoStore() {
		return false
	}

	// Skip the following checks:
	// 	- For private caches, private directive is allowed (no restriction).
	// 	- For private caches, Authorization is not a restriction.

	// The response must contain at least one explicit or heuristic cacheability indicator.
	return resCC.Public() ||
		resp.Header.Get("Expires") != "" ||
		resCC.MaxAgePresent() ||
		isHeuristicallyCacheableCode(resp.StatusCode)
}

type CacheabilityEvaluatorFunc func(
	resp *http.Response,
	reqCC CCRequestDirectives,
	resCC CCResponseDirectives,
) bool

func (f CacheabilityEvaluatorFunc) CanStoreResponse(
	resp *http.Response,
	reqCC CCRequestDirectives,
	resCC CCResponseDirectives,
) bool {
	return f(resp, reqCC, resCC)
}
func NewCacheabilityEvaluator() CacheabilityEvaluator {
	return CacheabilityEvaluatorFunc(canStoreResponse)
}

// StaleIfErrorPolicy describes the interface implemented by types that can
// evaluate cache control directives for storing responses (RFC 9111 §3) and
// determining whether a stale response can be served in case of an error (RFC 5861 §4).
type StaleIfErrorPolicy interface {
	CanStaleOnError(freshness *Freshness, sies ...StaleIfErrorer) bool
}

type StaleIfErrorer interface {
	// StaleIfError returns the duration for which the cache can serve a stale response
	// when an error occurs, according to the Stale-If-Error directive (RFC 5861 §4).
	StaleIfError() (dur time.Duration, valid bool)
}

var _ StaleIfErrorPolicy = (*staleIfErrorPolicy)(nil)

type staleIfErrorPolicy struct {
	clock Clock
}

func NewStaleIfErrorPolicy(clock Clock) *staleIfErrorPolicy {
	return &staleIfErrorPolicy{clock}
}

func (cce *staleIfErrorPolicy) CanStaleOnError(
	freshness *Freshness,
	sies ...StaleIfErrorer,
) bool {
	if len(sies) == 0 {
		return false
	}
	for _, sie := range sies {
		if sie == nil {
			continue
		}
		dur, valid := sie.StaleIfError()
		if !valid {
			continue
		}
		age := freshness.Age.Value + cce.clock.Since(freshness.Age.Timestamp)
		// If stale-if-error is set, allow extra staleness
		if age <= freshness.UsefulLife+dur {
			return true
		}
	}
	return false
}

// isStatusUnderstood reports whether the status code is understood by the cache.
func isStatusUnderstood(code int) bool {
	switch code {
	case http.StatusOK,
		http.StatusNonAuthoritativeInfo,
		// Range requests are not cacheable, so we don't consider 206 here.
		// http.StatusPartialContent,
		http.StatusMovedPermanently,
		http.StatusNotModified,
		http.StatusNotFound,
		http.StatusMethodNotAllowed,
		http.StatusGone,
		http.StatusRequestURITooLong,
		http.StatusNotImplemented,
		http.StatusPermanentRedirect:
		return true
	default:
		return false
	}
}

// isHeuristicallyCacheableCode reports whether the status code is heuristically cacheable
// per RFC9111 §4.2.2.
func isHeuristicallyCacheableCode(code int) bool {
	switch code {
	case http.StatusOK,
		http.StatusNonAuthoritativeInfo,
		http.StatusPartialContent,
		http.StatusMovedPermanently,
		http.StatusNotModified,
		http.StatusNotFound,
		http.StatusMethodNotAllowed,
		http.StatusGone,
		http.StatusRequestURITooLong,
		http.StatusNotImplemented,
		http.StatusPermanentRedirect:
		return true
	}
	return false
}
