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

// Package httpcache provides an implementation of http.RoundTripper that adds
// transparent HTTP response caching according to RFC 9111 (HTTP Caching).
//
// The main entry point is [NewTransport], which returns an [http.RoundTripper] for use with [http.Client].
// httpcache supports the required standard HTTP caching directives, as well as extension directives such as
// stale-while-revalidate, stale-if-error and immutable.
//
// Example usage:
//
//	package main
//
//	import (
//		"log/slog"
//		"net/http"
//		"time"
//
//		"github.com/bartventer/httpcache"
//
//		// Register a cache backend by importing the package
//		_ "github.com/bartventer/httpcache/store/fscache"
//	)
//
//	func main() {
//		dsn := "fscache://?appname=myapp" // Example DSN for the file system cache backend
//		client := &http.Client{
//			Transport: httpcache.NewTransport(
//				dsn,
//				httpcache.WithSWRTimeout(10*time.Second),
//				httpcache.WithLogger(slog.Default()),
//			),
//		}
//	}
package httpcache

import (
	"cmp"
	"context"
	"errors"
	"iter"
	"log/slog"
	"net/http"
	"time"

	"github.com/bartventer/httpcache/internal"
	"github.com/bartventer/httpcache/store"
	"github.com/bartventer/httpcache/store/driver"
)

const (
	DefaultSWRTimeout = 5 * time.Second

	CacheStatusHeader = internal.CacheStatusHeader
)

// transport is an implementation of [http.RoundTripper] that caches HTTP responses
// according to the HTTP caching rules defined in RFC 9111.
type transport struct {
	// Configurable options

	cache      internal.ResponseCache // Cache for storing and retrieving responses
	upstream   http.RoundTripper      // Underlying round tripper for upstream/origin requests
	swrTimeout time.Duration          // Timeout for Stale-While-Revalidate requests
	logger     *internal.Logger       // Logger for debug output, if needed

	// Internal details

	rmc   internal.RequestMethodChecker      // Checks if HTTP request methods are understood
	vm    internal.VaryMatcher               // Matches Vary headers to determine cache validity
	uk    internal.URLKeyer                  // Generates unique cache keys for requests
	fc    internal.FreshnessCalculator       // Calculates the freshness of cached responses
	ce    internal.CacheabilityEvaluator     // Evaluates if a response is cacheable
	siep  internal.StaleIfErrorPolicy        // Handles stale-if-error caching policies
	ci    internal.CacheInvalidator          // Invalidates cache entries based on conditions
	rs    internal.ResponseStorer            // Stores HTTP responses in the cache
	vrh   internal.ValidationResponseHandler // Processes validation responses for revalidation
	clock internal.Clock                     // Provides time-related operations, can be mocked for testing
}

// ErrOpenCache is used as the panic value when the cache cannot be opened.
// You may recover from this panic if you wish to handle the situation gracefully.
//
// Example usage:
//
//	defer func() {
//		if r := recover(); r != nil {
//			if err, ok := r.(error); ok && errors.Is(err, ErrOpenCache) {
//				// Handle the error gracefully, e.g., log it or return a default transport
//				log.Println("Failed to open cache:", err)
//				client := &http.Client{
//					Transport: http.DefaultTransport, // Fallback to default transport
//				}
//				// Use the fallback client as needed
//				_ = client
//			} else {
//				// Re-panic if it's not the expected error
//				panic(r)
//			}
//		}
//	}()
var ErrOpenCache = errors.New("httpcache: failed to open cache")

// NewTransport returns an [http.RoundTripper] that caches HTTP responses using
// the specified cache backend.
//
// The backend is selected via a DSN (e.g., "memcache://", "fscache://"), and
// should correlate to a registered cache driver in the [store] package.
// Panics with [ErrOpenCache] if the cache cannot be opened.
//
// To configure the transport, you can use functional options such as
// [WithUpstream], [WithSWRTimeout], and [WithLogger].
func NewTransport(dsn string, options ...Option) http.RoundTripper {
	cache, err := store.Open(dsn)
	if err != nil {
		panic(ErrOpenCache)
	}
	return newTransport(cache, options...)
}

func newTransport(conn driver.Conn, options ...Option) http.RoundTripper {
	rt := &transport{
		cache: internal.NewResponseCache(conn),
		rmc:   internal.NewRequestMethodChecker(),
		vm:    internal.NewVaryMatcher(internal.NewHeaderValueNormalizer()),
		uk:    internal.NewURLKeyer(),
		ce:    internal.NewCacheabilityEvaluator(),
		clock: internal.NewClock(),
	}

	for _, opt := range options {
		opt.apply(rt)
	}
	rt.upstream = cmp.Or(rt.upstream, http.DefaultTransport)
	rt.swrTimeout = cmp.Or(max(rt.swrTimeout, 0), DefaultSWRTimeout)
	rt.logger = cmp.Or(rt.logger, internal.NewLogger(slog.DiscardHandler))

	rt.fc = internal.NewFreshnessCalculator(rt.clock)
	rt.ci = internal.NewCacheInvalidator(rt.cache, rt.uk)
	rt.siep = internal.NewStaleIfErrorPolicy(rt.clock)
	rt.rs = internal.NewResponseStorer(
		rt.cache,
		internal.NewVaryHeaderNormalizer(),
		internal.NewVaryKeyer(),
	)
	rt.vrh = internal.NewValidationResponseHandler(
		rt.logger,
		rt.clock,
		rt.ci,
		rt.ce,
		rt.siep,
		rt.rs,
	)
	return rt
}

// NewClient returns a new [http.Client], configured with a transport that
// caches HTTP responses using the specified cache backend.
func NewClient(dsn string, options ...Option) *http.Client {
	return &http.Client{Transport: NewTransport(dsn, options...)}
}

var _ http.RoundTripper = (*transport)(nil)

func (r *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	urlKey := r.uk.URLKey(req.URL)

	if !r.rmc.IsRequestMethodUnderstood(req) {
		return r.handleUnrecognizedMethod(req, urlKey)
	}

	refs, err := r.cache.GetRefs(urlKey)
	if err != nil || len(refs) == 0 {
		return r.handleCacheMiss(req, urlKey, nil, -1)
	}

	refIndex, found := r.vm.VaryHeadersMatch(refs, req.Header)
	if !found {
		return r.handleCacheMiss(req, urlKey, refs, -1)
	}

	entry, err := r.cache.Get(refs[refIndex].ResponseID, req)
	if err != nil {
		r.logger.LogCacheError(
			"Error retrieving cache entry; possible corruption.",
			err,
			req,
			urlKey,
			internal.MiscFunc(func() internal.Misc {
				return internal.Misc{Refs: refs, RefIndex: refIndex}
			}),
		)
		return r.handleCacheMiss(req, urlKey, refs, refIndex)
	}

	return r.handleCacheHit(req, entry, urlKey, refs, refIndex)
}

func (r *transport) handleUnrecognizedMethod(
	req *http.Request,
	urlKey string,
) (*http.Response, error) {
	if !internal.IsUnsafeMethod(req.Method) {
		resp, err := r.upstream.RoundTrip(req)
		if err != nil {
			return nil, err
		}
		internal.CacheStatusBypass.ApplyTo(resp.Header)
		r.logger.LogCacheBypass(
			"Bypass; unrecognized (safe) method, served from upstream.",
			req,
			urlKey,
			nil,
		)
		return resp, nil
	}
	resp, err := r.upstream.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	if internal.IsNonErrorStatus(resp.StatusCode) {
		refs, _ := r.cache.GetRefs(urlKey)
		r.ci.InvalidateCache(req.URL, resp.Header, refs, urlKey)
	}
	internal.CacheStatusBypass.ApplyTo(resp.Header)
	r.logger.LogCacheBypass(
		"Bypass; unrecognized (unsafe) method, served from upstream.",
		req,
		urlKey,
		nil,
	)
	return resp, nil
}

func (r *transport) handleCacheMiss(
	req *http.Request,
	urlKey string,
	refs internal.ResponseRefs,
	refIndex int,
) (*http.Response, error) {
	ccReq := internal.ParseCCRequestDirectives(req.Header)
	if ccReq.OnlyIfCached() {
		r.logger.LogCacheMiss(
			req,
			urlKey,
			internal.MiscFunc(func() internal.Misc {
				return internal.Misc{
					CCReq:    ccReq,
					Refs:     refs,
					RefIndex: refIndex,
				}
			}),
		)
		return make504Response(req)
	}
	resp, start, end, err := r.roundTripTimed(req)
	if err != nil {
		return nil, err
	}
	ccResp := internal.ParseCCResponseDirectives(resp.Header)
	if r.ce.CanStoreResponse(resp, ccReq, ccResp) {
		_ = r.rs.StoreResponse(req, resp, urlKey, refs, start, end, refIndex)
	}
	internal.CacheStatusMiss.ApplyTo(resp.Header)
	r.logger.LogCacheMiss(req, urlKey, internal.MiscFunc(func() internal.Misc {
		return internal.Misc{
			CCReq:    ccReq,
			CCResp:   ccResp,
			Refs:     refs,
			RefIndex: refIndex,
		}
	}))
	return resp, nil
}

func (r *transport) handleCacheHit(
	req *http.Request,
	stored *internal.Response,
	urlKey string,
	refs internal.ResponseRefs,
	refIndex int,
) (*http.Response, error) {
	ccReq := internal.ParseCCRequestDirectives(req.Header)
	ccResp := internal.ParseCCResponseDirectives(stored.Data.Header)
	freshness := r.fc.CalculateFreshness(stored, ccReq, ccResp)
	respNoCacheFieldsRaw, hasRespNoCache := ccResp.NoCache()
	respNoCacheFieldsSeq, isRespNoCacheQualified := respNoCacheFieldsRaw.Value()

	// RFC 8246: If response is fresh and immutable, always serve from cache unless request has no-cache
	if !freshness.IsStale && ccResp.Immutable() && !ccReq.NoCache() {
		return r.serveFromCache(
			req,
			urlKey,
			stored,
			freshness,
			isRespNoCacheQualified,
			respNoCacheFieldsSeq,
		)
	}

	if (freshness.IsStale && ccResp.MustRevalidate()) ||
		(hasRespNoCache && !isRespNoCacheQualified) { // Unqualified no-cache: must revalidate before serving from cache
		goto revalidate
	}

	if ccReq.OnlyIfCached() || (!freshness.IsStale && !ccReq.NoCache()) {
		return r.serveFromCache(
			req,
			urlKey,
			stored,
			freshness,
			isRespNoCacheQualified,
			respNoCacheFieldsSeq,
		)
	}

	if swr, swrValid := ccResp.StaleWhileRevalidate(); freshness.IsStale && swrValid {
		age := freshness.Age.Value + r.clock.Since(freshness.Age.Timestamp)
		staleFor := age - freshness.UsefulLife
		if staleFor >= 0 && staleFor < swr {
			return r.handleStaleWhileRevalidate(req, stored, urlKey, freshness, ccReq)
		}
	}

revalidate:
	req = withConditionalHeaders(req, stored.Data.Header)
	resp, start, end, err := r.roundTripTimed(req)
	ctx := internal.RevalidationContext{
		URLKey:    urlKey,
		Start:     start,
		End:       end,
		CCReq:     ccReq,
		Stored:    stored,
		Refs:      refs,
		RefIndex:  refIndex,
		Freshness: freshness,
	}
	return r.vrh.HandleValidationResponse(ctx, req, resp, err)
}

func (r *transport) serveFromCache(
	req *http.Request,
	urlKey string,
	stored *internal.Response,
	freshness *internal.Freshness,
	noCacheQualified bool,
	noCacheFieldsSeq iter.Seq[string],
) (*http.Response, error) {
	if noCacheQualified {
		//Qualified no-cache: may serve from cache with fields stripped
		for field := range noCacheFieldsSeq {
			stored.Data.Header.Del(field)
		}
	}
	internal.SetAgeHeader(stored.Data, r.clock, freshness.Age)
	internal.CacheStatusHit.ApplyTo(stored.Data.Header)
	r.logger.LogCacheHit(req, urlKey, internal.MiscFunc(func() internal.Misc {
		return internal.Misc{
			Stored:    stored,
			Freshness: freshness,
		}
	}))
	return stored.Data, nil
}

// handleStaleWhileRevalidate serves a stale cached response immediately and triggers
// background revalidation in a separate goroutine (RFC 5861, §3).
func (r *transport) handleStaleWhileRevalidate(
	req *http.Request,
	stored *internal.Response,
	urlKey string,
	freshness *internal.Freshness,
	ccReq internal.CCRequestDirectives,
) (*http.Response, error) {
	req2 := req.Clone(req.Context())
	req2 = withConditionalHeaders(req2, stored.Data.Header)
	// Background revalidation is "best effort"; it is not guaranteed to complete
	// if the program exits before the goroutine finishes. This design choice was
	// made to keep the API simple and avoid requiring explicit shutdown coordination.
	//
	// Open a discussion at github.com/bartventer/httpcache/issues if your use case requires
	// guaranteed completion.
	go r.backgroundRevalidate(req2, stored, urlKey, freshness, ccReq)
	internal.CacheStatusStale.ApplyTo(stored.Data.Header)
	r.logger.LogCacheStaleRevalidate(req, urlKey, internal.MiscFunc(func() internal.Misc {
		return internal.Misc{
			CCReq:     ccReq,
			Stored:    stored,
			Freshness: freshness,
		}
	}))
	return stored.Data, nil
}

func (r *transport) backgroundRevalidate(
	req *http.Request,
	stored *internal.Response,
	urlKey string,
	freshness *internal.Freshness,
	ccReq internal.CCRequestDirectives,
) {
	ctx, cancel := context.WithTimeout(req.Context(), r.swrTimeout)
	defer cancel()
	req = req.WithContext(ctx)
	errc := make(chan error, 1)
	go func() {
		defer close(errc)
		//nolint:bodyclose // The response is not used, so we don't need to close it.
		resp, start, end, err := r.roundTripTimed(req)
		if err != nil {
			errc <- err
			return
		}
		select {
		case <-req.Context().Done():
			errc <- req.Context().Err()
			return
		default:
		}
		revalCtx := internal.RevalidationContext{
			URLKey:    urlKey,
			Start:     start,
			End:       end,
			CCReq:     ccReq,
			Stored:    stored,
			Freshness: freshness,
		}
		//nolint:bodyclose // The response is not used, so we don't need to close it.
		_, err = r.vrh.HandleValidationResponse(revalCtx, req, resp, nil)
		errc <- err
	}()

	select {
	case <-ctx.Done():
	case <-errc:
	}
}

func (r *transport) roundTripTimed(
	req *http.Request,
) (resp *http.Response, start, end time.Time, err error) {
	start = r.clock.Now()
	resp, err = r.upstream.RoundTrip(req)
	end = r.clock.Now()
	if resp != nil {
		_ = internal.FixDateHeader(resp.Header, end)
	}
	return
}
