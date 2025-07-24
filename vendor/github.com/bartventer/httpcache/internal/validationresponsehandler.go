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

type ValidationResponseHandler interface {
	HandleValidationResponse(
		ctx RevalidationContext,
		req *http.Request,
		resp *http.Response,
		err error,
	) (*http.Response, error)
}

type RevalidationContext struct {
	URLKey     string
	Start, End time.Time
	CCReq      CCRequestDirectives
	Stored     *Response
	Freshness  *Freshness
	Refs       ResponseRefs
	RefIndex   int
}

func (r RevalidationContext) ToMisc(ccResp CCResponseDirectives) MiscFunc {
	return MiscFunc(func() Misc {
		return Misc{
			CCReq:     r.CCReq,
			CCResp:    ccResp,
			Stored:    r.Stored,
			Freshness: r.Freshness,
			Refs:      r.Refs,
			RefIndex:  r.RefIndex,
		}
	})
}

type validationResponseHandler struct {
	l     *Logger
	clock Clock
	ci    CacheInvalidator
	ce    CacheabilityEvaluator
	siep  StaleIfErrorPolicy
	rs    ResponseStorer
}

func NewValidationResponseHandler(
	dl *Logger,
	clock Clock,
	ci CacheInvalidator,
	ce CacheabilityEvaluator,
	siep StaleIfErrorPolicy,
	rs ResponseStorer,
) *validationResponseHandler {
	return &validationResponseHandler{dl, clock, ci, ce, siep, rs}
}

func (r *validationResponseHandler) HandleValidationResponse(
	ctx RevalidationContext,
	req *http.Request,
	resp *http.Response,
	err error,
) (*http.Response, error) {
	if err == nil && req.Method == http.MethodGet && resp.StatusCode == http.StatusNotModified {
		// RFC 9111 §4.3.3 Handling Validation Responses (304 Not Modified)
		// RFC 9111 §4.3.4 Freshening Stored Responses upon Validation
		updateStoredHeaders(ctx.Stored.Data, resp)
		CacheStatusRevalidated.ApplyTo(ctx.Stored.Data.Header)
		r.l.LogCacheRevalidated(req, ctx.URLKey, ctx.ToMisc(nil))
		return ctx.Stored.Data, nil
	}

	var (
		ccResp     CCResponseDirectives
		ccRespOnce bool
	)
	if (err != nil || isStaleErrorAllowed(resp.StatusCode)) && req.Method == http.MethodGet {
		ccResp = ParseCCResponseDirectives(resp.Header)
		ccRespOnce = true
		if r.siep.CanStaleOnError(ctx.Freshness, ccResp) {
			// RFC 9111 §4.2.4 Serving Stale Responses
			// RFC 9111 §4.3.3 Handling Validation Responses (5xx errors)
			SetAgeHeader(ctx.Stored.Data, r.clock, ctx.Freshness.Age)
			CacheStatusStale.ApplyTo(ctx.Stored.Data.Header)
			r.l.LogCacheStaleIfError(req, ctx.URLKey, ctx.ToMisc(ccResp))
			return ctx.Stored.Data, nil
		}
	}

	if err != nil {
		return nil, err
	}

	if !ccRespOnce {
		ccResp = ParseCCResponseDirectives(resp.Header)
	}
	switch {
	case r.ce.CanStoreResponse(resp, ctx.CCReq, ccResp):
		// RFC 9111 §4.3.3 Handling Validation Responses (full response)
		// RFC 9111 §3.2 Storing Responses
		_ = r.rs.StoreResponse(req, resp, ctx.URLKey, ctx.Refs, ctx.Start, ctx.End, ctx.RefIndex)
		CacheStatusMiss.ApplyTo(resp.Header)
		r.l.LogCacheMiss(req, ctx.URLKey, ctx.ToMisc(ccResp))
	case IsUnsafeMethod(req.Method) && IsNonErrorStatus(resp.StatusCode):
		// RFC 9111 §4.4 Invalidation of Cache Entries
		r.ci.InvalidateCache(req.URL, resp.Header, ctx.Refs, ctx.URLKey)
		fallthrough
	default:
		CacheStatusBypass.ApplyTo(resp.Header)
		r.l.LogCacheBypass("Bypass; serving upstream response", req, ctx.URLKey, ctx.ToMisc(ccResp))
	}
	return resp, nil
}
