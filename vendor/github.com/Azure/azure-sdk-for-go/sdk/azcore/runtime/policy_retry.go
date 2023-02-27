//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"context"
	"errors"
	"io"
	"math"
	"math/rand"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/log"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/errorinfo"
)

const (
	defaultMaxRetries = 3
)

func setDefaults(o *policy.RetryOptions) {
	if o.MaxRetries == 0 {
		o.MaxRetries = defaultMaxRetries
	} else if o.MaxRetries < 0 {
		o.MaxRetries = 0
	}
	if o.MaxRetryDelay == 0 {
		o.MaxRetryDelay = 120 * time.Second
	} else if o.MaxRetryDelay < 0 {
		// not really an unlimited cap, but sufficiently large enough to be considered as such
		o.MaxRetryDelay = math.MaxInt64
	}
	if o.RetryDelay == 0 {
		o.RetryDelay = 4 * time.Second
	} else if o.RetryDelay < 0 {
		o.RetryDelay = 0
	}
	if o.StatusCodes == nil {
		o.StatusCodes = []int{
			http.StatusRequestTimeout,      // 408
			http.StatusTooManyRequests,     // 429
			http.StatusInternalServerError, // 500
			http.StatusBadGateway,          // 502
			http.StatusServiceUnavailable,  // 503
			http.StatusGatewayTimeout,      // 504
		}
	}
}

func calcDelay(o policy.RetryOptions, try int32) time.Duration { // try is >=1; never 0
	pow := func(number int64, exponent int32) int64 { // pow is nested helper function
		var result int64 = 1
		for n := int32(0); n < exponent; n++ {
			result *= number
		}
		return result
	}

	delay := time.Duration(pow(2, try)-1) * o.RetryDelay

	// Introduce some jitter:  [0.0, 1.0) / 2 = [0.0, 0.5) + 0.8 = [0.8, 1.3)
	delay = time.Duration(delay.Seconds() * (rand.Float64()/2 + 0.8) * float64(time.Second)) // NOTE: We want math/rand; not crypto/rand
	if delay > o.MaxRetryDelay {
		delay = o.MaxRetryDelay
	}
	return delay
}

// NewRetryPolicy creates a policy object configured using the specified options.
// Pass nil to accept the default values; this is the same as passing a zero-value options.
func NewRetryPolicy(o *policy.RetryOptions) policy.Policy {
	if o == nil {
		o = &policy.RetryOptions{}
	}
	p := &retryPolicy{options: *o}
	return p
}

type retryPolicy struct {
	options policy.RetryOptions
}

func (p *retryPolicy) Do(req *policy.Request) (resp *http.Response, err error) {
	options := p.options
	// check if the retry options have been overridden for this call
	if override := req.Raw().Context().Value(shared.CtxWithRetryOptionsKey{}); override != nil {
		options = override.(policy.RetryOptions)
	}
	setDefaults(&options)
	// Exponential retry algorithm: ((2 ^ attempt) - 1) * delay * random(0.8, 1.2)
	// When to retry: connection failure or temporary/timeout.
	var rwbody *retryableRequestBody
	if req.Body() != nil {
		// wrap the body so we control when it's actually closed.
		// do this outside the for loop so defers don't accumulate.
		rwbody = &retryableRequestBody{body: req.Body()}
		defer rwbody.realClose()
	}
	try := int32(1)
	for {
		resp = nil // reset
		log.Writef(log.EventRetryPolicy, "\n=====> Try=%d %s %s", try, req.Raw().Method, req.Raw().URL.String())

		// For each try, seek to the beginning of the Body stream. We do this even for the 1st try because
		// the stream may not be at offset 0 when we first get it and we want the same behavior for the
		// 1st try as for additional tries.
		err = req.RewindBody()
		if err != nil {
			return
		}
		// RewindBody() restores Raw().Body to its original state, so set our rewindable after
		if rwbody != nil {
			req.Raw().Body = rwbody
		}

		if options.TryTimeout == 0 {
			resp, err = req.Next()
		} else {
			// Set the per-try time for this particular retry operation and then Do the operation.
			tryCtx, tryCancel := context.WithTimeout(req.Raw().Context(), options.TryTimeout)
			clone := req.Clone(tryCtx)
			resp, err = clone.Next() // Make the request
			// if the body was already downloaded or there was an error it's safe to cancel the context now
			if err != nil {
				tryCancel()
			} else if _, ok := resp.Body.(*shared.NopClosingBytesReader); ok {
				tryCancel()
			} else {
				// must cancel the context after the body has been read and closed
				resp.Body = &contextCancelReadCloser{cf: tryCancel, body: resp.Body}
			}
		}
		if err == nil {
			log.Writef(log.EventRetryPolicy, "response %d", resp.StatusCode)
		} else {
			log.Writef(log.EventRetryPolicy, "error %v", err)
		}

		if err == nil && !HasStatusCode(resp, options.StatusCodes...) {
			// if there is no error and the response code isn't in the list of retry codes then we're done.
			return
		} else if ctxErr := req.Raw().Context().Err(); ctxErr != nil {
			// don't retry if the parent context has been cancelled or its deadline exceeded
			err = ctxErr
			log.Writef(log.EventRetryPolicy, "abort due to %v", err)
			return
		}

		// check if the error is not retriable
		var nre errorinfo.NonRetriable
		if errors.As(err, &nre) {
			// the error says it's not retriable so don't retry
			log.Writef(log.EventRetryPolicy, "non-retriable error %T", nre)
			return
		}

		if try == options.MaxRetries+1 {
			// max number of tries has been reached, don't sleep again
			log.Writef(log.EventRetryPolicy, "MaxRetries %d exceeded", options.MaxRetries)
			return
		}

		// drain before retrying so nothing is leaked
		Drain(resp)

		// use the delay from retry-after if available
		delay := shared.RetryAfter(resp)
		if delay <= 0 {
			delay = calcDelay(options, try)
		}
		log.Writef(log.EventRetryPolicy, "End Try #%d, Delay=%v", try, delay)
		select {
		case <-time.After(delay):
			try++
		case <-req.Raw().Context().Done():
			err = req.Raw().Context().Err()
			log.Writef(log.EventRetryPolicy, "abort due to %v", err)
			return
		}
	}
}

// WithRetryOptions adds the specified RetryOptions to the parent context.
// Use this to specify custom RetryOptions at the API-call level.
func WithRetryOptions(parent context.Context, options policy.RetryOptions) context.Context {
	return context.WithValue(parent, shared.CtxWithRetryOptionsKey{}, options)
}

// ********** The following type/methods implement the retryableRequestBody (a ReadSeekCloser)

// This struct is used when sending a body to the network
type retryableRequestBody struct {
	body io.ReadSeeker // Seeking is required to support retries
}

// Read reads a block of data from an inner stream and reports progress
func (b *retryableRequestBody) Read(p []byte) (n int, err error) {
	return b.body.Read(p)
}

func (b *retryableRequestBody) Seek(offset int64, whence int) (offsetFromStart int64, err error) {
	return b.body.Seek(offset, whence)
}

func (b *retryableRequestBody) Close() error {
	// We don't want the underlying transport to close the request body on transient failures so this is a nop.
	// The retry policy closes the request body upon success.
	return nil
}

func (b *retryableRequestBody) realClose() error {
	if c, ok := b.body.(io.Closer); ok {
		return c.Close()
	}
	return nil
}

// ********** The following type/methods implement the contextCancelReadCloser

// contextCancelReadCloser combines an io.ReadCloser with a cancel func.
// it ensures the cancel func is invoked once the body has been read and closed.
type contextCancelReadCloser struct {
	cf   context.CancelFunc
	body io.ReadCloser
}

func (rc *contextCancelReadCloser) Read(p []byte) (n int, err error) {
	return rc.body.Read(p)
}

func (rc *contextCancelReadCloser) Close() error {
	err := rc.body.Close()
	rc.cf()
	return err
}
