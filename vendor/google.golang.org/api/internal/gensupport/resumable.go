// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"google.golang.org/api/internal"
)

// ResumableUpload is used by the generated APIs to provide resumable uploads.
// It is not used by developers directly.
type ResumableUpload struct {
	Client *http.Client
	// URI is the resumable resource destination provided by the server after specifying "&uploadType=resumable".
	URI       string
	UserAgent string // User-Agent for header of the request
	// Media is the object being uploaded.
	Media *MediaBuffer
	// MediaType defines the media type, e.g. "image/jpeg".
	MediaType string

	mu       sync.Mutex // guards progress
	progress int64      // number of bytes uploaded so far

	// Callback is an optional function that will be periodically called with the cumulative number of bytes uploaded.
	Callback func(int64)

	// Retry optionally configures retries for requests made against the upload.
	Retry *RetryConfig

	// ChunkRetryDeadline configures the per-chunk deadline after which no further
	// retries should happen.
	ChunkRetryDeadline time.Duration

	// Track current request invocation ID and attempt count for retry metric
	// headers.
	invocationID string
	attempts     int
}

// Progress returns the number of bytes uploaded at this point.
func (rx *ResumableUpload) Progress() int64 {
	rx.mu.Lock()
	defer rx.mu.Unlock()
	return rx.progress
}

// doUploadRequest performs a single HTTP request to upload data.
// off specifies the offset in rx.Media from which data is drawn.
// size is the number of bytes in data.
// final specifies whether data is the final chunk to be uploaded.
func (rx *ResumableUpload) doUploadRequest(ctx context.Context, data io.Reader, off, size int64, final bool) (*http.Response, error) {
	req, err := http.NewRequest("POST", rx.URI, data)
	if err != nil {
		return nil, err
	}

	req.ContentLength = size
	var contentRange string
	if final {
		if size == 0 {
			contentRange = fmt.Sprintf("bytes */%v", off)
		} else {
			contentRange = fmt.Sprintf("bytes %v-%v/%v", off, off+size-1, off+size)
		}
	} else {
		contentRange = fmt.Sprintf("bytes %v-%v/*", off, off+size-1)
	}
	req.Header.Set("Content-Range", contentRange)
	req.Header.Set("Content-Type", rx.MediaType)
	req.Header.Set("User-Agent", rx.UserAgent)

	baseXGoogHeader := "gl-go/" + GoVersion() + " gdcl/" + internal.Version
	invocationHeader := fmt.Sprintf("gccl-invocation-id/%s gccl-attempt-count/%d", rx.invocationID, rx.attempts)
	req.Header.Set("X-Goog-Api-Client", strings.Join([]string{baseXGoogHeader, invocationHeader}, " "))

	// Google's upload endpoint uses status code 308 for a
	// different purpose than the "308 Permanent Redirect"
	// since-standardized in RFC 7238. Because of the conflict in
	// semantics, Google added this new request header which
	// causes it to not use "308" and instead reply with 200 OK
	// and sets the upload-specific "X-HTTP-Status-Code-Override:
	// 308" response header.
	req.Header.Set("X-GUploader-No-308", "yes")

	return SendRequest(ctx, rx.Client, req)
}

func statusResumeIncomplete(resp *http.Response) bool {
	// This is how the server signals "status resume incomplete"
	// when X-GUploader-No-308 is set to "yes":
	return resp != nil && resp.Header.Get("X-Http-Status-Code-Override") == "308"
}

// reportProgress calls a user-supplied callback to report upload progress.
// If old==updated, the callback is not called.
func (rx *ResumableUpload) reportProgress(old, updated int64) {
	if updated-old == 0 {
		return
	}
	rx.mu.Lock()
	rx.progress = updated
	rx.mu.Unlock()
	if rx.Callback != nil {
		rx.Callback(updated)
	}
}

// transferChunk performs a single HTTP request to upload a single chunk from rx.Media.
func (rx *ResumableUpload) transferChunk(ctx context.Context) (*http.Response, error) {
	chunk, off, size, err := rx.Media.Chunk()

	done := err == io.EOF
	if !done && err != nil {
		return nil, err
	}

	res, err := rx.doUploadRequest(ctx, chunk, off, int64(size), done)
	if err != nil {
		return res, err
	}

	// We sent "X-GUploader-No-308: yes" (see comment elsewhere in
	// this file), so we don't expect to get a 308.
	if res.StatusCode == 308 {
		return nil, errors.New("unexpected 308 response status code")
	}

	if res.StatusCode == http.StatusOK {
		rx.reportProgress(off, off+int64(size))
	}

	if statusResumeIncomplete(res) {
		rx.Media.Next()
	}
	return res, nil
}

// Upload starts the process of a resumable upload with a cancellable context.
// It retries using the provided back off strategy until cancelled or the
// strategy indicates to stop retrying.
// It is called from the auto-generated API code and is not visible to the user.
// Before sending an HTTP request, Upload calls any registered hook functions,
// and calls the returned functions after the request returns (see send.go).
// rx is private to the auto-generated API code.
// Exactly one of resp or err will be nil.  If resp is non-nil, the caller must call resp.Body.Close.
func (rx *ResumableUpload) Upload(ctx context.Context) (resp *http.Response, err error) {

	// There are a couple of cases where it's possible for err and resp to both
	// be non-nil. However, we expose a simpler contract to our callers: exactly
	// one of resp and err will be non-nil. This means that any response body
	// must be closed here before returning a non-nil error.
	var prepareReturn = func(resp *http.Response, err error) (*http.Response, error) {
		if err != nil {
			if resp != nil && resp.Body != nil {
				resp.Body.Close()
			}
			return nil, err
		}
		// This case is very unlikely but possible only if rx.ChunkRetryDeadline is
		// set to a very small value, in which case no requests will be sent before
		// the deadline. Return an error to avoid causing a panic.
		if resp == nil {
			return nil, fmt.Errorf("upload request to %v not sent, choose larger value for ChunkRetryDealine", rx.URI)
		}
		return resp, nil
	}
	// Configure retryable error criteria.
	errorFunc := rx.Retry.errorFunc()

	// Configure per-chunk retry deadline.
	var retryDeadline time.Duration
	if rx.ChunkRetryDeadline != 0 {
		retryDeadline = rx.ChunkRetryDeadline
	} else {
		retryDeadline = defaultRetryDeadline
	}

	// Send all chunks.
	for {
		var pause time.Duration

		// Each chunk gets its own initialized-at-zero backoff and invocation ID.
		bo := rx.Retry.backoff()
		quitAfter := time.After(retryDeadline)
		rx.attempts = 1
		rx.invocationID = uuid.New().String()

		// Retry loop for a single chunk.
		for {
			select {
			case <-ctx.Done():
				if err == nil {
					err = ctx.Err()
				}
				return prepareReturn(resp, err)
			case <-time.After(pause):
			case <-quitAfter:
				return prepareReturn(resp, err)
			}

			// Check for context cancellation or timeout once more. If more than one
			// case in the select statement above was satisfied at the same time, Go
			// will choose one arbitrarily.
			// That can cause an operation to go through even if the context was
			// canceled before or the timeout was reached.
			select {
			case <-ctx.Done():
				if err == nil {
					err = ctx.Err()
				}
				return prepareReturn(resp, err)
			case <-quitAfter:
				return prepareReturn(resp, err)
			default:
			}

			resp, err = rx.transferChunk(ctx)

			var status int
			if resp != nil {
				status = resp.StatusCode
			}

			// Check if we should retry the request.
			if !errorFunc(status, err) {
				break
			}

			rx.attempts++
			pause = bo.Pause()
			if resp != nil && resp.Body != nil {
				resp.Body.Close()
			}
		}

		// If the chunk was uploaded successfully, but there's still
		// more to go, upload the next chunk without any delay.
		if statusResumeIncomplete(resp) {
			resp.Body.Close()
			continue
		}

		return prepareReturn(resp, err)
	}
}
