// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"golang.org/x/net/context"
)

const (
	// statusTooManyRequests is returned by the storage API if the
	// per-project limits have been temporarily exceeded. The request
	// should be retried.
	// https://cloud.google.com/storage/docs/json_api/v1/status-codes#standardcodes
	statusTooManyRequests = 429
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

	// If not specified, a default exponential backoff strategy will be used.
	Backoff BackoffStrategy
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

func contextDone(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
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
	var pause time.Duration
	backoff := rx.Backoff
	if backoff == nil {
		backoff = DefaultBackoffStrategy()
	}

	for {
		// Ensure that we return in the case of cancelled context, even if pause is 0.
		if contextDone(ctx) {
			return nil, ctx.Err()
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(pause):
		}

		resp, err = rx.transferChunk(ctx)

		var status int
		if resp != nil {
			status = resp.StatusCode
		}

		// Check if we should retry the request.
		if shouldRetry(status, err) {
			var retry bool
			pause, retry = backoff.Pause()
			if retry {
				if resp != nil && resp.Body != nil {
					resp.Body.Close()
				}
				continue
			}
		}

		// If the chunk was uploaded successfully, but there's still
		// more to go, upload the next chunk without any delay.
		if statusResumeIncomplete(resp) {
			pause = 0
			backoff.Reset()
			resp.Body.Close()
			continue
		}

		// It's possible for err and resp to both be non-nil here, but we expose a simpler
		// contract to our callers: exactly one of resp and err will be non-nil.  This means
		// that any response body must be closed here before returning a non-nil error.
		if err != nil {
			if resp != nil && resp.Body != nil {
				resp.Body.Close()
			}
			return nil, err
		}

		return resp, nil
	}
}
