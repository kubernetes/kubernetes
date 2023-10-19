// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strings"
	"sync"
	"time"

	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/api/googleapi"
)

type typeReader struct {
	io.Reader
	typ string
}

// multipartReader combines the contents of multiple readers to create a multipart/related HTTP body.
// Close must be called if reads from the multipartReader are abandoned before reaching EOF.
type multipartReader struct {
	pr       *io.PipeReader
	ctype    string
	mu       sync.Mutex
	pipeOpen bool
}

// boundary optionally specifies the MIME boundary
func newMultipartReader(parts []typeReader, boundary string) *multipartReader {
	mp := &multipartReader{pipeOpen: true}
	var pw *io.PipeWriter
	mp.pr, pw = io.Pipe()
	mpw := multipart.NewWriter(pw)
	if boundary != "" {
		mpw.SetBoundary(boundary)
	}
	mp.ctype = "multipart/related; boundary=" + mpw.Boundary()
	go func() {
		for _, part := range parts {
			w, err := mpw.CreatePart(typeHeader(part.typ))
			if err != nil {
				mpw.Close()
				pw.CloseWithError(fmt.Errorf("googleapi: CreatePart failed: %v", err))
				return
			}
			_, err = io.Copy(w, part.Reader)
			if err != nil {
				mpw.Close()
				pw.CloseWithError(fmt.Errorf("googleapi: Copy failed: %v", err))
				return
			}
		}

		mpw.Close()
		pw.Close()
	}()
	return mp
}

func (mp *multipartReader) Read(data []byte) (n int, err error) {
	return mp.pr.Read(data)
}

func (mp *multipartReader) Close() error {
	mp.mu.Lock()
	if !mp.pipeOpen {
		mp.mu.Unlock()
		return nil
	}
	mp.pipeOpen = false
	mp.mu.Unlock()
	return mp.pr.Close()
}

// CombineBodyMedia combines a json body with media content to create a multipart/related HTTP body.
// It returns a ReadCloser containing the combined body, and the overall "multipart/related" content type, with random boundary.
//
// The caller must call Close on the returned ReadCloser if reads are abandoned before reaching EOF.
func CombineBodyMedia(body io.Reader, bodyContentType string, media io.Reader, mediaContentType string) (io.ReadCloser, string) {
	return combineBodyMedia(body, bodyContentType, media, mediaContentType, "")
}

// combineBodyMedia is CombineBodyMedia but with an optional mimeBoundary field.
func combineBodyMedia(body io.Reader, bodyContentType string, media io.Reader, mediaContentType, mimeBoundary string) (io.ReadCloser, string) {
	mp := newMultipartReader([]typeReader{
		{body, bodyContentType},
		{media, mediaContentType},
	}, mimeBoundary)
	return mp, mp.ctype
}

func typeHeader(contentType string) textproto.MIMEHeader {
	h := make(textproto.MIMEHeader)
	if contentType != "" {
		h.Set("Content-Type", contentType)
	}
	return h
}

// PrepareUpload determines whether the data in the supplied reader should be
// uploaded in a single request, or in sequential chunks.
// chunkSize is the size of the chunk that media should be split into.
//
// If chunkSize is zero, media is returned as the first value, and the other
// two return values are nil, true.
//
// Otherwise, a MediaBuffer is returned, along with a bool indicating whether the
// contents of media fit in a single chunk.
//
// After PrepareUpload has been called, media should no longer be used: the
// media content should be accessed via one of the return values.
func PrepareUpload(media io.Reader, chunkSize int) (r io.Reader, mb *MediaBuffer, singleChunk bool) {
	if chunkSize == 0 { // do not chunk
		return media, nil, true
	}
	mb = NewMediaBuffer(media, chunkSize)
	_, _, _, err := mb.Chunk()
	// If err is io.EOF, we can upload this in a single request. Otherwise, err is
	// either nil or a non-EOF error. If it is the latter, then the next call to
	// mb.Chunk will return the same error. Returning a MediaBuffer ensures that this
	// error will be handled at some point.
	return nil, mb, err == io.EOF
}

// MediaInfo holds information for media uploads. It is intended for use by generated
// code only.
type MediaInfo struct {
	// At most one of Media and MediaBuffer will be set.
	media              io.Reader
	buffer             *MediaBuffer
	singleChunk        bool
	mType              string
	size               int64 // mediaSize, if known.  Used only for calls to progressUpdater_.
	progressUpdater    googleapi.ProgressUpdater
	chunkRetryDeadline time.Duration
}

// NewInfoFromMedia should be invoked from the Media method of a call. It returns a
// MediaInfo populated with chunk size and content type, and a reader or MediaBuffer
// if needed.
func NewInfoFromMedia(r io.Reader, options []googleapi.MediaOption) *MediaInfo {
	mi := &MediaInfo{}
	opts := googleapi.ProcessMediaOptions(options)
	if !opts.ForceEmptyContentType {
		mi.mType = opts.ContentType
		if mi.mType == "" {
			r, mi.mType = gax.DetermineContentType(r)
		}
	}
	mi.chunkRetryDeadline = opts.ChunkRetryDeadline
	mi.media, mi.buffer, mi.singleChunk = PrepareUpload(r, opts.ChunkSize)
	return mi
}

// NewInfoFromResumableMedia should be invoked from the ResumableMedia method of a
// call. It returns a MediaInfo using the given reader, size and media type.
func NewInfoFromResumableMedia(r io.ReaderAt, size int64, mediaType string) *MediaInfo {
	rdr := ReaderAtToReader(r, size)
	mType := mediaType
	if mType == "" {
		rdr, mType = gax.DetermineContentType(rdr)
	}

	return &MediaInfo{
		size:        size,
		mType:       mType,
		buffer:      NewMediaBuffer(rdr, googleapi.DefaultUploadChunkSize),
		media:       nil,
		singleChunk: false,
	}
}

// SetProgressUpdater sets the progress updater for the media info.
func (mi *MediaInfo) SetProgressUpdater(pu googleapi.ProgressUpdater) {
	if mi != nil {
		mi.progressUpdater = pu
	}
}

// UploadType determines the type of upload: a single request, or a resumable
// series of requests.
func (mi *MediaInfo) UploadType() string {
	if mi.singleChunk {
		return "multipart"
	}
	return "resumable"
}

// UploadRequest sets up an HTTP request for media upload. It adds headers
// as necessary, and returns a replacement for the body and a function for http.Request.GetBody.
func (mi *MediaInfo) UploadRequest(reqHeaders http.Header, body io.Reader) (newBody io.Reader, getBody func() (io.ReadCloser, error), cleanup func()) {
	cleanup = func() {}
	if mi == nil {
		return body, nil, cleanup
	}
	var media io.Reader
	if mi.media != nil {
		// This only happens when the caller has turned off chunking. In that
		// case, we write all of media in a single non-retryable request.
		media = mi.media
	} else if mi.singleChunk {
		// The data fits in a single chunk, which has now been read into the MediaBuffer.
		// We obtain that chunk so we can write it in a single request. The request can
		// be retried because the data is stored in the MediaBuffer.
		media, _, _, _ = mi.buffer.Chunk()
	}
	toCleanup := []io.Closer{}
	if media != nil {
		fb := readerFunc(body)
		fm := readerFunc(media)
		combined, ctype := CombineBodyMedia(body, "application/json", media, mi.mType)
		toCleanup = append(toCleanup, combined)
		if fb != nil && fm != nil {
			getBody = func() (io.ReadCloser, error) {
				rb := ioutil.NopCloser(fb())
				rm := ioutil.NopCloser(fm())
				var mimeBoundary string
				if _, params, err := mime.ParseMediaType(ctype); err == nil {
					mimeBoundary = params["boundary"]
				}
				r, _ := combineBodyMedia(rb, "application/json", rm, mi.mType, mimeBoundary)
				toCleanup = append(toCleanup, r)
				return r, nil
			}
		}
		reqHeaders.Set("Content-Type", ctype)
		body = combined
	}
	if mi.buffer != nil && mi.mType != "" && !mi.singleChunk {
		// This happens when initiating a resumable upload session.
		// The initial request contains a JSON body rather than media.
		// It can be retried with a getBody function that re-creates the request body.
		fb := readerFunc(body)
		if fb != nil {
			getBody = func() (io.ReadCloser, error) {
				rb := ioutil.NopCloser(fb())
				toCleanup = append(toCleanup, rb)
				return rb, nil
			}
		}
		reqHeaders.Set("X-Upload-Content-Type", mi.mType)
	}
	// Ensure that any bodies created in getBody are cleaned up.
	cleanup = func() {
		for _, closer := range toCleanup {
			_ = closer.Close()
		}

	}
	return body, getBody, cleanup
}

// readerFunc returns a function that always returns an io.Reader that has the same
// contents as r, provided that can be done without consuming r. Otherwise, it
// returns nil.
// See http.NewRequest (in net/http/request.go).
func readerFunc(r io.Reader) func() io.Reader {
	switch r := r.(type) {
	case *bytes.Buffer:
		buf := r.Bytes()
		return func() io.Reader { return bytes.NewReader(buf) }
	case *bytes.Reader:
		snapshot := *r
		return func() io.Reader { r := snapshot; return &r }
	case *strings.Reader:
		snapshot := *r
		return func() io.Reader { r := snapshot; return &r }
	default:
		return nil
	}
}

// ResumableUpload returns an appropriately configured ResumableUpload value if the
// upload is resumable, or nil otherwise.
func (mi *MediaInfo) ResumableUpload(locURI string) *ResumableUpload {
	if mi == nil || mi.singleChunk {
		return nil
	}
	return &ResumableUpload{
		URI:       locURI,
		Media:     mi.buffer,
		MediaType: mi.mType,
		Callback: func(curr int64) {
			if mi.progressUpdater != nil {
				mi.progressUpdater(curr, mi.size)
			}
		},
		ChunkRetryDeadline: mi.chunkRetryDeadline,
	}
}

// SetGetBody sets the GetBody field of req to f. This was once needed
// to gracefully support Go 1.7 and earlier which didn't have that
// field.
//
// Deprecated: the code generator no longer uses this as of
// 2019-02-19. Nothing else should be calling this anyway, but we
// won't delete this immediately; it will be deleted in as early as 6
// months.
func SetGetBody(req *http.Request, f func() (io.ReadCloser, error)) {
	req.GetBody = f
}
