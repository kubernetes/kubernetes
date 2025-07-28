// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package registry

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"path"
	"strings"
	"sync"

	"github.com/google/go-containerregistry/internal/verify"
	v1 "github.com/google/go-containerregistry/pkg/v1"
)

// Returns whether this url should be handled by the blob handler
// This is complicated because blob is indicated by the trailing path, not the leading path.
// https://github.com/opencontainers/distribution-spec/blob/master/spec.md#pulling-a-layer
// https://github.com/opencontainers/distribution-spec/blob/master/spec.md#pushing-a-layer
func isBlob(req *http.Request) bool {
	elem := strings.Split(req.URL.Path, "/")
	elem = elem[1:]
	if elem[len(elem)-1] == "" {
		elem = elem[:len(elem)-1]
	}
	if len(elem) < 3 {
		return false
	}
	return elem[len(elem)-2] == "blobs" || (elem[len(elem)-3] == "blobs" &&
		elem[len(elem)-2] == "uploads")
}

// BlobHandler represents a minimal blob storage backend, capable of serving
// blob contents.
type BlobHandler interface {
	// Get gets the blob contents, or errNotFound if the blob wasn't found.
	Get(ctx context.Context, repo string, h v1.Hash) (io.ReadCloser, error)
}

// BlobStatHandler is an extension interface representing a blob storage
// backend that can serve metadata about blobs.
type BlobStatHandler interface {
	// Stat returns the size of the blob, or errNotFound if the blob wasn't
	// found, or redirectError if the blob can be found elsewhere.
	Stat(ctx context.Context, repo string, h v1.Hash) (int64, error)
}

// BlobPutHandler is an extension interface representing a blob storage backend
// that can write blob contents.
type BlobPutHandler interface {
	// Put puts the blob contents.
	//
	// The contents will be verified against the expected size and digest
	// as the contents are read, and an error will be returned if these
	// don't match. Implementations should return that error, or a wrapper
	// around that error, to return the correct error when these don't match.
	Put(ctx context.Context, repo string, h v1.Hash, rc io.ReadCloser) error
}

// BlobDeleteHandler is an extension interface representing a blob storage
// backend that can delete blob contents.
type BlobDeleteHandler interface {
	// Delete the blob contents.
	Delete(ctx context.Context, repo string, h v1.Hash) error
}

// redirectError represents a signal that the blob handler doesn't have the blob
// contents, but that those contents are at another location which registry
// clients should redirect to.
type redirectError struct {
	// Location is the location to find the contents.
	Location string

	// Code is the HTTP redirect status code to return to clients.
	Code int
}

type bytesCloser struct {
	*bytes.Reader
}

func (r *bytesCloser) Close() error {
	return nil
}

func (e redirectError) Error() string { return fmt.Sprintf("redirecting (%d): %s", e.Code, e.Location) }

// errNotFound represents an error locating the blob.
var errNotFound = errors.New("not found")

type memHandler struct {
	m    map[string][]byte
	lock sync.Mutex
}

func NewInMemoryBlobHandler() BlobHandler { return &memHandler{m: map[string][]byte{}} }

func (m *memHandler) Stat(_ context.Context, _ string, h v1.Hash) (int64, error) {
	m.lock.Lock()
	defer m.lock.Unlock()

	b, found := m.m[h.String()]
	if !found {
		return 0, errNotFound
	}
	return int64(len(b)), nil
}

func (m *memHandler) Get(_ context.Context, _ string, h v1.Hash) (io.ReadCloser, error) {
	m.lock.Lock()
	defer m.lock.Unlock()

	b, found := m.m[h.String()]
	if !found {
		return nil, errNotFound
	}
	return &bytesCloser{bytes.NewReader(b)}, nil
}

func (m *memHandler) Put(_ context.Context, _ string, h v1.Hash, rc io.ReadCloser) error {
	m.lock.Lock()
	defer m.lock.Unlock()

	defer rc.Close()
	all, err := io.ReadAll(rc)
	if err != nil {
		return err
	}
	m.m[h.String()] = all
	return nil
}

func (m *memHandler) Delete(_ context.Context, _ string, h v1.Hash) error {
	m.lock.Lock()
	defer m.lock.Unlock()

	if _, found := m.m[h.String()]; !found {
		return errNotFound
	}

	delete(m.m, h.String())
	return nil
}

// blobs
type blobs struct {
	blobHandler BlobHandler

	// Each upload gets a unique id that writes occur to until finalized.
	uploads map[string][]byte
	lock    sync.Mutex
	log     *log.Logger
}

func (b *blobs) handle(resp http.ResponseWriter, req *http.Request) *regError {
	elem := strings.Split(req.URL.Path, "/")
	elem = elem[1:]
	if elem[len(elem)-1] == "" {
		elem = elem[:len(elem)-1]
	}
	// Must have a path of form /v2/{name}/blobs/{upload,sha256:}
	if len(elem) < 4 {
		return &regError{
			Status:  http.StatusBadRequest,
			Code:    "NAME_INVALID",
			Message: "blobs must be attached to a repo",
		}
	}
	target := elem[len(elem)-1]
	service := elem[len(elem)-2]
	digest := req.URL.Query().Get("digest")
	contentRange := req.Header.Get("Content-Range")
	rangeHeader := req.Header.Get("Range")

	repo := req.URL.Host + path.Join(elem[1:len(elem)-2]...)

	switch req.Method {
	case http.MethodHead:
		h, err := v1.NewHash(target)
		if err != nil {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "NAME_INVALID",
				Message: "invalid digest",
			}
		}

		var size int64
		if bsh, ok := b.blobHandler.(BlobStatHandler); ok {
			size, err = bsh.Stat(req.Context(), repo, h)
			if errors.Is(err, errNotFound) {
				return regErrBlobUnknown
			} else if err != nil {
				var rerr redirectError
				if errors.As(err, &rerr) {
					http.Redirect(resp, req, rerr.Location, rerr.Code)
					return nil
				}
				return regErrInternal(err)
			}
		} else {
			rc, err := b.blobHandler.Get(req.Context(), repo, h)
			if errors.Is(err, errNotFound) {
				return regErrBlobUnknown
			} else if err != nil {
				var rerr redirectError
				if errors.As(err, &rerr) {
					http.Redirect(resp, req, rerr.Location, rerr.Code)
					return nil
				}
				return regErrInternal(err)
			}
			defer rc.Close()
			size, err = io.Copy(io.Discard, rc)
			if err != nil {
				return regErrInternal(err)
			}
		}

		resp.Header().Set("Content-Length", fmt.Sprint(size))
		resp.Header().Set("Docker-Content-Digest", h.String())
		resp.WriteHeader(http.StatusOK)
		return nil

	case http.MethodGet:
		h, err := v1.NewHash(target)
		if err != nil {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "NAME_INVALID",
				Message: "invalid digest",
			}
		}

		var size int64
		var r io.Reader
		if bsh, ok := b.blobHandler.(BlobStatHandler); ok {
			size, err = bsh.Stat(req.Context(), repo, h)
			if errors.Is(err, errNotFound) {
				return regErrBlobUnknown
			} else if err != nil {
				var rerr redirectError
				if errors.As(err, &rerr) {
					http.Redirect(resp, req, rerr.Location, rerr.Code)
					return nil
				}
				return regErrInternal(err)
			}

			rc, err := b.blobHandler.Get(req.Context(), repo, h)
			if errors.Is(err, errNotFound) {
				return regErrBlobUnknown
			} else if err != nil {
				var rerr redirectError
				if errors.As(err, &rerr) {
					http.Redirect(resp, req, rerr.Location, rerr.Code)
					return nil
				}

				return regErrInternal(err)
			}

			defer rc.Close()
			r = rc

		} else {
			tmp, err := b.blobHandler.Get(req.Context(), repo, h)
			if errors.Is(err, errNotFound) {
				return regErrBlobUnknown
			} else if err != nil {
				var rerr redirectError
				if errors.As(err, &rerr) {
					http.Redirect(resp, req, rerr.Location, rerr.Code)
					return nil
				}

				return regErrInternal(err)
			}
			defer tmp.Close()
			var buf bytes.Buffer
			io.Copy(&buf, tmp)
			size = int64(buf.Len())
			r = &buf
		}

		if rangeHeader != "" {
			start, end := int64(0), int64(0)
			if _, err := fmt.Sscanf(rangeHeader, "bytes=%d-%d", &start, &end); err != nil {
				return &regError{
					Status:  http.StatusRequestedRangeNotSatisfiable,
					Code:    "BLOB_UNKNOWN",
					Message: "We don't understand your Range",
				}
			}

			n := (end + 1) - start
			if ra, ok := r.(io.ReaderAt); ok {
				if end+1 > size {
					return &regError{
						Status:  http.StatusRequestedRangeNotSatisfiable,
						Code:    "BLOB_UNKNOWN",
						Message: fmt.Sprintf("range end %d > %d size", end+1, size),
					}
				}
				r = io.NewSectionReader(ra, start, n)
			} else {
				if _, err := io.CopyN(io.Discard, r, start); err != nil {
					return &regError{
						Status:  http.StatusRequestedRangeNotSatisfiable,
						Code:    "BLOB_UNKNOWN",
						Message: fmt.Sprintf("Failed to discard %d bytes", start),
					}
				}

				r = io.LimitReader(r, n)
			}

			resp.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
			resp.Header().Set("Content-Length", fmt.Sprint(n))
			resp.Header().Set("Docker-Content-Digest", h.String())
			resp.WriteHeader(http.StatusPartialContent)
		} else {
			resp.Header().Set("Content-Length", fmt.Sprint(size))
			resp.Header().Set("Docker-Content-Digest", h.String())
			resp.WriteHeader(http.StatusOK)
		}

		io.Copy(resp, r)
		return nil

	case http.MethodPost:
		bph, ok := b.blobHandler.(BlobPutHandler)
		if !ok {
			return regErrUnsupported
		}

		// It is weird that this is "target" instead of "service", but
		// that's how the index math works out above.
		if target != "uploads" {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "METHOD_UNKNOWN",
				Message: fmt.Sprintf("POST to /blobs must be followed by /uploads, got %s", target),
			}
		}

		if digest != "" {
			h, err := v1.NewHash(digest)
			if err != nil {
				return regErrDigestInvalid
			}

			vrc, err := verify.ReadCloser(req.Body, req.ContentLength, h)
			if err != nil {
				return regErrInternal(err)
			}
			defer vrc.Close()

			if err = bph.Put(req.Context(), repo, h, vrc); err != nil {
				if errors.As(err, &verify.Error{}) {
					log.Printf("Digest mismatch: %v", err)
					return regErrDigestMismatch
				}
				return regErrInternal(err)
			}
			resp.Header().Set("Docker-Content-Digest", h.String())
			resp.WriteHeader(http.StatusCreated)
			return nil
		}

		id := fmt.Sprint(rand.Int63())
		resp.Header().Set("Location", "/"+path.Join("v2", path.Join(elem[1:len(elem)-2]...), "blobs/uploads", id))
		resp.Header().Set("Range", "0-0")
		resp.WriteHeader(http.StatusAccepted)
		return nil

	case http.MethodPatch:
		if service != "uploads" {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "METHOD_UNKNOWN",
				Message: fmt.Sprintf("PATCH to /blobs must be followed by /uploads, got %s", service),
			}
		}

		if contentRange != "" {
			start, end := 0, 0
			if _, err := fmt.Sscanf(contentRange, "%d-%d", &start, &end); err != nil {
				return &regError{
					Status:  http.StatusRequestedRangeNotSatisfiable,
					Code:    "BLOB_UPLOAD_UNKNOWN",
					Message: "We don't understand your Content-Range",
				}
			}
			b.lock.Lock()
			defer b.lock.Unlock()
			if start != len(b.uploads[target]) {
				return &regError{
					Status:  http.StatusRequestedRangeNotSatisfiable,
					Code:    "BLOB_UPLOAD_UNKNOWN",
					Message: "Your content range doesn't match what we have",
				}
			}
			l := bytes.NewBuffer(b.uploads[target])
			io.Copy(l, req.Body)
			b.uploads[target] = l.Bytes()
			resp.Header().Set("Location", "/"+path.Join("v2", path.Join(elem[1:len(elem)-3]...), "blobs/uploads", target))
			resp.Header().Set("Range", fmt.Sprintf("0-%d", len(l.Bytes())-1))
			resp.WriteHeader(http.StatusNoContent)
			return nil
		}

		b.lock.Lock()
		defer b.lock.Unlock()
		if _, ok := b.uploads[target]; ok {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "BLOB_UPLOAD_INVALID",
				Message: "Stream uploads after first write are not allowed",
			}
		}

		l := &bytes.Buffer{}
		io.Copy(l, req.Body)

		b.uploads[target] = l.Bytes()
		resp.Header().Set("Location", "/"+path.Join("v2", path.Join(elem[1:len(elem)-3]...), "blobs/uploads", target))
		resp.Header().Set("Range", fmt.Sprintf("0-%d", len(l.Bytes())-1))
		resp.WriteHeader(http.StatusNoContent)
		return nil

	case http.MethodPut:
		bph, ok := b.blobHandler.(BlobPutHandler)
		if !ok {
			return regErrUnsupported
		}

		if service != "uploads" {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "METHOD_UNKNOWN",
				Message: fmt.Sprintf("PUT to /blobs must be followed by /uploads, got %s", service),
			}
		}

		if digest == "" {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "DIGEST_INVALID",
				Message: "digest not specified",
			}
		}

		b.lock.Lock()
		defer b.lock.Unlock()

		h, err := v1.NewHash(digest)
		if err != nil {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "NAME_INVALID",
				Message: "invalid digest",
			}
		}

		defer req.Body.Close()
		in := io.NopCloser(io.MultiReader(bytes.NewBuffer(b.uploads[target]), req.Body))

		size := int64(verify.SizeUnknown)
		if req.ContentLength > 0 {
			size = int64(len(b.uploads[target])) + req.ContentLength
		}

		vrc, err := verify.ReadCloser(in, size, h)
		if err != nil {
			return regErrInternal(err)
		}
		defer vrc.Close()

		if err := bph.Put(req.Context(), repo, h, vrc); err != nil {
			if errors.As(err, &verify.Error{}) {
				log.Printf("Digest mismatch: %v", err)
				return regErrDigestMismatch
			}
			return regErrInternal(err)
		}

		delete(b.uploads, target)
		resp.Header().Set("Docker-Content-Digest", h.String())
		resp.WriteHeader(http.StatusCreated)
		return nil

	case http.MethodDelete:
		bdh, ok := b.blobHandler.(BlobDeleteHandler)
		if !ok {
			return regErrUnsupported
		}

		h, err := v1.NewHash(target)
		if err != nil {
			return &regError{
				Status:  http.StatusBadRequest,
				Code:    "NAME_INVALID",
				Message: "invalid digest",
			}
		}
		if err := bdh.Delete(req.Context(), repo, h); err != nil {
			return regErrInternal(err)
		}
		resp.WriteHeader(http.StatusAccepted)
		return nil

	default:
		return &regError{
			Status:  http.StatusBadRequest,
			Code:    "METHOD_UNKNOWN",
			Message: "We don't understand your method + url",
		}
	}
}
