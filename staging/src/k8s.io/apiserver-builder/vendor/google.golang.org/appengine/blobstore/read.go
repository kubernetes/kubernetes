// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package blobstore

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"

	blobpb "google.golang.org/appengine/internal/blobstore"
)

// openBlob returns a reader for a blob. It always succeeds; if the blob does
// not exist then an error will be reported upon first read.
func openBlob(c context.Context, blobKey appengine.BlobKey) Reader {
	return &reader{
		c:       c,
		blobKey: blobKey,
	}
}

const readBufferSize = 256 * 1024

// reader is a blob reader. It implements the Reader interface.
type reader struct {
	c context.Context

	// Either blobKey or filename is set:
	blobKey  appengine.BlobKey
	filename string

	closeFunc func() // is nil if unavailable or already closed.

	// buf is the read buffer. r is how much of buf has been read.
	// off is the offset of buf[0] relative to the start of the blob.
	// An invariant is 0 <= r && r <= len(buf).
	// Reads that don't require an RPC call will increment r but not off.
	// Seeks may modify r without discarding the buffer, but only if the
	// invariant can be maintained.
	mu  sync.Mutex
	buf []byte
	r   int
	off int64
}

func (r *reader) Close() error {
	if f := r.closeFunc; f != nil {
		f()
	}
	r.closeFunc = nil
	return nil
}

func (r *reader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.r == len(r.buf) {
		if err := r.fetch(r.off + int64(r.r)); err != nil {
			return 0, err
		}
	}
	n := copy(p, r.buf[r.r:])
	r.r += n
	return n, nil
}

func (r *reader) ReadAt(p []byte, off int64) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	// Convert relative offsets to absolute offsets.
	ab0 := r.off + int64(r.r)
	ab1 := r.off + int64(len(r.buf))
	ap0 := off
	ap1 := off + int64(len(p))
	// Check if we can satisfy the read entirely out of the existing buffer.
	if r.off <= ap0 && ap1 <= ab1 {
		// Convert off from an absolute offset to a relative offset.
		rp0 := int(ap0 - r.off)
		return copy(p, r.buf[rp0:]), nil
	}
	// Restore the original Read/Seek offset after ReadAt completes.
	defer r.seek(ab0)
	// Repeatedly fetch and copy until we have filled p.
	n := 0
	for len(p) > 0 {
		if err := r.fetch(off + int64(n)); err != nil {
			return n, err
		}
		r.r = copy(p, r.buf)
		n += r.r
		p = p[r.r:]
	}
	return n, nil
}

func (r *reader) Seek(offset int64, whence int) (ret int64, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch whence {
	case os.SEEK_SET:
		ret = offset
	case os.SEEK_CUR:
		ret = r.off + int64(r.r) + offset
	case os.SEEK_END:
		return 0, errors.New("seeking relative to the end of a blob isn't supported")
	default:
		return 0, fmt.Errorf("invalid Seek whence value: %d", whence)
	}
	if ret < 0 {
		return 0, errors.New("negative Seek offset")
	}
	return r.seek(ret)
}

// fetch fetches readBufferSize bytes starting at the given offset. On success,
// the data is saved as r.buf.
func (r *reader) fetch(off int64) error {
	req := &blobpb.FetchDataRequest{
		BlobKey:    proto.String(string(r.blobKey)),
		StartIndex: proto.Int64(off),
		EndIndex:   proto.Int64(off + readBufferSize - 1), // EndIndex is inclusive.
	}
	res := &blobpb.FetchDataResponse{}
	if err := internal.Call(r.c, "blobstore", "FetchData", req, res); err != nil {
		return err
	}
	if len(res.Data) == 0 {
		return io.EOF
	}
	r.buf, r.r, r.off = res.Data, 0, off
	return nil
}

// seek seeks to the given offset with an effective whence equal to SEEK_SET.
// It discards the read buffer if the invariant cannot be maintained.
func (r *reader) seek(off int64) (int64, error) {
	delta := off - r.off
	if delta >= 0 && delta < int64(len(r.buf)) {
		r.r = int(delta)
		return off, nil
	}
	r.buf, r.r, r.off = nil, 0, off
	return off, nil
}
