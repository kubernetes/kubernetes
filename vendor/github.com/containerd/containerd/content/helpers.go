package content

import (
	"context"
	"fmt"
	"io"
	"sync"

	"github.com/containerd/containerd/errdefs"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

var (
	bufPool = sync.Pool{
		New: func() interface{} {
			return make([]byte, 1<<20)
		},
	}
)

// NewReader returns a io.Reader from a ReaderAt
func NewReader(ra ReaderAt) io.Reader {
	rd := io.NewSectionReader(ra, 0, ra.Size())
	return rd
}

// ReadBlob retrieves the entire contents of the blob from the provider.
//
// Avoid using this for large blobs, such as layers.
func ReadBlob(ctx context.Context, provider Provider, dgst digest.Digest) ([]byte, error) {
	ra, err := provider.ReaderAt(ctx, dgst)
	if err != nil {
		return nil, err
	}
	defer ra.Close()

	p := make([]byte, ra.Size())

	_, err = ra.ReadAt(p, 0)
	return p, err
}

// WriteBlob writes data with the expected digest into the content store. If
// expected already exists, the method returns immediately and the reader will
// not be consumed.
//
// This is useful when the digest and size are known beforehand.
//
// Copy is buffered, so no need to wrap reader in buffered io.
func WriteBlob(ctx context.Context, cs Ingester, ref string, r io.Reader, size int64, expected digest.Digest, opts ...Opt) error {
	cw, err := cs.Writer(ctx, ref, size, expected)
	if err != nil {
		if !errdefs.IsAlreadyExists(err) {
			return err
		}

		return nil // all ready present
	}
	defer cw.Close()

	return Copy(ctx, cw, r, size, expected, opts...)
}

// Copy copies data with the expected digest from the reader into the
// provided content store writer.
//
// This is useful when the digest and size are known beforehand. When
// the size or digest is unknown, these values may be empty.
//
// Copy is buffered, so no need to wrap reader in buffered io.
func Copy(ctx context.Context, cw Writer, r io.Reader, size int64, expected digest.Digest, opts ...Opt) error {
	ws, err := cw.Status()
	if err != nil {
		return err
	}

	if ws.Offset > 0 {
		r, err = seekReader(r, ws.Offset, size)
		if err != nil {
			if !isUnseekable(err) {
				return errors.Wrapf(err, "unabled to resume write to %v", ws.Ref)
			}

			// reader is unseekable, try to move the writer back to the start.
			if err := cw.Truncate(0); err != nil {
				return errors.Wrapf(err, "content writer truncate failed")
			}
		}
	}

	buf := bufPool.Get().([]byte)
	defer bufPool.Put(buf)

	if _, err := io.CopyBuffer(cw, r, buf); err != nil {
		return err
	}

	if err := cw.Commit(ctx, size, expected, opts...); err != nil {
		if !errdefs.IsAlreadyExists(err) {
			return errors.Wrapf(err, "failed commit on ref %q", ws.Ref)
		}
	}

	return nil
}

var errUnseekable = errors.New("seek not supported")

func isUnseekable(err error) bool {
	return errors.Cause(err) == errUnseekable
}

// seekReader attempts to seek the reader to the given offset, either by
// resolving `io.Seeker` or by detecting `io.ReaderAt`.
func seekReader(r io.Reader, offset, size int64) (io.Reader, error) {
	// attempt to resolve r as a seeker and setup the offset.
	seeker, ok := r.(io.Seeker)
	if ok {
		nn, err := seeker.Seek(offset, io.SeekStart)
		if nn != offset {
			return nil, fmt.Errorf("failed to seek to offset %v", offset)
		}

		if err != nil {
			return nil, err
		}

		return r, nil
	}

	// ok, let's try io.ReaderAt!
	readerAt, ok := r.(io.ReaderAt)
	if ok && size > offset {
		sr := io.NewSectionReader(readerAt, offset, size)
		return sr, nil
	}

	return r, errors.Wrapf(errUnseekable, "seek to offset %v failed", offset)
}
