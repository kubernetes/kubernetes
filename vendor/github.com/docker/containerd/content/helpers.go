package content

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// WriteBlob writes data with the expected digest into the content store. If
// expected already exists, the method returns immediately and the reader will
// not be consumed.
//
// This is useful when the digest and size are known beforehand.
//
// Copy is buffered, so no need to wrap reader in buffered io.
func WriteBlob(ctx context.Context, cs Ingester, ref string, r io.Reader, size int64, expected digest.Digest) error {
	cw, err := cs.Writer(ctx, ref, size, expected)
	if err != nil {
		if !IsExists(err) {
			return err
		}

		return nil // all ready present
	}
	defer cw.Close()

	ws, err := cw.Status()
	if err != nil {
		return err
	}

	if ws.Offset > 0 {
		r, err = seekReader(r, ws.Offset, size)
		if err != nil {
			if !isUnseekable(err) {
				return errors.Wrapf(err, "unabled to resume write to %v", ref)
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

	if err := cw.Commit(size, expected); err != nil {
		if !IsExists(err) {
			return errors.Wrapf(err, "failed commit on ref %q", ref)
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

func readFileString(path string) (string, error) {
	p, err := ioutil.ReadFile(path)
	return string(p), err
}
