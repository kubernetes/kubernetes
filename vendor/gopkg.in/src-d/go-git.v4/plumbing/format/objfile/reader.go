package objfile

import (
	"compress/zlib"
	"errors"
	"io"
	"strconv"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/format/packfile"
)

var (
	ErrClosed       = errors.New("objfile: already closed")
	ErrHeader       = errors.New("objfile: invalid header")
	ErrNegativeSize = errors.New("objfile: negative object size")
)

// Reader reads and decodes compressed objfile data from a provided io.Reader.
// Reader implements io.ReadCloser. Close should be called when finished with
// the Reader. Close will not close the underlying io.Reader.
type Reader struct {
	multi  io.Reader
	zlib   io.ReadCloser
	hasher plumbing.Hasher
}

// NewReader returns a new Reader reading from r.
func NewReader(r io.Reader) (*Reader, error) {
	zlib, err := zlib.NewReader(r)
	if err != nil {
		return nil, packfile.ErrZLib.AddDetails(err.Error())
	}

	return &Reader{
		zlib: zlib,
	}, nil
}

// Header reads the type and the size of object, and prepares the reader for read
func (r *Reader) Header() (t plumbing.ObjectType, size int64, err error) {
	var raw []byte
	raw, err = r.readUntil(' ')
	if err != nil {
		return
	}

	t, err = plumbing.ParseObjectType(string(raw))
	if err != nil {
		return
	}

	raw, err = r.readUntil(0)
	if err != nil {
		return
	}

	size, err = strconv.ParseInt(string(raw), 10, 64)
	if err != nil {
		err = ErrHeader
		return
	}

	defer r.prepareForRead(t, size)
	return
}

// readSlice reads one byte at a time from r until it encounters delim or an
// error.
func (r *Reader) readUntil(delim byte) ([]byte, error) {
	var buf [1]byte
	value := make([]byte, 0, 16)
	for {
		if n, err := r.zlib.Read(buf[:]); err != nil && (err != io.EOF || n == 0) {
			if err == io.EOF {
				return nil, ErrHeader
			}
			return nil, err
		}

		if buf[0] == delim {
			return value, nil
		}

		value = append(value, buf[0])
	}
}

func (r *Reader) prepareForRead(t plumbing.ObjectType, size int64) {
	r.hasher = plumbing.NewHasher(t, size)
	r.multi = io.TeeReader(r.zlib, r.hasher)
}

// Read reads len(p) bytes into p from the object data stream. It returns
// the number of bytes read (0 <= n <= len(p)) and any error encountered. Even
// if Read returns n < len(p), it may use all of p as scratch space during the
// call.
//
// If Read encounters the end of the data stream it will return err == io.EOF,
// either in the current call if n > 0 or in a subsequent call.
func (r *Reader) Read(p []byte) (n int, err error) {
	return r.multi.Read(p)
}

// Hash returns the hash of the object data stream that has been read so far.
func (r *Reader) Hash() plumbing.Hash {
	return r.hasher.Sum()
}

// Close releases any resources consumed by the Reader. Calling Close does not
// close the wrapped io.Reader originally passed to NewReader.
func (r *Reader) Close() error {
	if err := r.zlib.Close(); err != nil {
		return err
	}

	return nil
}
