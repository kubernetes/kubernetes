package s3crypto

import (
	"crypto/sha256"
	"hash"
	"io"
)

// lengthReader returns the content length
type lengthReader interface {
	GetContentLength() int64
}

type contentLengthReader struct {
	contentLength int64
	body          io.Reader
}

func newContentLengthReader(f io.Reader) *contentLengthReader {
	return &contentLengthReader{body: f}
}

func (r *contentLengthReader) Read(b []byte) (int, error) {
	n, err := r.body.Read(b)
	if err != nil && err != io.EOF {
		return n, err
	}
	r.contentLength += int64(n)
	return n, err
}

func (r *contentLengthReader) GetContentLength() int64 {
	return r.contentLength
}

type sha256Writer struct {
	sha256 []byte
	hash   hash.Hash
	out    io.Writer
}

func newSHA256Writer(f io.Writer) *sha256Writer {
	return &sha256Writer{hash: sha256.New(), out: f}
}
func (r *sha256Writer) Write(b []byte) (int, error) {
	r.hash.Write(b)
	return r.out.Write(b)
}

func (r *sha256Writer) GetValue() []byte {
	return r.hash.Sum(nil)
}
