package s3crypto

import (
	"crypto/md5"
	"crypto/sha256"
	"hash"
	"io"
)

// hashReader is used for calculating SHA256 when following the sigv4 specification.
// Additionally this used for calculating the unencrypted MD5.
type hashReader interface {
	GetValue() []byte
	GetContentLength() int64
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

type md5Reader struct {
	contentLength int64
	hash          hash.Hash
	body          io.Reader
}

func newMD5Reader(body io.Reader) *md5Reader {
	return &md5Reader{hash: md5.New(), body: body}
}

func (w *md5Reader) Read(b []byte) (int, error) {
	n, err := w.body.Read(b)
	if err != nil && err != io.EOF {
		return n, err
	}
	w.contentLength += int64(n)
	w.hash.Write(b[:n])
	return n, err
}

func (w *md5Reader) GetValue() []byte {
	return w.hash.Sum(nil)
}

func (w *md5Reader) GetContentLength() int64 {
	return w.contentLength
}
