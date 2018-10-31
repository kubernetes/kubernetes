// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package xz supports the compression and decompression of xz files. It
// supports version 1.0.4 of the specification without the non-LZMA2
// filters. See http://tukaani.org/xz/xz-file-format-1.0.4.txt
package xz

import (
	"bytes"
	"errors"
	"fmt"
	"hash"
	"io"

	"github.com/ulikunitz/xz/internal/xlog"
	"github.com/ulikunitz/xz/lzma"
)

// ReaderConfig defines the parameters for the xz reader. The
// SingleStream parameter requests the reader to assume that the
// underlying stream contains only a single stream.
type ReaderConfig struct {
	DictCap      int
	SingleStream bool
}

// fill replaces all zero values with their default values.
func (c *ReaderConfig) fill() {
	if c.DictCap == 0 {
		c.DictCap = 8 * 1024 * 1024
	}
}

// Verify checks the reader parameters for Validity. Zero values will be
// replaced by default values.
func (c *ReaderConfig) Verify() error {
	if c == nil {
		return errors.New("xz: reader parameters are nil")
	}
	lc := lzma.Reader2Config{DictCap: c.DictCap}
	if err := lc.Verify(); err != nil {
		return err
	}
	return nil
}

// Reader supports the reading of one or multiple xz streams.
type Reader struct {
	ReaderConfig

	xz io.Reader
	sr *streamReader
}

// streamReader decodes a single xz stream
type streamReader struct {
	ReaderConfig

	xz      io.Reader
	br      *blockReader
	newHash func() hash.Hash
	h       header
	index   []record
}

// NewReader creates a new xz reader using the default parameters.
// The function reads and checks the header of the first XZ stream. The
// reader will process multiple streams including padding.
func NewReader(xz io.Reader) (r *Reader, err error) {
	return ReaderConfig{}.NewReader(xz)
}

// NewReader creates an xz stream reader. The created reader will be
// able to process multiple streams and padding unless a SingleStream
// has been set in the reader configuration c.
func (c ReaderConfig) NewReader(xz io.Reader) (r *Reader, err error) {
	if err = c.Verify(); err != nil {
		return nil, err
	}
	r = &Reader{
		ReaderConfig: c,
		xz:           xz,
	}
	if r.sr, err = c.newStreamReader(xz); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	return r, nil
}

var errUnexpectedData = errors.New("xz: unexpected data after stream")

// Read reads uncompressed data from the stream.
func (r *Reader) Read(p []byte) (n int, err error) {
	for n < len(p) {
		if r.sr == nil {
			if r.SingleStream {
				data := make([]byte, 1)
				_, err = io.ReadFull(r.xz, data)
				if err != io.EOF {
					return n, errUnexpectedData
				}
				return n, io.EOF
			}
			for {
				r.sr, err = r.ReaderConfig.newStreamReader(r.xz)
				if err != errPadding {
					break
				}
			}
			if err != nil {
				return n, err
			}
		}
		k, err := r.sr.Read(p[n:])
		n += k
		if err != nil {
			if err == io.EOF {
				r.sr = nil
				continue
			}
			return n, err
		}
	}
	return n, nil
}

var errPadding = errors.New("xz: padding (4 zero bytes) encountered")

// newStreamReader creates a new xz stream reader using the given configuration
// parameters. NewReader reads and checks the header of the xz stream.
func (c ReaderConfig) newStreamReader(xz io.Reader) (r *streamReader, err error) {
	if err = c.Verify(); err != nil {
		return nil, err
	}
	data := make([]byte, HeaderLen)
	if _, err := io.ReadFull(xz, data[:4]); err != nil {
		return nil, err
	}
	if bytes.Equal(data[:4], []byte{0, 0, 0, 0}) {
		return nil, errPadding
	}
	if _, err = io.ReadFull(xz, data[4:]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	r = &streamReader{
		ReaderConfig: c,
		xz:           xz,
		index:        make([]record, 0, 4),
	}
	if err = r.h.UnmarshalBinary(data); err != nil {
		return nil, err
	}
	xlog.Debugf("xz header %s", r.h)
	if r.newHash, err = newHashFunc(r.h.flags); err != nil {
		return nil, err
	}
	return r, nil
}

// errIndex indicates an error with the xz file index.
var errIndex = errors.New("xz: error in xz file index")

// readTail reads the index body and the xz footer.
func (r *streamReader) readTail() error {
	index, n, err := readIndexBody(r.xz)
	if err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	if len(index) != len(r.index) {
		return fmt.Errorf("xz: index length is %d; want %d",
			len(index), len(r.index))
	}
	for i, rec := range r.index {
		if rec != index[i] {
			return fmt.Errorf("xz: record %d is %v; want %v",
				i, rec, index[i])
		}
	}

	p := make([]byte, footerLen)
	if _, err = io.ReadFull(r.xz, p); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	var f footer
	if err = f.UnmarshalBinary(p); err != nil {
		return err
	}
	xlog.Debugf("xz footer %s", f)
	if f.flags != r.h.flags {
		return errors.New("xz: footer flags incorrect")
	}
	if f.indexSize != int64(n)+1 {
		return errors.New("xz: index size in footer wrong")
	}
	return nil
}

// Read reads actual data from the xz stream.
func (r *streamReader) Read(p []byte) (n int, err error) {
	for n < len(p) {
		if r.br == nil {
			bh, hlen, err := readBlockHeader(r.xz)
			if err != nil {
				if err == errIndexIndicator {
					if err = r.readTail(); err != nil {
						return n, err
					}
					return n, io.EOF
				}
				return n, err
			}
			xlog.Debugf("block %v", *bh)
			r.br, err = r.ReaderConfig.newBlockReader(r.xz, bh,
				hlen, r.newHash())
			if err != nil {
				return n, err
			}
		}
		k, err := r.br.Read(p[n:])
		n += k
		if err != nil {
			if err == io.EOF {
				r.index = append(r.index, r.br.record())
				r.br = nil
			} else {
				return n, err
			}
		}
	}
	return n, nil
}

// countingReader is a reader that counts the bytes read.
type countingReader struct {
	r io.Reader
	n int64
}

// Read reads data from the wrapped reader and adds it to the n field.
func (lr *countingReader) Read(p []byte) (n int, err error) {
	n, err = lr.r.Read(p)
	lr.n += int64(n)
	return n, err
}

// blockReader supports the reading of a block.
type blockReader struct {
	lxz       countingReader
	header    *blockHeader
	headerLen int
	n         int64
	hash      hash.Hash
	r         io.Reader
	err       error
}

// newBlockReader creates a new block reader.
func (c *ReaderConfig) newBlockReader(xz io.Reader, h *blockHeader,
	hlen int, hash hash.Hash) (br *blockReader, err error) {

	br = &blockReader{
		lxz:       countingReader{r: xz},
		header:    h,
		headerLen: hlen,
		hash:      hash,
	}

	fr, err := c.newFilterReader(&br.lxz, h.filters)
	if err != nil {
		return nil, err
	}
	br.r = io.TeeReader(fr, br.hash)

	return br, nil
}

// uncompressedSize returns the uncompressed size of the block.
func (br *blockReader) uncompressedSize() int64 {
	return br.n
}

// compressedSize returns the compressed size of the block.
func (br *blockReader) compressedSize() int64 {
	return br.lxz.n
}

// unpaddedSize computes the unpadded size for the block.
func (br *blockReader) unpaddedSize() int64 {
	n := int64(br.headerLen)
	n += br.compressedSize()
	n += int64(br.hash.Size())
	return n
}

// record returns the index record for the current block.
func (br *blockReader) record() record {
	return record{br.unpaddedSize(), br.uncompressedSize()}
}

// errBlockSize indicates that the size of the block in the block header
// is wrong.
var errBlockSize = errors.New("xz: wrong uncompressed size for block")

// Read reads data from the block.
func (br *blockReader) Read(p []byte) (n int, err error) {
	n, err = br.r.Read(p)
	br.n += int64(n)

	u := br.header.uncompressedSize
	if u >= 0 && br.uncompressedSize() > u {
		return n, errors.New("xz: wrong uncompressed size for block")
	}
	c := br.header.compressedSize
	if c >= 0 && br.compressedSize() > c {
		return n, errors.New("xz: wrong compressed size for block")
	}
	if err != io.EOF {
		return n, err
	}
	if br.uncompressedSize() < u || br.compressedSize() < c {
		return n, io.ErrUnexpectedEOF
	}

	s := br.hash.Size()
	k := padLen(br.lxz.n)
	q := make([]byte, k+s, k+2*s)
	if _, err = io.ReadFull(br.lxz.r, q); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return n, err
	}
	if !allZeros(q[:k]) {
		return n, errors.New("xz: non-zero block padding")
	}
	checkSum := q[k:]
	computedSum := br.hash.Sum(checkSum[s:])
	if !bytes.Equal(checkSum, computedSum) {
		return n, errors.New("xz: checksum error for block")
	}
	return n, io.EOF
}

func (c *ReaderConfig) newFilterReader(r io.Reader, f []filter) (fr io.Reader,
	err error) {

	if err = verifyFilters(f); err != nil {
		return nil, err
	}

	fr = r
	for i := len(f) - 1; i >= 0; i-- {
		fr, err = f[i].reader(fr, c)
		if err != nil {
			return nil, err
		}
	}
	return fr, nil
}
