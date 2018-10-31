// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xz

import (
	"errors"
	"hash"
	"io"

	"github.com/ulikunitz/xz/lzma"
)

// WriterConfig describe the parameters for an xz writer.
type WriterConfig struct {
	Properties *lzma.Properties
	DictCap    int
	BufSize    int
	BlockSize  int64
	// checksum method: CRC32, CRC64 or SHA256
	CheckSum byte
	// match algorithm
	Matcher lzma.MatchAlgorithm
}

// fill replaces zero values with default values.
func (c *WriterConfig) fill() {
	if c.Properties == nil {
		c.Properties = &lzma.Properties{LC: 3, LP: 0, PB: 2}
	}
	if c.DictCap == 0 {
		c.DictCap = 8 * 1024 * 1024
	}
	if c.BufSize == 0 {
		c.BufSize = 4096
	}
	if c.BlockSize == 0 {
		c.BlockSize = maxInt64
	}
	if c.CheckSum == 0 {
		c.CheckSum = CRC64
	}
}

// Verify checks the configuration for errors. Zero values will be
// replaced by default values.
func (c *WriterConfig) Verify() error {
	if c == nil {
		return errors.New("xz: writer configuration is nil")
	}
	c.fill()
	lc := lzma.Writer2Config{
		Properties: c.Properties,
		DictCap:    c.DictCap,
		BufSize:    c.BufSize,
		Matcher:    c.Matcher,
	}
	if err := lc.Verify(); err != nil {
		return err
	}
	if c.BlockSize <= 0 {
		return errors.New("xz: block size out of range")
	}
	if err := verifyFlags(c.CheckSum); err != nil {
		return err
	}
	return nil
}

// filters creates the filter list for the given parameters.
func (c *WriterConfig) filters() []filter {
	return []filter{&lzmaFilter{int64(c.DictCap)}}
}

// maxInt64 defines the maximum 64-bit signed integer.
const maxInt64 = 1<<63 - 1

// verifyFilters checks the filter list for the length and the right
// sequence of filters.
func verifyFilters(f []filter) error {
	if len(f) == 0 {
		return errors.New("xz: no filters")
	}
	if len(f) > 4 {
		return errors.New("xz: more than four filters")
	}
	for _, g := range f[:len(f)-1] {
		if g.last() {
			return errors.New("xz: last filter is not last")
		}
	}
	if !f[len(f)-1].last() {
		return errors.New("xz: wrong last filter")
	}
	return nil
}

// newFilterWriteCloser converts a filter list into a WriteCloser that
// can be used by a blockWriter.
func (c *WriterConfig) newFilterWriteCloser(w io.Writer, f []filter) (fw io.WriteCloser, err error) {
	if err = verifyFilters(f); err != nil {
		return nil, err
	}
	fw = nopWriteCloser(w)
	for i := len(f) - 1; i >= 0; i-- {
		fw, err = f[i].writeCloser(fw, c)
		if err != nil {
			return nil, err
		}
	}
	return fw, nil
}

// nopWCloser implements a WriteCloser with a Close method not doing
// anything.
type nopWCloser struct {
	io.Writer
}

// Close returns nil and doesn't do anything else.
func (c nopWCloser) Close() error {
	return nil
}

// nopWriteCloser converts the Writer into a WriteCloser with a Close
// function that does nothing beside returning nil.
func nopWriteCloser(w io.Writer) io.WriteCloser {
	return nopWCloser{w}
}

// Writer compresses data written to it. It is an io.WriteCloser.
type Writer struct {
	WriterConfig

	xz      io.Writer
	bw      *blockWriter
	newHash func() hash.Hash
	h       header
	index   []record
	closed  bool
}

// newBlockWriter creates a new block writer writes the header out.
func (w *Writer) newBlockWriter() error {
	var err error
	w.bw, err = w.WriterConfig.newBlockWriter(w.xz, w.newHash())
	if err != nil {
		return err
	}
	if err = w.bw.writeHeader(w.xz); err != nil {
		return err
	}
	return nil
}

// closeBlockWriter closes a block writer and records the sizes in the
// index.
func (w *Writer) closeBlockWriter() error {
	var err error
	if err = w.bw.Close(); err != nil {
		return err
	}
	w.index = append(w.index, w.bw.record())
	return nil
}

// NewWriter creates a new xz writer using default parameters.
func NewWriter(xz io.Writer) (w *Writer, err error) {
	return WriterConfig{}.NewWriter(xz)
}

// NewWriter creates a new Writer using the given configuration parameters.
func (c WriterConfig) NewWriter(xz io.Writer) (w *Writer, err error) {
	if err = c.Verify(); err != nil {
		return nil, err
	}
	w = &Writer{
		WriterConfig: c,
		xz:           xz,
		h:            header{c.CheckSum},
		index:        make([]record, 0, 4),
	}
	if w.newHash, err = newHashFunc(c.CheckSum); err != nil {
		return nil, err
	}
	data, err := w.h.MarshalBinary()
	if _, err = xz.Write(data); err != nil {
		return nil, err
	}
	if err = w.newBlockWriter(); err != nil {
		return nil, err
	}
	return w, nil

}

// Write compresses the uncompressed data provided.
func (w *Writer) Write(p []byte) (n int, err error) {
	if w.closed {
		return 0, errClosed
	}
	for {
		k, err := w.bw.Write(p[n:])
		n += k
		if err != errNoSpace {
			return n, err
		}
		if err = w.closeBlockWriter(); err != nil {
			return n, err
		}
		if err = w.newBlockWriter(); err != nil {
			return n, err
		}
	}
}

// Close closes the writer and adds the footer to the Writer. Close
// doesn't close the underlying writer.
func (w *Writer) Close() error {
	if w.closed {
		return errClosed
	}
	w.closed = true
	var err error
	if err = w.closeBlockWriter(); err != nil {
		return err
	}

	f := footer{flags: w.h.flags}
	if f.indexSize, err = writeIndex(w.xz, w.index); err != nil {
		return err
	}
	data, err := f.MarshalBinary()
	if err != nil {
		return err
	}
	if _, err = w.xz.Write(data); err != nil {
		return err
	}
	return nil
}

// countingWriter is a writer that counts all data written to it.
type countingWriter struct {
	w io.Writer
	n int64
}

// Write writes data to the countingWriter.
func (cw *countingWriter) Write(p []byte) (n int, err error) {
	n, err = cw.w.Write(p)
	cw.n += int64(n)
	if err == nil && cw.n < 0 {
		return n, errors.New("xz: counter overflow")
	}
	return
}

// blockWriter is writes a single block.
type blockWriter struct {
	cxz countingWriter
	// mw combines io.WriteCloser w and the hash.
	mw        io.Writer
	w         io.WriteCloser
	n         int64
	blockSize int64
	closed    bool
	headerLen int

	filters []filter
	hash    hash.Hash
}

// newBlockWriter creates a new block writer.
func (c *WriterConfig) newBlockWriter(xz io.Writer, hash hash.Hash) (bw *blockWriter, err error) {
	bw = &blockWriter{
		cxz:       countingWriter{w: xz},
		blockSize: c.BlockSize,
		filters:   c.filters(),
		hash:      hash,
	}
	bw.w, err = c.newFilterWriteCloser(&bw.cxz, bw.filters)
	if err != nil {
		return nil, err
	}
	bw.mw = io.MultiWriter(bw.w, bw.hash)
	return bw, nil
}

// writeHeader writes the header. If the function is called after Close
// the commpressedSize and uncompressedSize fields will be filled.
func (bw *blockWriter) writeHeader(w io.Writer) error {
	h := blockHeader{
		compressedSize:   -1,
		uncompressedSize: -1,
		filters:          bw.filters,
	}
	if bw.closed {
		h.compressedSize = bw.compressedSize()
		h.uncompressedSize = bw.uncompressedSize()
	}
	data, err := h.MarshalBinary()
	if err != nil {
		return err
	}
	if _, err = w.Write(data); err != nil {
		return err
	}
	bw.headerLen = len(data)
	return nil
}

// compressed size returns the amount of data written to the underlying
// stream.
func (bw *blockWriter) compressedSize() int64 {
	return bw.cxz.n
}

// uncompressedSize returns the number of data written to the
// blockWriter
func (bw *blockWriter) uncompressedSize() int64 {
	return bw.n
}

// unpaddedSize returns the sum of the header length, the uncompressed
// size of the block and the hash size.
func (bw *blockWriter) unpaddedSize() int64 {
	if bw.headerLen <= 0 {
		panic("xz: block header not written")
	}
	n := int64(bw.headerLen)
	n += bw.compressedSize()
	n += int64(bw.hash.Size())
	return n
}

// record returns the record for the current stream. Call Close before
// calling this method.
func (bw *blockWriter) record() record {
	return record{bw.unpaddedSize(), bw.uncompressedSize()}
}

var errClosed = errors.New("xz: writer already closed")

var errNoSpace = errors.New("xz: no space")

// Write writes uncompressed data to the block writer.
func (bw *blockWriter) Write(p []byte) (n int, err error) {
	if bw.closed {
		return 0, errClosed
	}

	t := bw.blockSize - bw.n
	if int64(len(p)) > t {
		err = errNoSpace
		p = p[:t]
	}

	var werr error
	n, werr = bw.mw.Write(p)
	bw.n += int64(n)
	if werr != nil {
		return n, werr
	}
	return n, err
}

// Close closes the writer.
func (bw *blockWriter) Close() error {
	if bw.closed {
		return errClosed
	}
	bw.closed = true
	if err := bw.w.Close(); err != nil {
		return err
	}
	s := bw.hash.Size()
	k := padLen(bw.cxz.n)
	p := make([]byte, k+s)
	bw.hash.Sum(p[k:k])
	if _, err := bw.cxz.w.Write(p); err != nil {
		return err
	}
	return nil
}
