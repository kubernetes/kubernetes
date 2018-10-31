// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"bytes"
	"errors"
	"io"
)

// Writer2Config is used to create a Writer2 using parameters.
type Writer2Config struct {
	// The properties for the encoding. If the it is nil the value
	// {LC: 3, LP: 0, PB: 2} will be chosen.
	Properties *Properties
	// The capacity of the dictionary. If DictCap is zero, the value
	// 8 MiB will be chosen.
	DictCap int
	// Size of the lookahead buffer; value 0 indicates default size
	// 4096
	BufSize int
	// Match algorithm
	Matcher MatchAlgorithm
}

// fill replaces zero values with default values.
func (c *Writer2Config) fill() {
	if c.Properties == nil {
		c.Properties = &Properties{LC: 3, LP: 0, PB: 2}
	}
	if c.DictCap == 0 {
		c.DictCap = 8 * 1024 * 1024
	}
	if c.BufSize == 0 {
		c.BufSize = 4096
	}
}

// Verify checks the Writer2Config for correctness. Zero values will be
// replaced by default values.
func (c *Writer2Config) Verify() error {
	c.fill()
	var err error
	if c == nil {
		return errors.New("lzma: WriterConfig is nil")
	}
	if c.Properties == nil {
		return errors.New("lzma: WriterConfig has no Properties set")
	}
	if err = c.Properties.verify(); err != nil {
		return err
	}
	if !(MinDictCap <= c.DictCap && int64(c.DictCap) <= MaxDictCap) {
		return errors.New("lzma: dictionary capacity is out of range")
	}
	if !(maxMatchLen <= c.BufSize) {
		return errors.New("lzma: lookahead buffer size too small")
	}
	if c.Properties.LC+c.Properties.LP > 4 {
		return errors.New("lzma: sum of lc and lp exceeds 4")
	}
	if err = c.Matcher.verify(); err != nil {
		return err
	}
	return nil
}

// Writer2 supports the creation of an LZMA2 stream. But note that
// written data is buffered, so call Flush or Close to write data to the
// underlying writer. The Close method writes the end-of-stream marker
// to the stream. So you may be able to concatenate the output of two
// writers as long the output of the first writer has only been flushed
// but not closed.
//
// Any change to the fields Properties, DictCap must be done before the
// first call to Write, Flush or Close.
type Writer2 struct {
	w io.Writer

	start   *state
	encoder *encoder

	cstate chunkState
	ctype  chunkType

	buf bytes.Buffer
	lbw LimitedByteWriter
}

// NewWriter2 creates an LZMA2 chunk sequence writer with the default
// parameters and options.
func NewWriter2(lzma2 io.Writer) (w *Writer2, err error) {
	return Writer2Config{}.NewWriter2(lzma2)
}

// NewWriter2 creates a new LZMA2 writer using the given configuration.
func (c Writer2Config) NewWriter2(lzma2 io.Writer) (w *Writer2, err error) {
	if err = c.Verify(); err != nil {
		return nil, err
	}
	w = &Writer2{
		w:      lzma2,
		start:  newState(*c.Properties),
		cstate: start,
		ctype:  start.defaultChunkType(),
	}
	w.buf.Grow(maxCompressed)
	w.lbw = LimitedByteWriter{BW: &w.buf, N: maxCompressed}
	m, err := c.Matcher.new(c.DictCap)
	if err != nil {
		return nil, err
	}
	d, err := newEncoderDict(c.DictCap, c.BufSize, m)
	if err != nil {
		return nil, err
	}
	w.encoder, err = newEncoder(&w.lbw, cloneState(w.start), d, 0)
	if err != nil {
		return nil, err
	}
	return w, nil
}

// written returns the number of bytes written to the current chunk
func (w *Writer2) written() int {
	if w.encoder == nil {
		return 0
	}
	return int(w.encoder.Compressed()) + w.encoder.dict.Buffered()
}

// errClosed indicates that the writer is closed.
var errClosed = errors.New("lzma: writer closed")

// Writes data to LZMA2 stream. Note that written data will be buffered.
// Use Flush or Close to ensure that data is written to the underlying
// writer.
func (w *Writer2) Write(p []byte) (n int, err error) {
	if w.cstate == stop {
		return 0, errClosed
	}
	for n < len(p) {
		m := maxUncompressed - w.written()
		if m <= 0 {
			panic("lzma: maxUncompressed reached")
		}
		var q []byte
		if n+m < len(p) {
			q = p[n : n+m]
		} else {
			q = p[n:]
		}
		k, err := w.encoder.Write(q)
		n += k
		if err != nil && err != ErrLimit {
			return n, err
		}
		if err == ErrLimit || k == m {
			if err = w.flushChunk(); err != nil {
				return n, err
			}
		}
	}
	return n, nil
}

// writeUncompressedChunk writes an uncompressed chunk to the LZMA2
// stream.
func (w *Writer2) writeUncompressedChunk() error {
	u := w.encoder.Compressed()
	if u <= 0 {
		return errors.New("lzma: can't write empty uncompressed chunk")
	}
	if u > maxUncompressed {
		panic("overrun of uncompressed data limit")
	}
	switch w.ctype {
	case cLRND:
		w.ctype = cUD
	default:
		w.ctype = cU
	}
	w.encoder.state = w.start

	header := chunkHeader{
		ctype:        w.ctype,
		uncompressed: uint32(u - 1),
	}
	hdata, err := header.MarshalBinary()
	if err != nil {
		return err
	}
	if _, err = w.w.Write(hdata); err != nil {
		return err
	}
	_, err = w.encoder.dict.CopyN(w.w, int(u))
	return err
}

// writeCompressedChunk writes a compressed chunk to the underlying
// writer.
func (w *Writer2) writeCompressedChunk() error {
	if w.ctype == cU || w.ctype == cUD {
		panic("chunk type uncompressed")
	}

	u := w.encoder.Compressed()
	if u <= 0 {
		return errors.New("writeCompressedChunk: empty chunk")
	}
	if u > maxUncompressed {
		panic("overrun of uncompressed data limit")
	}
	c := w.buf.Len()
	if c <= 0 {
		panic("no compressed data")
	}
	if c > maxCompressed {
		panic("overrun of compressed data limit")
	}
	header := chunkHeader{
		ctype:        w.ctype,
		uncompressed: uint32(u - 1),
		compressed:   uint16(c - 1),
		props:        w.encoder.state.Properties,
	}
	hdata, err := header.MarshalBinary()
	if err != nil {
		return err
	}
	if _, err = w.w.Write(hdata); err != nil {
		return err
	}
	_, err = io.Copy(w.w, &w.buf)
	return err
}

// writes a single chunk to the underlying writer.
func (w *Writer2) writeChunk() error {
	u := int(uncompressedHeaderLen + w.encoder.Compressed())
	c := headerLen(w.ctype) + w.buf.Len()
	if u < c {
		return w.writeUncompressedChunk()
	}
	return w.writeCompressedChunk()
}

// flushChunk terminates the current chunk. The encoder will be reset
// to support the next chunk.
func (w *Writer2) flushChunk() error {
	if w.written() == 0 {
		return nil
	}
	var err error
	if err = w.encoder.Close(); err != nil {
		return err
	}
	if err = w.writeChunk(); err != nil {
		return err
	}
	w.buf.Reset()
	w.lbw.N = maxCompressed
	if err = w.encoder.Reopen(&w.lbw); err != nil {
		return err
	}
	if err = w.cstate.next(w.ctype); err != nil {
		return err
	}
	w.ctype = w.cstate.defaultChunkType()
	w.start = cloneState(w.encoder.state)
	return nil
}

// Flush writes all buffered data out to the underlying stream. This
// could result in multiple chunks to be created.
func (w *Writer2) Flush() error {
	if w.cstate == stop {
		return errClosed
	}
	for w.written() > 0 {
		if err := w.flushChunk(); err != nil {
			return err
		}
	}
	return nil
}

// Close terminates the LZMA2 stream with an EOS chunk.
func (w *Writer2) Close() error {
	if w.cstate == stop {
		return errClosed
	}
	if err := w.Flush(); err != nil {
		return nil
	}
	// write zero byte EOS chunk
	_, err := w.w.Write([]byte{0})
	if err != nil {
		return err
	}
	w.cstate = stop
	return nil
}
