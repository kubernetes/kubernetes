// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lzma supports the decoding and encoding of LZMA streams.
// Reader and Writer support the classic LZMA format. Reader2 and
// Writer2 support the decoding and encoding of LZMA2 streams.
//
// The package is written completely in Go and doesn't rely on any external
// library.
package lzma

import (
	"errors"
	"io"
)

// ReaderConfig stores the parameters for the reader of the classic LZMA
// format.
type ReaderConfig struct {
	DictCap int
}

// fill converts the zero values of the configuration to the default values.
func (c *ReaderConfig) fill() {
	if c.DictCap == 0 {
		c.DictCap = 8 * 1024 * 1024
	}
}

// Verify checks the reader configuration for errors. Zero values will
// be replaced by default values.
func (c *ReaderConfig) Verify() error {
	c.fill()
	if !(MinDictCap <= c.DictCap && int64(c.DictCap) <= MaxDictCap) {
		return errors.New("lzma: dictionary capacity is out of range")
	}
	return nil
}

// Reader provides a reader for LZMA files or streams.
type Reader struct {
	lzma io.Reader
	h    header
	d    *decoder
}

// NewReader creates a new reader for an LZMA stream using the classic
// format. NewReader reads and checks the header of the LZMA stream.
func NewReader(lzma io.Reader) (r *Reader, err error) {
	return ReaderConfig{}.NewReader(lzma)
}

// NewReader creates a new reader for an LZMA stream in the classic
// format. The function reads and verifies the the header of the LZMA
// stream.
func (c ReaderConfig) NewReader(lzma io.Reader) (r *Reader, err error) {
	if err = c.Verify(); err != nil {
		return nil, err
	}
	data := make([]byte, HeaderLen)
	if _, err := io.ReadFull(lzma, data); err != nil {
		if err == io.EOF {
			return nil, errors.New("lzma: unexpected EOF")
		}
		return nil, err
	}
	r = &Reader{lzma: lzma}
	if err = r.h.unmarshalBinary(data); err != nil {
		return nil, err
	}
	if r.h.dictCap < MinDictCap {
		return nil, errors.New("lzma: dictionary capacity too small")
	}
	dictCap := r.h.dictCap
	if c.DictCap > dictCap {
		dictCap = c.DictCap
	}

	state := newState(r.h.properties)
	dict, err := newDecoderDict(dictCap)
	if err != nil {
		return nil, err
	}
	r.d, err = newDecoder(ByteReader(lzma), state, dict, r.h.size)
	if err != nil {
		return nil, err
	}
	return r, nil
}

// EOSMarker indicates that an EOS marker has been encountered.
func (r *Reader) EOSMarker() bool {
	return r.d.eosMarker
}

// Read returns uncompressed data.
func (r *Reader) Read(p []byte) (n int, err error) {
	return r.d.Read(p)
}
