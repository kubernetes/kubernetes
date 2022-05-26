// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"compress/bzip2"
	"compress/flate"
	"compress/zlib"
	"golang.org/x/crypto/openpgp/errors"
	"io"
	"strconv"
)

// Compressed represents a compressed OpenPGP packet. The decompressed contents
// will contain more OpenPGP packets. See RFC 4880, section 5.6.
type Compressed struct {
	Body io.Reader
}

const (
	NoCompression      = flate.NoCompression
	BestSpeed          = flate.BestSpeed
	BestCompression    = flate.BestCompression
	DefaultCompression = flate.DefaultCompression
)

// CompressionConfig contains compressor configuration settings.
type CompressionConfig struct {
	// Level is the compression level to use. It must be set to
	// between -1 and 9, with -1 causing the compressor to use the
	// default compression level, 0 causing the compressor to use
	// no compression and 1 to 9 representing increasing (better,
	// slower) compression levels. If Level is less than -1 or
	// more then 9, a non-nil error will be returned during
	// encryption. See the constants above for convenient common
	// settings for Level.
	Level int
}

func (c *Compressed) parse(r io.Reader) error {
	var buf [1]byte
	_, err := readFull(r, buf[:])
	if err != nil {
		return err
	}

	switch buf[0] {
	case 1:
		c.Body = flate.NewReader(r)
	case 2:
		c.Body, err = zlib.NewReader(r)
	case 3:
		c.Body = bzip2.NewReader(r)
	default:
		err = errors.UnsupportedError("unknown compression algorithm: " + strconv.Itoa(int(buf[0])))
	}

	return err
}

// compressedWriterCloser represents the serialized compression stream
// header and the compressor. Its Close() method ensures that both the
// compressor and serialized stream header are closed. Its Write()
// method writes to the compressor.
type compressedWriteCloser struct {
	sh io.Closer      // Stream Header
	c  io.WriteCloser // Compressor
}

func (cwc compressedWriteCloser) Write(p []byte) (int, error) {
	return cwc.c.Write(p)
}

func (cwc compressedWriteCloser) Close() (err error) {
	err = cwc.c.Close()
	if err != nil {
		return err
	}

	return cwc.sh.Close()
}

// SerializeCompressed serializes a compressed data packet to w and
// returns a WriteCloser to which the literal data packets themselves
// can be written and which MUST be closed on completion. If cc is
// nil, sensible defaults will be used to configure the compression
// algorithm.
func SerializeCompressed(w io.WriteCloser, algo CompressionAlgo, cc *CompressionConfig) (literaldata io.WriteCloser, err error) {
	compressed, err := serializeStreamHeader(w, packetTypeCompressed)
	if err != nil {
		return
	}

	_, err = compressed.Write([]byte{uint8(algo)})
	if err != nil {
		return
	}

	level := DefaultCompression
	if cc != nil {
		level = cc.Level
	}

	var compressor io.WriteCloser
	switch algo {
	case CompressionZIP:
		compressor, err = flate.NewWriter(compressed, level)
	case CompressionZLIB:
		compressor, err = zlib.NewWriterLevel(compressed, level)
	default:
		s := strconv.Itoa(int(algo))
		err = errors.UnsupportedError("Unsupported compression algorithm: " + s)
	}
	if err != nil {
		return
	}

	literaldata = compressedWriteCloser{compressed, compressor}

	return
}
