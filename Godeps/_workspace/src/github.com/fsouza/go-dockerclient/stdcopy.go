// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package docker

import (
	"encoding/binary"
	"errors"
	"io"
)

const (
	stdWriterPrefixLen = 8
	stdWriterFdIndex   = 0
	stdWriterSizeIndex = 4
)

var errInvalidStdHeader = errors.New("Unrecognized input header")

func stdCopy(dstout, dsterr io.Writer, src io.Reader) (written int64, err error) {
	var (
		buf       = make([]byte, 32*1024+stdWriterPrefixLen+1)
		bufLen    = len(buf)
		nr, nw    int
		er, ew    error
		out       io.Writer
		frameSize int
	)
	for {
		for nr < stdWriterPrefixLen {
			var nr2 int
			nr2, er = src.Read(buf[nr:])
			if er == io.EOF {
				if nr < stdWriterPrefixLen && nr2 < stdWriterPrefixLen {
					return written, nil
				}
				nr += nr2
				break
			} else if er != nil {
				return 0, er
			}
			nr += nr2
		}
		switch buf[stdWriterFdIndex] {
		case 0:
			fallthrough
		case 1:
			out = dstout
		case 2:
			out = dsterr
		default:
			return 0, errInvalidStdHeader
		}
		frameSize = int(binary.BigEndian.Uint32(buf[stdWriterSizeIndex : stdWriterSizeIndex+4]))
		if frameSize+stdWriterPrefixLen > bufLen {
			buf = append(buf, make([]byte, frameSize+stdWriterPrefixLen-len(buf)+1)...)
			bufLen = len(buf)
		}
		for nr < frameSize+stdWriterPrefixLen {
			var nr2 int
			nr2, er = src.Read(buf[nr:])
			if er == io.EOF {
				if nr == 0 {
					return written, nil
				}
				nr += nr2
				break
			} else if er != nil {
				return 0, er
			}
			nr += nr2
		}
		bound := frameSize + stdWriterPrefixLen
		if bound > nr {
			bound = nr
		}
		nw, ew = out.Write(buf[stdWriterPrefixLen:bound])
		if nw > 0 {
			written += int64(nw)
		}
		if ew != nil {
			return 0, ew
		}
		if nw != frameSize {
			return written, io.ErrShortWrite
		}
		copy(buf, buf[frameSize+stdWriterPrefixLen:])
		nr -= frameSize + stdWriterPrefixLen
	}
}
