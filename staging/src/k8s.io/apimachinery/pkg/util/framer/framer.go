/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package framer implements simple frame decoding techniques for an io.ReadCloser
package framer

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
)

type lengthDelimitedFrameWriter struct {
	w io.Writer
	h [4]byte
}

func NewLengthDelimitedFrameWriter(w io.Writer) io.Writer {
	return &lengthDelimitedFrameWriter{w: w}
}

// Write writes a single frame to the nested writer, prepending it with the length
// in bytes of data (as a 4 byte, bigendian uint32).
func (w *lengthDelimitedFrameWriter) Write(data []byte) (int, error) {
	binary.BigEndian.PutUint32(w.h[:], uint32(len(data)))
	n, err := w.w.Write(w.h[:])
	if err != nil {
		return 0, err
	}
	if n != len(w.h) {
		return 0, io.ErrShortWrite
	}
	return w.w.Write(data)
}

type lengthDelimitedFrameReader struct {
	r         io.ReadCloser
	remaining int
}

// NewLengthDelimitedFrameReader returns an io.Reader that will decode length-prefixed
// frames off of a stream.
//
// The protocol is:
//
//	stream: message ...
//	message: prefix body
//	prefix: 4 byte uint32 in BigEndian order, denotes length of body
//	body: bytes (0..prefix)
//
// If the buffer passed to Read is not long enough to contain an entire frame, io.ErrShortRead
// will be returned along with the number of bytes read.
func NewLengthDelimitedFrameReader(r io.ReadCloser) io.ReadCloser {
	return &lengthDelimitedFrameReader{r: r}
}

// Read attempts to read an entire frame into data. If that is not possible, io.ErrShortBuffer
// is returned and subsequent calls will attempt to read the last frame. A frame is complete when
// err is nil.
func (r *lengthDelimitedFrameReader) Read(data []byte) (int, error) {
	if r.remaining <= 0 {
		header := [4]byte{}
		n, err := io.ReadAtLeast(r.r, header[:4], 4)
		if err != nil {
			return 0, err
		}
		if n != 4 {
			return 0, io.ErrUnexpectedEOF
		}
		frameLength := int(binary.BigEndian.Uint32(header[:]))
		r.remaining = frameLength
	}

	expect := r.remaining
	max := expect
	if max > len(data) {
		max = len(data)
	}
	n, err := io.ReadAtLeast(r.r, data[:max], int(max))
	r.remaining -= n
	if err != nil {
		return n, err
	}
	if r.remaining > 0 {
		return n, io.ErrShortBuffer
	}
	if n != expect {
		return n, io.ErrUnexpectedEOF
	}

	return n, nil
}

func (r *lengthDelimitedFrameReader) Close() error {
	return r.r.Close()
}

type jsonFrameReader struct {
	r         io.ReadCloser
	decoder   *json.Decoder
	remaining []byte
}

// NewJSONFramedReader returns an io.Reader that will decode individual JSON objects off
// of a wire.
//
// The boundaries between each frame are valid JSON objects. A JSON parsing error will terminate
// the read.
func NewJSONFramedReader(r io.ReadCloser) io.ReadCloser {
	return &jsonFrameReader{
		r:       r,
		decoder: json.NewDecoder(r),
	}
}

// ReadFrame decodes the next JSON object in the stream, or returns an error. The returned
// byte slice will be modified the next time ReadFrame is invoked and should not be altered.
func (r *jsonFrameReader) Read(data []byte) (int, error) {
	// Return whatever remaining data exists from an in progress frame
	if n := len(r.remaining); n > 0 {
		if n <= len(data) {
			//nolint:staticcheck // SA4006,SA4010 underlying array of data is modified here.
			data = append(data[0:0], r.remaining...)
			r.remaining = nil
			return n, nil
		}

		n = len(data)
		//nolint:staticcheck // SA4006,SA4010 underlying array of data is modified here.
		data = append(data[0:0], r.remaining[:n]...)
		r.remaining = r.remaining[n:]
		return n, io.ErrShortBuffer
	}

	// RawMessage#Unmarshal appends to data - we reset the slice down to 0 and will either see
	// data written to data, or be larger than data and a different array.
	m := json.RawMessage(data[:0])
	if err := r.decoder.Decode(&m); err != nil {
		return 0, err
	}

	// If capacity of data is less than length of the message, decoder will allocate a new slice
	// and set m to it, which means we need to copy the partial result back into data and preserve
	// the remaining result for subsequent reads.
	if len(m) > cap(data) {
		copy(data, m)
		r.remaining = m[len(data):]
		return len(data), io.ErrShortBuffer
	}

	if len(m) > len(data) {
		// The bytes beyond len(data) were stored in data's underlying array, which we do
		// not own after this function returns.
		r.remaining = append([]byte(nil), m[len(data):]...)
		return len(data), io.ErrShortBuffer
	}

	return len(m), nil
}

func (r *jsonFrameReader) Close() error {
	return r.r.Close()
}

// lineDelimitedFrameReader is a fast-path frame reader for streams that are
// already known to be compact JSON values separated by `\n`. It does not
// understand JSON; it just splits on `\n`. This is correct only for producers
// that never emit raw newlines outside of frame boundaries — e.g. anything
// that goes through json.Marshal / json.Encoder, including the apiserver
// watch path (StreamSerializer.Serializer is always Pretty=false).
//
// For anything that may produce pretty-printed JSON, or any whitespace
// separator other than `\n`, use NewJSONFramedReader instead.
type lineDelimitedFrameReader struct {
	r       io.ReadCloser
	reader  *bufio.Reader
	inFrame bool // bytes have been emitted for the current frame; the terminating `\n` has not been seen yet
}

// NewLineDelimitedFrameReader returns an io.Reader that splits a stream into
// frames separated by `\n`. The returned reader implements the same Read
// contract as NewJSONFramedReader (one frame per successful call,
// io.ErrShortBuffer for partial frames, io.EOF at clean end of stream), so it
// is a drop-in for callers like streaming.Decoder.
//
// Contract on the input: each frame must be a single line (no embedded raw
// `\n` bytes). A trailing `\n` at end of stream is optional — like
// bufio.ScanLines, a final frame with no terminating newline is returned
// as a complete frame at EOF rather than as an error. Stray empty lines
// between frames are tolerated and skipped. Compact JSON satisfies this;
// pretty-printed JSON does not.
func NewLineDelimitedFrameReader(r io.ReadCloser) io.ReadCloser {
	return &lineDelimitedFrameReader{
		r:      r,
		reader: bufio.NewReader(r),
	}
}

// Read decodes the next line-delimited frame in the stream into data.
// See NewLineDelimitedFrameReader for the contract.
func (r *lineDelimitedFrameReader) Read(data []byte) (int, error) {
	if len(data) == 0 {
		return 0, io.ErrShortBuffer
	}
	n := 0
	for {
		if r.reader.Buffered() == 0 {
			if _, err := r.reader.Peek(1); err != nil {
				if err == io.EOF {
					if r.inFrame {
						// Trailing frame without `\n` — treat as
						// complete (matches bufio.ScanLines).
						r.inFrame = false
						return n, nil
					}
					return 0, io.EOF
				}
				return n, err
			}
		}
		buf, err := r.reader.Peek(r.reader.Buffered())
		if err != nil {
			return n, err
		}

		// Defensively skip stray `\n` between frames — a producer that
		// emits an empty line (or two trailing `\n`s) shouldn't surface
		// as an empty frame to the caller.
		if !r.inFrame {
			skip := 0
			for skip < len(buf) && buf[skip] == '\n' {
				skip++
			}
			if skip > 0 {
				if err := r.discard(skip); err != nil {
					return n, err
				}
				if skip == len(buf) {
					continue
				}
				buf = buf[skip:]
			}
		}

		if j := bytes.IndexByte(buf, '\n'); j >= 0 {
			cp := copy(data[n:], buf[:j])
			n += cp
			if cp < j {
				// data filled before the newline; the frame
				// continues on the next call.
				if err := r.discard(cp); err != nil {
					return n, err
				}
				r.inFrame = true
				return n, io.ErrShortBuffer
			}
			// frame complete; consume buf[:j+1] (drops the `\n`)
			if err := r.discard(j + 1); err != nil {
				return n, err
			}
			r.inFrame = false
			return n, nil
		}

		// no newline in buf; copy what we can and refill
		cp := copy(data[n:], buf)
		n += cp
		if err := r.discard(cp); err != nil {
			return n, err
		}
		r.inFrame = true
		if cp < len(buf) {
			return n, io.ErrShortBuffer
		}
	}
}

func (r *lineDelimitedFrameReader) Close() error {
	return r.r.Close()
}

func (r *lineDelimitedFrameReader) discard(n int) error {
	_, err := r.reader.Discard(n)
	return err
}
