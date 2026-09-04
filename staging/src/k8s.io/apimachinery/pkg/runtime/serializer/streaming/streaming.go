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

// Package streaming implements encoder and decoder for streams
// of runtime.Objects over io.Writer/Readers.
package streaming

import (
	"bytes"
	"fmt"
	"io"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Encoder is a runtime.Encoder on a stream.
type Encoder interface {
	// Encode will write the provided object to the stream or return an error. It obeys the same
	// contract as runtime.VersionedEncoder.
	Encode(obj runtime.Object) error
}

// Decoder is a runtime.Decoder from a stream.
type Decoder interface {
	// Decode will return io.EOF when no more objects are available.
	Decode(defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error)
	// Close closes the underlying stream.
	Close() error
}

type Framer interface {
	// ReadFrame will return io.EOF when no more frames are available. The
	// returned Frame must be released when the caller is done with it.
	ReadFrame() (*Frame, error)
	Close() error
}

// Serializer is a factory for creating encoders and decoders that work over streams.
type Serializer interface {
	NewEncoder(w io.Writer) Encoder
	NewDecoder(r io.ReadCloser) Decoder
}

type decoder struct {
	reader    io.ReadCloser
	decoder   runtime.Decoder
	maxBytes  int
	resetRead bool
	pool      sync.Pool
}

// NewDecoder creates a streaming decoder that reads object chunks from r and decodes them with d.
// The reader is expected to return ErrShortRead if the provided buffer is not large enough to read
// an entire object.
func NewDecoder(r io.ReadCloser, d runtime.Decoder) Decoder {
	decoder := &decoder{
		reader:   r,
		decoder:  d,
		maxBytes: 16 * 1024 * 1024,
	}
	decoder.pool.New = func() any {
		return make([]byte, 1024)
	}
	return decoder
}

var ErrObjectTooLarge = fmt.Errorf("object to decode was longer than maximum allowed size")

// Decode reads the next object from the stream and decodes it.
func (d *decoder) Decode(defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	frame, err := d.ReadFrame()
	if err != nil {
		return nil, nil, err
	}
	defer frame.Release()
	return d.decoder.Decode(frame.Data(), defaults, into)
}

// ReadFrame reads the next raw object frame from the stream.
func (d *decoder) ReadFrame() (*Frame, error) {
	buf := d.getBuffer()
	base := 0
	for {
		n, err := d.reader.Read(buf[base:])
		if err == io.ErrShortBuffer {
			if n == 0 {
				d.putBuffer(buf)
				return nil, fmt.Errorf("got short buffer with n=0, base=%d, cap=%d", base, cap(buf))
			}
			if d.resetRead {
				continue
			}
			// double the buffer size up to maxBytes
			if len(buf) < d.maxBytes {
				base += n
				newLen := len(buf) * 2
				if newLen > d.maxBytes {
					newLen = d.maxBytes
				}
				buf = append(buf, make([]byte, newLen-len(buf))...)
				continue
			}
			// must read the rest of the frame (until we stop getting ErrShortBuffer)
			d.resetRead = true
			d.putBuffer(buf)
			return nil, ErrObjectTooLarge
		}
		if err != nil {
			d.putBuffer(buf)
			return nil, err
		}
		if d.resetRead {
			// now that we have drained the large read, continue
			d.resetRead = false
			continue
		}
		base += n
		break
	}

	return &Frame{data: buf[:base], release: d.putBuffer}, nil
}

type Frame struct {
	data []byte

	once    sync.Once
	release func(buf []byte)
}

func (f *Frame) Data() []byte {
	return f.data
}

func (f *Frame) Release() {
	if f == nil {
		return
	}
	f.once.Do(func() {
		if f.release != nil {
			f.release(f.data)
		}
		f.data = nil
		f.release = nil
	})
}

func (d *decoder) Close() error {
	return d.reader.Close()
}

func (d *decoder) getBuffer() []byte {
	return d.pool.Get().([]byte)
}

func (d *decoder) putBuffer(buf []byte) {
	if cap(buf) > d.maxBytes {
		return
	}
	d.pool.Put(buf[:cap(buf)])
}

type encoder struct {
	writer  io.Writer
	encoder runtime.Encoder
	buf     *bytes.Buffer
}

// NewEncoder returns a new streaming encoder.
func NewEncoder(w io.Writer, e runtime.Encoder) Encoder {
	return &encoder{
		writer:  w,
		encoder: e,
		buf:     &bytes.Buffer{},
	}
}

// Encode writes the provided object to the nested writer.
func (e *encoder) Encode(obj runtime.Object) error {
	if err := e.encoder.Encode(obj, e.buf); err != nil {
		return err
	}
	_, err := e.writer.Write(e.buf.Bytes())
	e.buf.Reset()
	return err
}
