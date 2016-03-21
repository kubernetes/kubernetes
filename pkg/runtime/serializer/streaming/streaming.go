/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// Framer is a factory for creating readers and writers that obey a particular framing pattern.
type Framer interface {
	NewFrameReader(r io.Reader) io.Reader
	NewFrameWriter(w io.Writer) io.Writer
}

// Encoder is a runtime.Encoder on a stream.
type Encoder interface {
	// Encode will write the provided object to the stream or return an error. It obeys the same
	// contract as runtime.Encoder.
	Encode(obj runtime.Object, overrides ...unversioned.GroupVersion) error
}

// Decoder is a runtime.Decoder from a stream.
type Decoder interface {
	// Decode will return io.EOF when no more objects are available.
	Decode(defaults *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error)
}

// Serializer is a factory for creating encoders and decoders that work over streams.
type Serializer interface {
	NewEncoder(w io.Writer) Encoder
	NewDecoder(r io.Reader) Decoder
}

type decoder struct {
	reader  io.Reader
	decoder runtime.Decoder
	buf     []byte
}

// NewDecoder creates a streaming decoder that reads object chunks from r and decodes them with d.
// The reader is expected to return ErrShortRead if the provided buffer is not large enough to read
// an entire object.
func NewDecoder(r io.Reader, d runtime.Decoder) Decoder {
	return &decoder{
		reader:  r,
		decoder: d,
		buf:     make([]byte, 1024*1024),
	}
}

// Decode reads the next object from the stream and decodes it.
func (d *decoder) Decode(defaults *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	// TODO: instead of depending on a fixed sized buffer, we should handle ErrShortRead specially and
	// grow the buffer capacity up to a maximum amount. Requires the framer to allow repeated reads to
	// the stream until the frame is finished.
	n, err := d.reader.Read(d.buf)
	if err != nil {
		return nil, nil, err
	}
	return d.decoder.Decode(d.buf[:n], defaults, into)
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
func (e *encoder) Encode(obj runtime.Object, overrides ...unversioned.GroupVersion) error {
	if err := e.encoder.EncodeToStream(obj, e.buf, overrides...); err != nil {
		return err
	}
	_, err := e.writer.Write(e.buf.Bytes())
	e.buf.Reset()
	return err
}
