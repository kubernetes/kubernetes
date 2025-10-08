// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package io

import (
	"encoding/binary"
	"io"

	"github.com/gogo/protobuf/proto"
)

const uint32BinaryLen = 4

func NewUint32DelimitedWriter(w io.Writer, byteOrder binary.ByteOrder) WriteCloser {
	return &uint32Writer{w, byteOrder, nil, make([]byte, uint32BinaryLen)}
}

func NewSizeUint32DelimitedWriter(w io.Writer, byteOrder binary.ByteOrder, size int) WriteCloser {
	return &uint32Writer{w, byteOrder, make([]byte, size), make([]byte, uint32BinaryLen)}
}

type uint32Writer struct {
	w         io.Writer
	byteOrder binary.ByteOrder
	buffer    []byte
	lenBuf    []byte
}

func (this *uint32Writer) writeFallback(msg proto.Message) error {
	data, err := proto.Marshal(msg)
	if err != nil {
		return err
	}

	length := uint32(len(data))
	this.byteOrder.PutUint32(this.lenBuf, length)
	if _, err = this.w.Write(this.lenBuf); err != nil {
		return err
	}
	_, err = this.w.Write(data)
	return err
}

func (this *uint32Writer) WriteMsg(msg proto.Message) error {
	m, ok := msg.(marshaler)
	if !ok {
		return this.writeFallback(msg)
	}

	n, ok := getSize(m)
	if !ok {
		return this.writeFallback(msg)
	}

	size := n + uint32BinaryLen
	if size > len(this.buffer) {
		this.buffer = make([]byte, size)
	}

	this.byteOrder.PutUint32(this.buffer, uint32(n))
	if _, err := m.MarshalTo(this.buffer[uint32BinaryLen:]); err != nil {
		return err
	}

	_, err := this.w.Write(this.buffer[:size])
	return err
}

func (this *uint32Writer) Close() error {
	if closer, ok := this.w.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

type uint32Reader struct {
	r         io.Reader
	byteOrder binary.ByteOrder
	lenBuf    []byte
	buf       []byte
	maxSize   int
}

func NewUint32DelimitedReader(r io.Reader, byteOrder binary.ByteOrder, maxSize int) ReadCloser {
	return &uint32Reader{r, byteOrder, make([]byte, 4), nil, maxSize}
}

func (this *uint32Reader) ReadMsg(msg proto.Message) error {
	if _, err := io.ReadFull(this.r, this.lenBuf); err != nil {
		return err
	}
	length32 := this.byteOrder.Uint32(this.lenBuf)
	length := int(length32)
	if length < 0 || length > this.maxSize {
		return io.ErrShortBuffer
	}
	if length > len(this.buf) {
		this.buf = make([]byte, length)
	}
	_, err := io.ReadFull(this.r, this.buf[:length])
	if err != nil {
		return err
	}
	return proto.Unmarshal(this.buf[:length], msg)
}

func (this *uint32Reader) Close() error {
	if closer, ok := this.r.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
