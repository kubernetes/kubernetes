// Extensions for Protocol Buffers to create more go like structures.
//
// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
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

package io_test

import (
	"bytes"
	"encoding/binary"
	"github.com/gogo/protobuf/io"
	"github.com/gogo/protobuf/test"
	goio "io"
	"math/rand"
	"testing"
	"time"
)

func iotest(writer io.WriteCloser, reader io.ReadCloser) error {
	size := 1000
	msgs := make([]*test.NinOptNative, size)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range msgs {
		msgs[i] = test.NewPopulatedNinOptNative(r, true)
		//issue 31
		if i == 5 {
			msgs[i] = &test.NinOptNative{}
		}
		//issue 31
		if i == 999 {
			msgs[i] = &test.NinOptNative{}
		}
		err := writer.WriteMsg(msgs[i])
		if err != nil {
			return err
		}
	}
	if err := writer.Close(); err != nil {
		return err
	}
	i := 0
	for {
		msg := &test.NinOptNative{}
		if err := reader.ReadMsg(msg); err != nil {
			if err == goio.EOF {
				break
			}
			return err
		}
		if err := msg.VerboseEqual(msgs[i]); err != nil {
			return err
		}
		i++
	}
	if i != size {
		panic("not enough messages read")
	}
	if err := reader.Close(); err != nil {
		return err
	}
	return nil
}

type buffer struct {
	*bytes.Buffer
	closed bool
}

func (this *buffer) Close() error {
	this.closed = true
	return nil
}

func newBuffer() *buffer {
	return &buffer{bytes.NewBuffer(nil), false}
}

func TestBigUint32Normal(t *testing.T) {
	buf := newBuffer()
	writer := io.NewUint32DelimitedWriter(buf, binary.BigEndian)
	reader := io.NewUint32DelimitedReader(buf, binary.BigEndian, 1024*1024)
	if err := iotest(writer, reader); err != nil {
		t.Error(err)
	}
	if !buf.closed {
		t.Fatalf("did not close buffer")
	}
}

func TestBigUint32MaxSize(t *testing.T) {
	buf := newBuffer()
	writer := io.NewUint32DelimitedWriter(buf, binary.BigEndian)
	reader := io.NewUint32DelimitedReader(buf, binary.BigEndian, 20)
	if err := iotest(writer, reader); err != goio.ErrShortBuffer {
		t.Error(err)
	} else {
		t.Logf("%s", err)
	}
}

func TestLittleUint32Normal(t *testing.T) {
	buf := newBuffer()
	writer := io.NewUint32DelimitedWriter(buf, binary.LittleEndian)
	reader := io.NewUint32DelimitedReader(buf, binary.LittleEndian, 1024*1024)
	if err := iotest(writer, reader); err != nil {
		t.Error(err)
	}
	if !buf.closed {
		t.Fatalf("did not close buffer")
	}
}

func TestLittleUint32MaxSize(t *testing.T) {
	buf := newBuffer()
	writer := io.NewUint32DelimitedWriter(buf, binary.LittleEndian)
	reader := io.NewUint32DelimitedReader(buf, binary.LittleEndian, 20)
	if err := iotest(writer, reader); err != goio.ErrShortBuffer {
		t.Error(err)
	} else {
		t.Logf("%s", err)
	}
}

func TestVarintNormal(t *testing.T) {
	buf := newBuffer()
	writer := io.NewDelimitedWriter(buf)
	reader := io.NewDelimitedReader(buf, 1024*1024)
	if err := iotest(writer, reader); err != nil {
		t.Error(err)
	}
	if !buf.closed {
		t.Fatalf("did not close buffer")
	}
}

func TestVarintNoClose(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writer := io.NewDelimitedWriter(buf)
	reader := io.NewDelimitedReader(buf, 1024*1024)
	if err := iotest(writer, reader); err != nil {
		t.Error(err)
	}
}

//issue 32
func TestVarintMaxSize(t *testing.T) {
	buf := newBuffer()
	writer := io.NewDelimitedWriter(buf)
	reader := io.NewDelimitedReader(buf, 20)
	if err := iotest(writer, reader); err != goio.ErrShortBuffer {
		t.Error(err)
	} else {
		t.Logf("%s", err)
	}
}

func TestVarintError(t *testing.T) {
	buf := newBuffer()
	buf.Write([]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f})
	reader := io.NewDelimitedReader(buf, 1024*1024)
	msg := &test.NinOptNative{}
	err := reader.ReadMsg(msg)
	if err == nil {
		t.Fatalf("Expected error")
	}
}

func TestFull(t *testing.T) {
	buf := newBuffer()
	writer := io.NewFullWriter(buf)
	reader := io.NewFullReader(buf, 1024*1024)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	msgIn := test.NewPopulatedNinOptNative(r, true)
	if err := writer.WriteMsg(msgIn); err != nil {
		panic(err)
	}
	if err := writer.Close(); err != nil {
		panic(err)
	}
	msgOut := &test.NinOptNative{}
	if err := reader.ReadMsg(msgOut); err != nil {
		panic(err)
	}
	if err := msgIn.VerboseEqual(msgOut); err != nil {
		panic(err)
	}
	if err := reader.ReadMsg(msgOut); err != nil {
		if err != goio.EOF {
			panic(err)
		}
	}
	if err := reader.Close(); err != nil {
		panic(err)
	}
	if !buf.closed {
		t.Fatalf("did not close buffer")
	}
}
