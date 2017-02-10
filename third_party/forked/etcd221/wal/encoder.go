// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package wal

import (
	"bufio"
	"encoding/binary"
	"hash"
	"io"
	"sync"

	"github.com/coreos/etcd/pkg/crc"
	"github.com/coreos/etcd/wal/walpb"
)

type encoder struct {
	mu sync.Mutex
	bw *bufio.Writer

	crc       hash.Hash32
	buf       []byte
	uint64buf []byte
}

func newEncoder(w io.Writer, prevCrc uint32) *encoder {
	return &encoder{
		bw:  bufio.NewWriter(w),
		crc: crc.New(prevCrc, crcTable),
		// 1MB buffer
		buf:       make([]byte, 1024*1024),
		uint64buf: make([]byte, 8),
	}
}

func (e *encoder) encode(rec *walpb.Record) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.crc.Write(rec.Data)
	rec.Crc = e.crc.Sum32()
	var (
		data []byte
		err  error
		n    int
	)

	if rec.Size() > len(e.buf) {
		data, err = rec.Marshal()
		if err != nil {
			return err
		}
	} else {
		n, err = rec.MarshalTo(e.buf)
		if err != nil {
			return err
		}
		data = e.buf[:n]
	}
	if err := writeInt64(e.bw, int64(len(data)), e.uint64buf); err != nil {
		return err
	}
	_, err = e.bw.Write(data)
	return err
}

func (e *encoder) flush() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.bw.Flush()
}

func writeInt64(w io.Writer, n int64, buf []byte) error {
	// http://golang.org/src/encoding/binary/binary.go
	binary.LittleEndian.PutUint64(buf, uint64(n))
	_, err := w.Write(buf)
	return err
}
