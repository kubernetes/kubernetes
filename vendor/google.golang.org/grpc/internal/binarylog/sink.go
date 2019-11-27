/*
 *
 * Copyright 2018 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package binarylog

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	pb "google.golang.org/grpc/binarylog/grpc_binarylog_v1"
	"google.golang.org/grpc/grpclog"
)

var (
	defaultSink Sink = &noopSink{} // TODO(blog): change this default (file in /tmp).
)

// SetDefaultSink sets the sink where binary logs will be written to.
//
// Not thread safe. Only set during initialization.
func SetDefaultSink(s Sink) {
	if defaultSink != nil {
		defaultSink.Close()
	}
	defaultSink = s
}

// Sink writes log entry into the binary log sink.
type Sink interface {
	// Write will be called to write the log entry into the sink.
	//
	// It should be thread-safe so it can be called in parallel.
	Write(*pb.GrpcLogEntry) error
	// Close will be called when the Sink is replaced by a new Sink.
	Close() error
}

type noopSink struct{}

func (ns *noopSink) Write(*pb.GrpcLogEntry) error { return nil }
func (ns *noopSink) Close() error                 { return nil }

// newWriterSink creates a binary log sink with the given writer.
//
// Write() marshalls the proto message and writes it to the given writer. Each
// message is prefixed with a 4 byte big endian unsigned integer as the length.
//
// No buffer is done, Close() doesn't try to close the writer.
func newWriterSink(w io.Writer) *writerSink {
	return &writerSink{out: w}
}

type writerSink struct {
	out io.Writer
}

func (ws *writerSink) Write(e *pb.GrpcLogEntry) error {
	b, err := proto.Marshal(e)
	if err != nil {
		grpclog.Infof("binary logging: failed to marshal proto message: %v", err)
	}
	hdr := make([]byte, 4)
	binary.BigEndian.PutUint32(hdr, uint32(len(b)))
	if _, err := ws.out.Write(hdr); err != nil {
		return err
	}
	if _, err := ws.out.Write(b); err != nil {
		return err
	}
	return nil
}

func (ws *writerSink) Close() error { return nil }

type bufWriteCloserSink struct {
	mu     sync.Mutex
	closer io.Closer
	out    *writerSink   // out is built on buf.
	buf    *bufio.Writer // buf is kept for flush.

	writeStartOnce sync.Once
	writeTicker    *time.Ticker
}

func (fs *bufWriteCloserSink) Write(e *pb.GrpcLogEntry) error {
	// Start the write loop when Write is called.
	fs.writeStartOnce.Do(fs.startFlushGoroutine)
	fs.mu.Lock()
	if err := fs.out.Write(e); err != nil {
		fs.mu.Unlock()
		return err
	}
	fs.mu.Unlock()
	return nil
}

const (
	bufFlushDuration = 60 * time.Second
)

func (fs *bufWriteCloserSink) startFlushGoroutine() {
	fs.writeTicker = time.NewTicker(bufFlushDuration)
	go func() {
		for range fs.writeTicker.C {
			fs.mu.Lock()
			fs.buf.Flush()
			fs.mu.Unlock()
		}
	}()
}

func (fs *bufWriteCloserSink) Close() error {
	if fs.writeTicker != nil {
		fs.writeTicker.Stop()
	}
	fs.mu.Lock()
	fs.buf.Flush()
	fs.closer.Close()
	fs.out.Close()
	fs.mu.Unlock()
	return nil
}

func newBufWriteCloserSink(o io.WriteCloser) Sink {
	bufW := bufio.NewWriter(o)
	return &bufWriteCloserSink{
		closer: o,
		out:    newWriterSink(bufW),
		buf:    bufW,
	}
}

// NewTempFileSink creates a temp file and returns a Sink that writes to this
// file.
func NewTempFileSink() (Sink, error) {
	tempFile, err := ioutil.TempFile("/tmp", "grpcgo_binarylog_*.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %v", err)
	}
	return newBufWriteCloserSink(tempFile), nil
}
