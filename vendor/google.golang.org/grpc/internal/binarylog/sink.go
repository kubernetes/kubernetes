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
	"io"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	binlogpb "google.golang.org/grpc/binarylog/grpc_binarylog_v1"
)

var (
	// DefaultSink is the sink where the logs will be written to. It's exported
	// for the binarylog package to update.
	DefaultSink Sink = &noopSink{} // TODO(blog): change this default (file in /tmp).
)

// Sink writes log entry into the binary log sink.
//
// sink is a copy of the exported binarylog.Sink, to avoid circular dependency.
type Sink interface {
	// Write will be called to write the log entry into the sink.
	//
	// It should be thread-safe so it can be called in parallel.
	Write(*binlogpb.GrpcLogEntry) error
	// Close will be called when the Sink is replaced by a new Sink.
	Close() error
}

type noopSink struct{}

func (ns *noopSink) Write(*binlogpb.GrpcLogEntry) error { return nil }
func (ns *noopSink) Close() error                       { return nil }

// newWriterSink creates a binary log sink with the given writer.
//
// Write() marshals the proto message and writes it to the given writer. Each
// message is prefixed with a 4 byte big endian unsigned integer as the length.
//
// No buffer is done, Close() doesn't try to close the writer.
func newWriterSink(w io.Writer) Sink {
	return &writerSink{out: w}
}

type writerSink struct {
	out io.Writer
}

func (ws *writerSink) Write(e *binlogpb.GrpcLogEntry) error {
	b, err := proto.Marshal(e)
	if err != nil {
		grpclogLogger.Errorf("binary logging: failed to marshal proto message: %v", err)
		return err
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

type bufferedSink struct {
	mu             sync.Mutex
	closer         io.Closer
	out            Sink          // out is built on buf.
	buf            *bufio.Writer // buf is kept for flush.
	flusherStarted bool

	writeTicker *time.Ticker
	done        chan struct{}
}

func (fs *bufferedSink) Write(e *binlogpb.GrpcLogEntry) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if !fs.flusherStarted {
		// Start the write loop when Write is called.
		fs.startFlushGoroutine()
		fs.flusherStarted = true
	}
	if err := fs.out.Write(e); err != nil {
		return err
	}
	return nil
}

const (
	bufFlushDuration = 60 * time.Second
)

func (fs *bufferedSink) startFlushGoroutine() {
	fs.writeTicker = time.NewTicker(bufFlushDuration)
	go func() {
		for {
			select {
			case <-fs.done:
				return
			case <-fs.writeTicker.C:
			}
			fs.mu.Lock()
			if err := fs.buf.Flush(); err != nil {
				grpclogLogger.Warningf("failed to flush to Sink: %v", err)
			}
			fs.mu.Unlock()
		}
	}()
}

func (fs *bufferedSink) Close() error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if fs.writeTicker != nil {
		fs.writeTicker.Stop()
	}
	close(fs.done)
	if err := fs.buf.Flush(); err != nil {
		grpclogLogger.Warningf("failed to flush to Sink: %v", err)
	}
	if err := fs.closer.Close(); err != nil {
		grpclogLogger.Warningf("failed to close the underlying WriterCloser: %v", err)
	}
	if err := fs.out.Close(); err != nil {
		grpclogLogger.Warningf("failed to close the Sink: %v", err)
	}
	return nil
}

// NewBufferedSink creates a binary log sink with the given WriteCloser.
//
// Write() marshals the proto message and writes it to the given writer. Each
// message is prefixed with a 4 byte big endian unsigned integer as the length.
//
// Content is kept in a buffer, and is flushed every 60 seconds.
//
// Close closes the WriteCloser.
func NewBufferedSink(o io.WriteCloser) Sink {
	bufW := bufio.NewWriter(o)
	return &bufferedSink{
		closer: o,
		out:    newWriterSink(bufW),
		buf:    bufW,
		done:   make(chan struct{}),
	}
}
