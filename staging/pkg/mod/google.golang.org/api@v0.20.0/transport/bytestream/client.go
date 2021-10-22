// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bytestream provides a client for any service that exposes a ByteStream API.
//
// Note: This package is a work-in-progress.  Backwards-incompatible changes should be expected.
package bytestream

// This file contains the client implementation of Bytestream declared at:
// https://github.com/googleapis/googleapis/blob/master/google/bytestream/bytestream.proto

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"google.golang.org/grpc"

	pb "google.golang.org/genproto/googleapis/bytestream"
)

const (
	// MaxBufSize is the maximum buffer size (in bytes) received in a read chunk or sent in a write chunk.
	MaxBufSize  = 2 * 1024 * 1024
	backoffBase = 10 * time.Millisecond
	backoffMax  = 1 * time.Second
	maxTries    = 5
)

// Client is the go wrapper around a ByteStreamClient and provides an interface to it.
type Client struct {
	client  pb.ByteStreamClient
	options []grpc.CallOption
}

// NewClient creates a new bytestream.Client.
func NewClient(cc *grpc.ClientConn, options ...grpc.CallOption) *Client {
	return &Client{
		client:  pb.NewByteStreamClient(cc),
		options: options,
	}
}

// Reader reads from a byte stream.
type Reader struct {
	ctx          context.Context
	c            *Client
	readClient   pb.ByteStream_ReadClient
	resourceName string
	err          error
	buf          []byte
}

// ResourceName gets the resource name this Reader is reading.
func (r *Reader) ResourceName() string {
	return r.resourceName
}

// Read implements io.Reader.
// Read buffers received bytes that do not fit in p.
func (r *Reader) Read(p []byte) (int, error) {
	if r.err != nil {
		return 0, r.err
	}

	var backoffDelay time.Duration
	for tries := 0; len(r.buf) == 0 && tries < maxTries; tries++ {
		// No data in buffer.
		resp, err := r.readClient.Recv()
		if err != nil {
			r.err = err
			return 0, err
		}
		r.buf = resp.Data
		if len(r.buf) != 0 {
			break
		}

		// back off
		if backoffDelay < backoffBase {
			backoffDelay = backoffBase
		} else {
			backoffDelay = time.Duration(float64(backoffDelay) * 1.3 * (1 - 0.4*rand.Float64()))
		}
		if backoffDelay > backoffMax {
			backoffDelay = backoffMax
		}
		select {
		case <-time.After(backoffDelay):
		case <-r.ctx.Done():
			if err := r.ctx.Err(); err != nil {
				r.err = err
			}
			return 0, r.err
		}
	}

	// Copy from buffer.
	n := copy(p, r.buf)
	r.buf = r.buf[n:]
	return n, nil
}

// Close implements io.Closer.
func (r *Reader) Close() error {
	if r.readClient == nil {
		return nil
	}
	err := r.readClient.CloseSend()
	r.readClient = nil
	return err
}

// NewReader creates a new Reader to read a resource.
func (c *Client) NewReader(ctx context.Context, resourceName string) (*Reader, error) {
	return c.NewReaderAt(ctx, resourceName, 0)
}

// NewReaderAt creates a new Reader to read a resource from the given offset.
func (c *Client) NewReaderAt(ctx context.Context, resourceName string, offset int64) (*Reader, error) {
	// readClient is set up for Read(). ReadAt() will copy needed fields into its reentrantReader.
	readClient, err := c.client.Read(ctx, &pb.ReadRequest{
		ResourceName: resourceName,
		ReadOffset:   offset,
	}, c.options...)
	if err != nil {
		return nil, err
	}

	return &Reader{
		ctx:          ctx,
		c:            c,
		resourceName: resourceName,
		readClient:   readClient,
	}, nil
}

// Writer writes to a byte stream.
type Writer struct {
	ctx          context.Context
	writeClient  pb.ByteStream_WriteClient
	resourceName string
	offset       int64
	err          error
}

// ResourceName gets the resource name this Writer is writing.
func (w *Writer) ResourceName() string {
	return w.resourceName
}

// Write implements io.Writer.
func (w *Writer) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}

	n := 0
	for n < len(p) {
		bufSize := len(p) - n
		if bufSize > MaxBufSize {
			bufSize = MaxBufSize
		}
		r := pb.WriteRequest{
			WriteOffset: w.offset,
			FinishWrite: false,
			Data:        p[n : n+bufSize],
		}
		// Bytestream only requires the resourceName to be sent in the first WriteRequest.
		if w.offset == 0 {
			r.ResourceName = w.resourceName
		}
		err := w.writeClient.Send(&r)
		if err != nil {
			w.err = err
			return n, err
		}
		w.offset += int64(bufSize)
		n += bufSize
	}
	return n, nil
}

// Close implements io.Closer. It is the caller's responsibility to call Close() when writing is done.
func (w *Writer) Close() error {
	err := w.writeClient.Send(&pb.WriteRequest{
		ResourceName: w.resourceName,
		WriteOffset:  w.offset,
		FinishWrite:  true,
		Data:         nil,
	})
	if err != nil {
		w.err = err
		return fmt.Errorf("Send(WriteRequest< FinishWrite >) failed: %v", err)
	}
	resp, err := w.writeClient.CloseAndRecv()
	if err != nil {
		w.err = err
		return fmt.Errorf("CloseAndRecv: %v", err)
	}
	if resp == nil {
		err = fmt.Errorf("expected a response on close, got %v", resp)
	} else if resp.CommittedSize != w.offset {
		err = fmt.Errorf("server only wrote %d bytes, want %d", resp.CommittedSize, w.offset)
	}
	w.err = err
	return err
}

// NewWriter creates a new Writer to write a resource.
//
// resourceName specifies the name of the resource.
// The resource will be available after Close has been called.
//
// It is the caller's responsibility to call Close when writing is done.
//
// TODO: There is currently no way to resume a write. Maybe NewWriter should begin with a call to QueryWriteStatus.
func (c *Client) NewWriter(ctx context.Context, resourceName string) (*Writer, error) {
	wc, err := c.client.Write(ctx, c.options...)
	if err != nil {
		return nil, err
	}
	return &Writer{
		ctx:          ctx,
		writeClient:  wc,
		resourceName: resourceName,
	}, nil
}
