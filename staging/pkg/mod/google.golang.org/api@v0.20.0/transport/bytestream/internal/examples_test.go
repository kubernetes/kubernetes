// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"bytes"
	"context"
	"io"
	"log"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

type ExampleReadHandler struct {
	buf  []byte
	name string // In this example, the service can handle one name only.
}

func (mr *ExampleReadHandler) GetReader(ctx context.Context, name string) (io.ReaderAt, error) {
	if mr.name == "" {
		mr.name = name
		log.Printf("read from name: %q", name)
	} else if mr.name != name {
		return nil, grpc.Errorf(codes.NotFound, "reader has name %q, name %q not allowed", mr.name, name)
	}
	return bytes.NewReader(mr.buf), nil
}

// Close can be a no-op.
func (mr *ExampleReadHandler) Close(ctx context.Context, name string) error {
	return nil
}

type ExampleWriteHandler struct {
	buf  bytes.Buffer // bytes.Buffer implements io.Writer
	name string       // In this example, the service can handle one name only.
}

// Handle writes to a given name.
func (mw *ExampleWriteHandler) GetWriter(ctx context.Context, name string, initOffset int64) (io.Writer, error) {
	if mw.name == "" {
		mw.name = name
		log.Printf("write to name: %q", name)
	} else if mw.name != name {
		return nil, grpc.Errorf(codes.NotFound, "reader has name %q, name=%q not allowed", mw.name, name)
	}
	// TODO: initOffset is ignored.
	return &mw.buf, nil
}

// Close can be a no-op.
func (mw *ExampleWriteHandler) Close(ctx context.Context, name string) error {
	return nil
}

func ExampleNewServer() {
	reader := &ExampleReadHandler{
		buf:  []byte("Hello World!"),
		name: "foo",
	}
	writer := &ExampleWriteHandler{}
	gsrv := grpc.NewServer()
	bytestreamServer, err := NewServer(gsrv, reader, writer)
	if err != nil {
		log.Printf("NewServer: %v", err)
		return
	}

	// Start accepting incoming connections.
	// See gRPC docs and newGRPCServer in google.golang.org/api/transport/bytestream/client_test.go.
	_ = bytestreamServer
}
