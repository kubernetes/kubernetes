// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytestream

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"testing"

	"google.golang.org/api/transport/bytestream/internal"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	pb "google.golang.org/genproto/googleapis/bytestream"
)

const testData = "0123456789"

// A grpcServer is an in-process gRPC server, listening on a system-chosen port on
// the local loopback interface. Servers are for testing only and are not
// intended to be used in production code.
// (Copied from "cloud.google.com/internal/testutil/server_test.go")
//
// To create a server, make a new grpcServer, register your handlers, then call
// Start:
//
//	srv, err := NewServer()
//	...
//	mypb.RegisterMyServiceServer(srv.Gsrv, &myHandler)
//	....
//	srv.Start()
//
// Clients should connect to the server with no security:
//
//	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
//	...
type grpcServer struct {
	Addr string
	l    net.Listener
	Gsrv *grpc.Server
}

type TestSetup struct {
	ctx     context.Context
	rpcTest *grpcServer
	server  *internal.Server
	client  *Client
}

func TestClientRead(t *testing.T) {
	testCases := []struct {
		name         string
		input        string
		resourceName string
		extraBufSize int
		extraError   error
		want         string
		wantErr      bool
		wantEOF      bool
	}{
		{
			name:         "test foo",
			input:        testData,
			resourceName: "foo",
			want:         testData,
		}, {
			name:         "test bar",
			input:        testData,
			resourceName: "bar",
			want:         testData,
		}, {
			name:         "test bar extraBufSize=1",
			input:        testData,
			resourceName: "bar",
			extraBufSize: 1,
			want:         testData,
			wantEOF:      true,
		}, {
			name:         "test bar extraBufSize=2",
			input:        testData,
			resourceName: "bar",
			extraBufSize: 2,
			want:         testData,
			wantEOF:      true,
		}, {
			name:         "empty resource name",
			input:        testData,
			resourceName: "",
			extraBufSize: 1,
			wantErr:      true,
		}, {
			name:         "read after error returns error again",
			input:        testData,
			resourceName: "does not matter",
			extraBufSize: 1,
			extraError:   errors.New("some error"),
			wantErr:      true,
		},
	}

	for _, tc := range testCases {
		bufSize := len(tc.want) + tc.extraBufSize
		if bufSize == 0 {
			t.Errorf("%s: This is probably wrong. Read returning 0 bytes?", tc.name)
			continue
		}

		setup := newTestSetup(tc.input)
		r, err := setup.client.NewReader(setup.ctx, tc.resourceName)
		if err != nil {
			t.Errorf("%s: NewReader(%q): %v", tc.name, tc.resourceName, err)
			continue
		}
		if tc.extraError != nil {
			r.err = tc.extraError
		}
		buf := make([]byte, bufSize)
		gotEOF := false
		total := 0
		for total < bufSize && err == nil {
			var n int
			n, err = r.Read(buf[total:])
			total += n
		}
		if err == io.EOF {
			gotEOF = true
			err = nil
			doubleCheckBuf := make([]byte, bufSize)
			n2, err2 := r.Read(doubleCheckBuf)
			if err2 != io.EOF {
				t.Errorf("%s: read and got EOF, double-check: read %d bytes got err=%v", tc.name, n2, err2)
				continue
			}
		}
		setup.Close()

		if gotErr := err != nil; tc.wantErr != gotErr {
			t.Errorf("%s: read %d bytes, got err=%v, wantErr=%t", tc.name, total, err, tc.wantErr)
			continue
		}
		if tc.wantEOF != gotEOF {
			t.Errorf("%s: read %d bytes, gotEOF=%t, wantEOF=%t", tc.name, total, gotEOF, tc.wantEOF)
			continue
		}
		if got := string(buf[:total]); got != tc.want {
			t.Errorf("%s: read %q, want %q", tc.name, got, tc.want)
			continue
		}
	}
}

func TestClientWrite(t *testing.T) {
	testCases := []struct {
		name         string
		resourceName string
		data         string
		results      []int
		wantWriteErr bool
		wantCloseErr bool
	}{
		{
			name:         "test foo",
			resourceName: "foo",
			data:         testData,
			results:      []int{len(testData)},
		}, {
			name:         "empty resource name",
			resourceName: "",
			data:         testData,
			results:      []int{10},
			//wantWriteErr: true,
			wantCloseErr: true,
		}, {
			name:         "test bar",
			resourceName: "bar",
			data:         testData,
			results:      []int{len(testData)},
		},
	}

	var setup *TestSetup

tcFor:
	for _, tc := range testCases {
		if setup != nil {
			setup.Close()
		}
		setup = newTestSetup("")
		buf := []byte(tc.data)
		var ofs int
		w, err := setup.client.NewWriter(setup.ctx, tc.resourceName)
		if err != nil {
			t.Errorf("%s: NewWriter(): %v", tc.name, err)
			continue
		}

		for i := 0; i < len(tc.results); i++ {
			if ofs >= len(tc.data) {
				t.Errorf("%s [%d]: Attempting to write more than tc.input: ofs=%d len(buf)=%d",
					tc.name, i, ofs, len(tc.data))
				continue tcFor
			}
			n, err := w.Write(buf[ofs:])
			ofs += n
			if gotErr := err != nil; gotErr != tc.wantWriteErr {
				t.Errorf("%s [%d]: Write() got n=%d err=%v, wantWriteErr=%t", tc.name, i, n, err, tc.wantWriteErr)
				continue tcFor
			} else if tc.wantWriteErr && i+1 < len(tc.results) {
				t.Errorf("%s: wantWriteErr and got err after %d results, len(results)=%d is too long.", tc.name, i+1, len(tc.results))
				continue tcFor
			}
			if n != tc.results[i] {
				t.Errorf("%s [%d]: Write() wrote %d bytes, want %d bytes", tc.name, i, n, tc.results[i])
				continue tcFor
			}
		}

		err = w.Close()
		if gotErr := err != nil; gotErr != tc.wantCloseErr {
			t.Errorf("%s: Close() got err=%v, wantCloseErr=%t", tc.name, err, tc.wantCloseErr)
			continue tcFor
		}
	}
	setup.Close()
}

func TestClientRead_AfterSetupClose(t *testing.T) {
	setup := newTestSetup("closed")
	setup.Close()
	_, err := setup.client.NewReader(setup.ctx, "should fail")
	if err == nil {
		t.Errorf("NewReader(%q): err=%v", "should fail", err)
	}
}

func TestClientWrite_AfterSetupClose(t *testing.T) {
	setup := newTestSetup("closed")
	setup.Close()
	_, err := setup.client.NewWriter(setup.ctx, "should fail")
	if err == nil {
		t.Fatalf("NewWriter(%q): err=%v", "shoudl fail", err)
	}
}

type UnsendableWriteClient struct {
	closeAndRecvWriteResponse *pb.WriteResponse
	closeAndRecvError         error
}

func (w *UnsendableWriteClient) Send(*pb.WriteRequest) error {
	if w.closeAndRecvError != nil {
		return nil
	}
	return errors.New("UnsendableWriteClient.Send() fails unless closeAndRecvError is set")
}

func (w *UnsendableWriteClient) CloseAndRecv() (*pb.WriteResponse, error) {
	if w.closeAndRecvError == nil {
		log.Fatalf("UnsendableWriteClient.Close() when closeAndRecvError == nil.")
	}
	return w.closeAndRecvWriteResponse, w.closeAndRecvError
}

func (w *UnsendableWriteClient) Context() context.Context {
	log.Fatalf("UnsendableWriteClient.Context() should never be called")
	return context.Background()
}
func (w *UnsendableWriteClient) CloseSend() error {
	return errors.New("UnsendableWriteClient.CloseSend() should never be called")
}
func (w *UnsendableWriteClient) Header() (metadata.MD, error) {
	log.Fatalf("UnsendableWriteClient.Header() should never be called")
	return metadata.MD{}, nil
}
func (w *UnsendableWriteClient) Trailer() metadata.MD {
	log.Fatalf("UnsendableWriteClient.Trailer() should never be called")
	return metadata.MD{}
}
func (w *UnsendableWriteClient) SendMsg(m interface{}) error {
	log.Fatalf("UnsendableWriteClient.SendMsg() should never be called")
	return nil
}
func (w *UnsendableWriteClient) RecvMsg(m interface{}) error {
	log.Fatalf("UnsendableWriteClient.RecvMsg() should never be called")
	return nil
}

func TestClientWrite_WriteFails(t *testing.T) {
	setup := newTestSetup("")
	w, err := setup.client.NewWriter(setup.ctx, "")
	if err != nil {
		t.Fatalf("NewWriter(): %v", err)
	}
	defer setup.Close()
	w.writeClient = &UnsendableWriteClient{}
	_, err = w.Write([]byte(testData))
	if err == nil {
		t.Errorf("Write() should fail")
	}
}

func TestClientWrite_CloseAndRecvFails(t *testing.T) {
	setup := newTestSetup("")
	w, err := setup.client.NewWriter(setup.ctx, "CloseAndRecvFails")
	if err != nil {
		t.Fatalf("NewWriter(): %v", err)
	}
	defer setup.Close()
	n, err := w.Write([]byte(testData))
	if err != nil {
		t.Errorf("Write() failed: %v", err)
		return
	}
	if n != len(testData) {
		t.Errorf("Write() got n=%d, want n=%d", n, len(testData))
		return
	}
	w.writeClient = &UnsendableWriteClient{
		closeAndRecvError: errors.New("CloseAndRecv() must fail"),
	}
	if err = w.Close(); err == nil {
		t.Errorf("Close() should fail")
		return
	}
}

type TestWriteHandler struct {
	buf  bytes.Buffer // bytes.Buffer implements io.Writer
	name string       // This service can handle one name only.
}

func (w *TestWriteHandler) GetWriter(ctx context.Context, name string, initOffset int64) (io.Writer, error) {
	if w.name == "" {
		w.name = name
	} else if w.name != name {
		return nil, fmt.Errorf("writer already has name=%q, now a new name=%q confuses me", w.name, name)
	}
	// initOffset is ignored.
	return &w.buf, nil
}

func (w *TestWriteHandler) Close(ctx context.Context, name string) error {
	w.name = ""
	w.buf.Reset()
	return nil
}

type TestReadHandler struct {
	buf  string
	name string // This service can handle one name only.
}

// GetWriter() returns an io.ReaderAt to accept reads from the given name.
func (r *TestReadHandler) GetReader(ctx context.Context, name string) (io.ReaderAt, error) {
	if r.name == "" {
		r.name = name
	} else if r.name != name {
		return nil, fmt.Errorf("reader already has name=%q, now a new name=%q confuses me", r.name, name)
	}
	return bytes.NewReader([]byte(r.buf)), nil
}

// Close does nothing.
func (r *TestReadHandler) Close(ctx context.Context, name string) error {
	return nil
}

// newGRPCServer creates a new grpcServer. The grpcServer will be listening for gRPC connections
// at the address named by the Addr field, without TLS.
func newGRPCServer() (*grpcServer, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	s := &grpcServer{
		Addr: l.Addr().String(),
		l:    l,
		Gsrv: grpc.NewServer(),
	}
	return s, nil
}

// Start causes the server to start accepting incoming connections.
// Call Start after registering handlers.
func (s *grpcServer) Start() {
	go s.Gsrv.Serve(s.l)
}

// Close shuts down the server.
func (s *grpcServer) Close() {
	s.Gsrv.Stop()
	s.l.Close()
}

func newTestSetup(input string) *TestSetup {
	testSetup := &TestSetup{
		ctx: context.Background(),
	}
	testReadHandler := &TestReadHandler{
		buf: input,
	}
	var err error
	if testSetup.rpcTest, err = newGRPCServer(); err != nil {
		log.Fatalf("newGRPCServer: %v", err)
	}
	if testSetup.server, err = internal.NewServer(testSetup.rpcTest.Gsrv, testReadHandler, &TestWriteHandler{}); err != nil {
		log.Fatalf("internal.NewServer: %v", err)
	}
	testSetup.rpcTest.Start()

	conn, err := grpc.Dial(testSetup.rpcTest.Addr, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("grpc.Dial: %v", err)
	}
	testSetup.client = NewClient(conn, grpc.FailFast(true))
	return testSetup
}

func (testSetup *TestSetup) Close() {
	testSetup.rpcTest.Close()
}
