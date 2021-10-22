// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"sync"
	"testing"

	"google.golang.org/grpc"

	pb "google.golang.org/genproto/googleapis/bytestream"
)

const (
	testName = "testName"
	testData = "0123456789"
)

var (
	setupServerOnce sync.Once
	server          *Server
)

func TestNewServerWithInvalidInputs(t *testing.T) {
	_, err := NewServer(grpc.NewServer(), nil, nil)
	if err == nil {
		t.Fatal("NewServer(nil, nil) should not succeed")
	}
}

func TestServerWrite(t *testing.T) {
	testCases := []struct {
		name              string
		writeHandler      WriteHandler
		input             []interface{}
		writeCount        int
		allowEmptyCommits bool
		allowOverwrite    bool
		wantErr           bool
		wantResponse      int
	}{
		{
			name:         "empty resource name",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					FinishWrite: true,
					Data:        []byte(testData),
				},
			},
			writeCount:   1,
			wantErr:      true,
			wantResponse: 0,
		}, {
			name:         "Recv returns io.EOF",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				io.EOF,
			},
			writeCount:   1,
			wantErr:      false,
			wantResponse: 0,
		}, {
			name:         "Recv returns error, 0 WriteRequests",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				errors.New("Recv returns error, 0 WriteRequests"),
			},
			writeCount:   1,
			wantErr:      true,
			wantResponse: 0,
		}, {
			name:         "simple test",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
				io.EOF,
			},
			writeCount:   1,
			wantResponse: 1,
		}, {
			name:         "Recv returns error, 1 WriteRequests",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  false,
					Data:         []byte(testData),
				},
				errors.New("Recv returns error, 1 WriteRequests"),
			},
			writeCount:   1,
			wantErr:      true,
			wantResponse: 0,
		}, {
			name:         "attempt to overwrite the same name",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
				io.EOF,
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
			},
			writeCount:   2,
			wantErr:      true,
			wantResponse: 1,
		}, {
			name:         "overwrite with the same name + AllowOverwrite",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
				io.EOF,
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
				io.EOF,
			},
			writeCount:     2,
			allowOverwrite: true,
			wantResponse:   2,
		}, {
			name:         "two WriteRequests - 1st is empty",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  false,
					Data:         nil,
				},
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
				io.EOF,
			},
			writeCount:        1,
			wantResponse:      1,
			allowEmptyCommits: true,
		}, {
			name:         "two WriteRequests - 2nd is empty",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  false,
					Data:         []byte(testData),
				},
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  int64(len(testData)),
					FinishWrite:  true,
					Data:         nil,
				},
				io.EOF,
			},
			writeCount:   1,
			wantResponse: 1,
		}, {
			name:         "two WriteRequests - all empty",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  false,
					Data:         nil,
				},
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         nil,
				},
			},
			writeCount:        1,
			wantErr:           true,
			wantResponse:      1,
			allowEmptyCommits: true,
		}, {
			name:         "two WriteRequests - varying offset",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  100,
					FinishWrite:  false,
					Data:         []byte(testData),
				},
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  100 + int64(len(testData)),
					FinishWrite:  true,
					Data:         []byte(testData),
				},
				io.EOF,
			},
			writeCount:   1,
			wantResponse: 1,
		}, {
			name:         "two WriteRequests - disjoint offset",
			writeHandler: &TestWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  100,
					FinishWrite:  false,
					Data:         []byte(testData),
				},
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  200,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
			},
			writeCount:   1,
			wantErr:      true,
			wantResponse: 0,
		}, {
			name:         "fails with UngettableWriteHandler",
			writeHandler: &UngettableWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
			},
			writeCount: 1,
			wantErr:    true,
		}, {
			name:         "fails with UnwritableWriteHandler",
			writeHandler: &UnwritableWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
			},
			writeCount: 1,
			wantErr:    true,
		}, {
			name:         "fails with UnclosableWriteHandler",
			writeHandler: &UnclosableWriteHandler{},
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
			},
			writeCount:   1,
			wantErr:      true,
			wantResponse: 1,
		}, {
			name:         "fails with nil WriteHandler",
			writeHandler: nil,
			input: []interface{}{
				&pb.WriteRequest{
					ResourceName: testName,
					WriteOffset:  0,
					FinishWrite:  true,
					Data:         []byte(testData),
				},
			},
			writeCount: 1,
			wantErr:    true,
		},
	}

	ctx := context.Background()
	for _, tc := range testCases {
		readHandler := &TestReadHandler{}
		if tc.writeHandler != nil {
			readHandler = nil
		}
		setupServer(readHandler, tc.writeHandler)
		server.AllowOverwrite = tc.allowOverwrite
		var requestCount, responseCount int
		var err error

		for i := 0; i < tc.writeCount; i++ {
			err = server.rpc.Write(&fakeWriteServerImpl{
				ctx: ctx,
				receiver: func() (*pb.WriteRequest, error) {
					if requestCount >= len(tc.input) {
						t.Fatalf("%s: got %d call(s) to Recv, want %d from len(input)", tc.name, requestCount+1, len(tc.input))
					}
					v := tc.input[requestCount]
					requestCount++
					request, ok := v.(*pb.WriteRequest)
					if ok {
						return request, nil
					}
					err, ok := v.(error)
					if !ok {
						t.Fatalf("%s: unknown input: %v", tc.name, v)
					}
					return nil, err
				},
				sender: func(response *pb.WriteResponse) error {
					if !tc.allowEmptyCommits && response.CommittedSize == 0 {
						t.Fatalf("%s: invalid response: WriteResponse %v", tc.name, response)
					}
					responseCount++
					return nil
				},
			})
			gotErr := (err != nil)
			if i+1 < tc.writeCount {
				if gotErr {
					t.Errorf("%s: Write got err=%v, wantErr=%t, but on Write[%d/%d]. Error should not happen until last call to Write.", tc.name, err, tc.wantErr, i+1, tc.writeCount)
					break // The t.Errorf conditions below may erroneously fire, pay them no mind.
				}
			} else if gotErr != tc.wantErr {
				t.Errorf("%s: Write got err=%v, wantErr=%t", tc.name, err, tc.wantErr)
				break // The t.Errorf conditions below may erroneously fire, pay them no mind.
			}
		}
		if requestCount != len(tc.input) {
			t.Errorf("%s: got %d call(s) to Recv, want %d", tc.name, requestCount, len(tc.input))
		}
		if responseCount != tc.wantResponse {
			t.Errorf("%s: got %d call(s) to SendProto, want %d", tc.name, responseCount, tc.wantResponse)
		}
	}
}

func TestServerWrite_SendAndCloseError(t *testing.T) {
	const (
		wantRequest  = 2
		wantResponse = 1
	)

	ctx := context.Background()
	setupServer(nil, &TestWriteHandler{})
	var requestCount, responseCount int

	err := server.rpc.Write(&fakeWriteServerImpl{
		ctx: ctx,
		receiver: func() (*pb.WriteRequest, error) {
			if requestCount >= wantRequest {
				t.Fatalf("got %d call(s) to Recv, want %d", requestCount+1, wantRequest)
			}
			requestCount++
			return &pb.WriteRequest{
				ResourceName: testName,
				WriteOffset:  0,
				FinishWrite:  true,
				Data:         []byte(testData),
			}, nil
		},
		sender: func(response *pb.WriteResponse) error {
			responseCount++
			return errors.New("TestServerWrite SendProto error")
		},
	})
	if err == nil {
		t.Errorf("Write should have failed, but succeeded")
	}
	if requestCount != wantRequest {
		t.Errorf("got %d call(s) to Recv, want %d", requestCount, wantRequest)
	}
	if responseCount != wantResponse {
		t.Errorf("got %d call(s) to SendProto, want %d", responseCount, wantResponse)
	}
}

func TestQueryWriteStatus(t *testing.T) {
	testCases := []struct {
		name         string
		existingName string
		requestName  string
		wantErr      bool
	}{
		{
			name:         "existing name should work",
			existingName: testName,
			requestName:  testName,
		}, {
			name:         "missing name should break",
			existingName: testName,
			requestName:  "invalidName",
			wantErr:      true,
		},
	}

	ctx := context.Background()
	for _, tc := range testCases {
		setupServer(nil, &TestWriteHandler{})
		server.status[tc.existingName] = &pb.QueryWriteStatusResponse{}

		_, err := server.rpc.QueryWriteStatus(ctx, &pb.QueryWriteStatusRequest{
			ResourceName: tc.requestName,
		})

		if gotErr := (err != nil); gotErr != tc.wantErr {
			t.Errorf("%s: QueryWriteStatus(%q) got err=%v, wantErr=%t", tc.name, tc.requestName, err, tc.wantErr)
		}
	}
}

func TestServerRead(t *testing.T) {
	testCases := []struct {
		name         string
		readHandler  ReadHandler
		input        *pb.ReadRequest
		readCount    int
		wantErr      bool
		wantResponse []string
	}{
		{
			name:        "empty resource name",
			readHandler: &TestReadHandler{},
			input: &pb.ReadRequest{
				ReadLimit: 1,
			},
			readCount:    1,
			wantErr:      true,
			wantResponse: []string{},
		}, {
			name:        "test ReadLimit=-1",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    -1,
			},
			readCount: 1,
			wantErr:   true,
		}, {
			name:        "test ReadLimit=1",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    1,
			},
			readCount:    1,
			wantResponse: []string{"0"},
		}, {
			name:        "test ReadLimit=2",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    2,
			},
			readCount:    1,
			wantResponse: []string{"01"},
		}, {
			name:        "test ReadOffset=1 ReadLimit=2",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   1,
				ReadLimit:    2,
			},
			readCount:    1,
			wantResponse: []string{"12"},
		}, {
			name:        "test ReadOffset=2 ReadLimit=2",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   2,
				ReadLimit:    2,
			},
			readCount:    1,
			wantResponse: []string{"23"},
		}, {
			name:        "read all testData at exactly the limit",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    int64(len(testData)),
			},
			readCount:    1,
			wantResponse: []string{"0123456789"},
		}, {
			name:        "read all testData",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    int64(len(testData)) * 2,
			},
			readCount:    1,
			wantResponse: []string{"0123456789"},
		}, {
			name:        "read all testData 2 times",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    int64(len(testData)) * 2,
			},
			readCount:    2,
			wantResponse: []string{"0123456789", "0123456789"},
		}, {
			name:        "test ReadLimit=0",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    0,
			},
			readCount:    1,
			wantResponse: []string{"0123456789"},
		}, {
			name:        "test ReadLimit=1000",
			readHandler: &TestReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    1000,
			},
			readCount:    1,
			wantResponse: []string{"0123456789"},
		}, {
			name:        "fails with UngettableReadHandler",
			readHandler: &UngettableReadHandler{},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    int64(len(testData)),
			},
			readCount: 1,
			wantErr:   true,
		}, {
			name:        "fails with UnreadableReadHandler",
			readHandler: &UnreadableReadHandler{},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    int64(len(testData)),
			},
			readCount: 1,
			wantErr:   true,
		}, {
			name:        "fails with UnclosableReadHandler",
			readHandler: &UnclosableReadHandler{buf: testData},
			input: &pb.ReadRequest{
				ResourceName: testName,
				ReadOffset:   0,
				ReadLimit:    int64(len(testData)) * 2,
			},
			readCount:    1,
			wantErr:      true,
			wantResponse: []string{"0123456789"},
		}, {
			name:        "fails with nil ReadRequest",
			readHandler: &TestReadHandler{buf: testData},
			readCount:   1,
			wantErr:     true,
		}, {
			name:        "fails with nil ReadHandler",
			readHandler: nil,
			readCount:   1,
			wantErr:     true,
		},
	}

	ctx := context.Background()
	for _, tc := range testCases {
		var writeHandler WriteHandler
		if tc.readHandler == nil {
			writeHandler = &TestWriteHandler{}
		}
		setupServer(tc.readHandler, writeHandler)
		var responseCount int
		var err error

		for i := 0; i < tc.readCount; i++ {
			err = server.rpc.Read(tc.input, &fakeReadServerImpl{
				ctx: ctx,
				sender: func(response *pb.ReadResponse) error {
					if responseCount >= len(tc.wantResponse) {
						t.Fatalf("%s: got %d call(s) to Send(), want %d", tc.name, responseCount+1, len(tc.wantResponse))
					}
					if got, want := string(response.Data), tc.wantResponse[responseCount]; got != want {
						t.Fatalf("%s: response[%d] got %q, want %q", tc.name, responseCount, got, want)
					}
					responseCount++
					return nil
				},
			})
			gotErr := (err != nil)
			if i+1 < tc.readCount {
				if gotErr {
					t.Errorf("%s: Read got err=%v, wantErr=%t, but on Read[%d/%d]. Error should not happen until last call to Read", tc.name, err, tc.wantErr, i+1, tc.readCount)
					break
				}
			} else if gotErr != tc.wantErr {
				t.Errorf("%s: Read got err=%v, wantErr=%t", tc.name, err, tc.wantErr)
				break
			}
		}
		if responseCount != len(tc.wantResponse) {
			t.Errorf("%s: got %d call(s) to Send, want %d", tc.name, responseCount, len(tc.wantResponse))
		}
	}
}

func TestServerRead_SendError(t *testing.T) {
	setupServer(&TestReadHandler{buf: testData}, nil)

	err := server.rpc.Read(&pb.ReadRequest{
		ResourceName: testName,
		ReadOffset:   0,
		ReadLimit:    int64(len(testData)) * 2,
	}, &fakeReadServerImpl{
		ctx: context.Background(),
		sender: func(response *pb.ReadResponse) error {
			if string(response.Data) != testData {
				t.Fatalf("Send: got %v, want %q", response, testData)
			}
			return errors.New("TestServerRead Send() error")
		},
	})

	if err == nil {
		t.Fatal("Read() should have failed, but succeeded")
	}
}

type fakeWriteServerImpl struct {
	pb.ByteStream_WriteServer
	ctx      context.Context
	receiver func() (*pb.WriteRequest, error)
	sender   func(*pb.WriteResponse) error
}

func (fake *fakeWriteServerImpl) Context() context.Context {
	return fake.ctx
}

func (fake *fakeWriteServerImpl) Recv() (*pb.WriteRequest, error) {
	return fake.receiver()
}

func (fake *fakeWriteServerImpl) SendMsg(m interface{}) error {
	return fake.sender(m.(*pb.WriteResponse))
}

func (fake *fakeWriteServerImpl) SendAndClose(m *pb.WriteResponse) error {
	fake.sender(m)
	return nil
}

type fakeReadServerImpl struct {
	pb.ByteStream_ReadServer
	ctx    context.Context
	sender func(*pb.ReadResponse) error
}

func (fake *fakeReadServerImpl) Context() context.Context {
	return fake.ctx
}

func (fake *fakeReadServerImpl) Send(response *pb.ReadResponse) error {
	return fake.sender(response)
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

type UngettableWriteHandler struct{}

func (w *UngettableWriteHandler) GetWriter(ctx context.Context, name string, initOffset int64) (io.Writer, error) {
	return nil, errors.New("UngettableWriteHandler.GetWriter() always fails")
}

func (w *UngettableWriteHandler) Close(ctx context.Context, name string) error {
	return nil
}

type UnwritableWriter struct{}

func (w *UnwritableWriter) Write(p []byte) (int, error) {
	return 0, errors.New("UnwritableWriter.Write() always fails")
}

type UnwritableWriteHandler struct{}

func (w *UnwritableWriteHandler) GetWriter(ctx context.Context, name string, initOffset int64) (io.Writer, error) {
	return &UnwritableWriter{}, nil
}

func (w *UnwritableWriteHandler) Close(ctx context.Context, name string) error {
	return nil
}

type UnclosableWriter struct{}

func (w *UnclosableWriter) Write(p []byte) (int, error) {
	return len(p), nil
}

type UnclosableWriteHandler struct{}

func (w *UnclosableWriteHandler) GetWriter(ctx context.Context, name string, initOffset int64) (io.Writer, error) {
	return &UnclosableWriter{}, nil
}

func (w *UnclosableWriteHandler) Close(ctx context.Context, name string) error {
	return errors.New("UnclosableWriteHandler.Close() always fails")
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

type UngettableReadHandler struct{}

func (r *UngettableReadHandler) GetReader(ctx context.Context, name string) (io.ReaderAt, error) {
	return nil, errors.New("UngettableReadHandler.GetReader() always fails")
}

func (r *UngettableReadHandler) Close(ctx context.Context, name string) error {
	return nil
}

type UnreadableReader struct{}

func (r *UnreadableReader) ReadAt(p []byte, offset int64) (int, error) {
	return 0, errors.New("UnreadableReader.ReadAt() always fails")
}

type UnreadableReadHandler struct{}

func (r *UnreadableReadHandler) GetReader(ctx context.Context, name string) (io.ReaderAt, error) {
	return &UnreadableReader{}, nil
}

func (r *UnreadableReadHandler) Close(ctx context.Context, name string) error {
	return nil
}

type UnclosableReadHandler struct {
	buf string
}

func (r *UnclosableReadHandler) GetReader(ctx context.Context, name string) (io.ReaderAt, error) {
	return bytes.NewReader([]byte(r.buf)), nil
}

func (r *UnclosableReadHandler) Close(ctx context.Context, name string) error {
	return fmt.Errorf("UnclosableReader.Close(%s) always fails", name)
}

func registerServer() {
	gsrv := grpc.NewServer()
	var err error
	server, err = NewServer(gsrv, &TestReadHandler{}, &TestWriteHandler{})
	if err != nil {
		log.Fatalf("NewServer() failed: %v", err)
	}
}

func setupServer(readHandler ReadHandler, writeHandler WriteHandler) {
	setupServerOnce.Do(registerServer)
	server.status = make(map[string]*pb.QueryWriteStatusResponse)
	server.readHandler = readHandler
	server.writeHandler = writeHandler
}
