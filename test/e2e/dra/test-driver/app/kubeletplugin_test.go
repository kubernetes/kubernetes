/*
Copyright The Kubernetes Authors.

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

package app

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"google.golang.org/grpc"
)

// TestRecordGRPCStreamError checks that the error of a stream call stays visible
// when the call slice is reallocated while the stream is open.
func TestRecordGRPCStreamError(t *testing.T) {
	// Room for one record: the call recorded below has to reallocate the slice.
	ex := &ExamplePlugin{gRPCCalls: make([]*GRPCCall, 0, 1)}
	streamErr := errors.New("stream failed")
	unary := func(ctx context.Context, req interface{}) (interface{}, error) { return nil, nil }
	err := ex.recordGRPCStream(nil, nil, &grpc.StreamServerInfo{FullMethod: "/test/Stream"}, func(srv interface{}, stream grpc.ServerStream) error {
		_, _ = ex.recordGRPCCall(context.Background(), nil, &grpc.UnaryServerInfo{FullMethod: "/test/Unary"}, unary)
		return streamErr
	})
	if !errors.Is(err, streamErr) {
		t.Fatalf("recordGRPCStream returned %v, want %v", err, streamErr)
	}
	calls := ex.GetGRPCCalls()
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2: %+v", len(calls), calls)
	}
	if calls[0].FullMethod != "/test/Stream" || !errors.Is(calls[0].Err, streamErr) {
		t.Errorf("stream call recorded as %+v, want FullMethod /test/Stream and Err %v", calls[0], streamErr)
	}
}

// TestRecordGRPCStreamReset checks that a reset while a stream is open neither
// panics nor resurrects the call.
func TestRecordGRPCStreamReset(t *testing.T) {
	ex := &ExamplePlugin{}
	err := ex.recordGRPCStream(nil, nil, &grpc.StreamServerInfo{FullMethod: "/test/Stream"}, func(srv interface{}, stream grpc.ServerStream) error {
		ex.ResetGRPCCalls()
		return errors.New("stream failed")
	})
	if err == nil {
		t.Fatal("recordGRPCStream returned nil, want the handler's error")
	}
	if calls := ex.GetGRPCCalls(); len(calls) != 0 {
		t.Errorf("got %d calls after reset, want 0: %+v", len(calls), calls)
	}
}

// TestRecordGRPCCallReset checks that a reset during a unary call neither panics
// nor lets that call overwrite a record made after the reset.
func TestRecordGRPCCallReset(t *testing.T) {
	firstErr := errors.New("first failed")
	second := GRPCCall{FullMethod: "/test/Second", Request: "second request", Response: "second response"}
	for _, tc := range []struct {
		name          string
		recordAnother bool
		want          []GRPCCall
	}{
		{name: "empty history", want: []GRPCCall{}},
		{name: "new call after the reset", recordAnother: true, want: []GRPCCall{second}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ex := &ExamplePlugin{}
			resp, err := ex.recordGRPCCall(context.Background(), "first request", &grpc.UnaryServerInfo{FullMethod: "/test/First"}, func(ctx context.Context, req interface{}) (interface{}, error) {
				ex.ResetGRPCCalls()
				if tc.recordAnother {
					_, _ = ex.recordGRPCCall(ctx, second.Request, &grpc.UnaryServerInfo{FullMethod: second.FullMethod}, func(ctx context.Context, req interface{}) (interface{}, error) {
						return second.Response, nil
					})
				}
				return "first response", firstErr
			})
			if resp != "first response" || !errors.Is(err, firstErr) {
				t.Fatalf("recordGRPCCall returned (%v, %v), want (first response, %v)", resp, err, firstErr)
			}
			if calls := ex.GetGRPCCalls(); !reflect.DeepEqual(calls, tc.want) {
				t.Errorf("got %+v after reset, want %+v", calls, tc.want)
			}
		})
	}
}

// TestGetGRPCCallsSnapshot checks that a slice handed out by GetGRPCCalls does
// not change when the call completes, the caller edits it, or the history is reset.
func TestGetGRPCCallsSnapshot(t *testing.T) {
	ex := &ExamplePlugin{}
	var during []GRPCCall
	_, _ = ex.recordGRPCCall(context.Background(), nil, &grpc.UnaryServerInfo{FullMethod: "/test/Unary"}, func(ctx context.Context, req interface{}) (interface{}, error) {
		during = ex.GetGRPCCalls()
		return "done", errors.New("unary failed")
	})
	want := []GRPCCall{{FullMethod: "/test/Unary"}}
	if !reflect.DeepEqual(during, want) {
		t.Fatalf("snapshot taken during the call is %+v, want %+v", during, want)
	}
	after := ex.GetGRPCCalls()
	if len(after) != 1 || after[0].Response != "done" || after[0].Err == nil {
		t.Fatalf("history after the call is %+v, want the response and the error filled in", after)
	}
	if !reflect.DeepEqual(during, want) {
		t.Errorf("snapshot changed when the call completed: %+v", during)
	}
	after[0].FullMethod = "/test/Edited"
	if got := ex.GetGRPCCalls(); got[0].FullMethod != "/test/Unary" {
		t.Errorf("editing a snapshot changed the history: %+v", got)
	}
	ex.ResetGRPCCalls()
	if got := ex.GetGRPCCalls(); len(got) != 0 {
		t.Errorf("history after the reset is %+v, want empty", got)
	}
	if during[0].FullMethod != "/test/Unary" || after[0].Response != "done" {
		t.Errorf("snapshots changed by the reset: during=%+v after=%+v", during, after)
	}
}
