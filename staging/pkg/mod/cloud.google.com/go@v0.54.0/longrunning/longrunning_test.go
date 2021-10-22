// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package lro supports Long Running Operations for the Google Cloud Libraries.
//
// This package is still experimental and subject to change.
package longrunning

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	"github.com/golang/protobuf/ptypes/duration"
	gax "github.com/googleapis/gax-go/v2"
	pb "google.golang.org/genproto/googleapis/longrunning"
	rpcstatus "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type getterService struct {
	operationsClient

	// clock represents the fake current time of the service.
	// It is the running sum of the of the duration we have slept.
	clock time.Duration

	// getTimes records the times at which GetOperation is called.
	getTimes []time.Duration

	// results are the fake results that GetOperation should return.
	results []*pb.Operation
}

func (s *getterService) GetOperation(context.Context, *pb.GetOperationRequest, ...gax.CallOption) (*pb.Operation, error) {
	i := len(s.getTimes)
	s.getTimes = append(s.getTimes, s.clock)
	if i >= len(s.results) {
		return nil, errors.New("unexpected call")
	}
	return s.results[i], nil
}

func (s *getterService) sleeper() sleeper {
	return func(_ context.Context, d time.Duration) error {
		s.clock += d
		return nil
	}
}

func TestWait(t *testing.T) {
	responseDur := ptypes.DurationProto(42 * time.Second)
	responseAny, err := ptypes.MarshalAny(responseDur)
	if err != nil {
		t.Fatal(err)
	}

	s := &getterService{
		results: []*pb.Operation{
			{Name: "foo"},
			{Name: "foo"},
			{Name: "foo"},
			{Name: "foo"},
			{Name: "foo"},
			{
				Name: "foo",
				Done: true,
				Result: &pb.Operation_Response{
					Response: responseAny,
				},
			},
		},
	}
	op := &Operation{
		c:     s,
		proto: &pb.Operation{Name: "foo"},
	}
	if op.Done() {
		t.Fatal("operation should not have completed yet")
	}

	var resp duration.Duration
	bo := gax.Backoff{
		Initial: 1 * time.Second,
		Max:     3 * time.Second,
	}
	if err := op.wait(context.Background(), &resp, &bo, s.sleeper()); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(&resp, responseDur) {
		t.Errorf("response, got %v, want %v", resp, responseDur)
	}
	if !op.Done() {
		t.Errorf("operation should have completed")
	}

	maxWait := []time.Duration{
		1 * time.Second,
		2 * time.Second,
		3 * time.Second,
		3 * time.Second,
		3 * time.Second,
	}
	for i := 0; i < len(s.getTimes)-1; i++ {
		w := s.getTimes[i+1] - s.getTimes[i]
		if mw := maxWait[i]; w > mw {
			t.Errorf("backoff, waited %s, max %s", w, mw)
		}
	}
}

func TestPollRequestError(t *testing.T) {
	const opName = "foo"

	// All calls error.
	s := &getterService{}
	op := &Operation{
		c:     s,
		proto: &pb.Operation{Name: opName},
	}
	if err := op.Poll(context.Background(), nil); err == nil {
		t.Fatalf("Poll should error")
	}
	if n := op.Name(); n != opName {
		t.Errorf("operation name, got %q, want %q", n, opName)
	}
	if op.Done() {
		t.Errorf("operation should not have completed; we failed to fetch state")
	}
}

func TestPollErrorResult(t *testing.T) {
	const (
		errCode = codes.NotFound
		errMsg  = "my error"
	)
	op := &Operation{
		proto: &pb.Operation{
			Name: "foo",
			Done: true,
			Result: &pb.Operation_Error{
				Error: &rpcstatus.Status{
					Code:    int32(errCode),
					Message: errMsg,
				},
			},
		},
	}
	err := op.Poll(context.Background(), nil)
	if got := status.Code(err); got != errCode {
		t.Errorf("error code, want %s, got %s", errCode, got)
	}
	if got := grpc.ErrorDesc(err); got != errMsg {
		t.Errorf("error code, want %s, got %s", errMsg, got)
	}
	if !op.Done() {
		t.Errorf("operation should have completed")
	}
}

type errService struct {
	operationsClient
	errCancel, errDelete error
}

func (s *errService) CancelOperation(context.Context, *pb.CancelOperationRequest, ...gax.CallOption) error {
	return s.errCancel
}

func (s *errService) DeleteOperation(context.Context, *pb.DeleteOperationRequest, ...gax.CallOption) error {
	return s.errDelete
}

func TestCancelReturnsError(t *testing.T) {
	s := &errService{
		errCancel: errors.New("cancel error"),
	}
	op := &Operation{
		c:     s,
		proto: &pb.Operation{Name: "foo"},
	}
	if got, want := op.Cancel(context.Background()), s.errCancel; got != want {
		t.Errorf("cancel, got error %s, want %s", got, want)
	}
}

func TestDeleteReturnsError(t *testing.T) {
	s := &errService{
		errDelete: errors.New("delete error"),
	}
	op := &Operation{
		c:     s,
		proto: &pb.Operation{Name: "foo"},
	}
	if got, want := op.Delete(context.Background()), s.errDelete; got != want {
		t.Errorf("cancel, got error %s, want %s", got, want)
	}
}
