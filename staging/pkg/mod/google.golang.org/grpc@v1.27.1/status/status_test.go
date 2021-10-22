/*
 *
 * Copyright 2017 gRPC authors.
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

package status

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	apb "github.com/golang/protobuf/ptypes/any"
	dpb "github.com/golang/protobuf/ptypes/duration"
	"github.com/google/go-cmp/cmp"
	cpb "google.golang.org/genproto/googleapis/rpc/code"
	epb "google.golang.org/genproto/googleapis/rpc/errdetails"
	spb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc/codes"
)

// errEqual is essentially a copy of testutils.StatusErrEqual(), to avoid a
// cyclic dependency.
func errEqual(err1, err2 error) bool {
	status1, ok := FromError(err1)
	if !ok {
		return false
	}
	status2, ok := FromError(err2)
	if !ok {
		return false
	}
	return proto.Equal(status1.Proto(), status2.Proto())
}

func TestErrorsWithSameParameters(t *testing.T) {
	const description = "some description"
	e1 := Errorf(codes.AlreadyExists, description)
	e2 := Errorf(codes.AlreadyExists, description)
	if e1 == e2 || !errEqual(e1, e2) {
		t.Fatalf("Errors should be equivalent but unique - e1: %v, %v  e2: %p, %v", e1.(*statusError), e1, e2.(*statusError), e2)
	}
}

func TestFromToProto(t *testing.T) {
	s := &spb.Status{
		Code:    int32(codes.Internal),
		Message: "test test test",
		Details: []*apb.Any{{TypeUrl: "foo", Value: []byte{3, 2, 1}}},
	}

	err := FromProto(s)
	if got := err.Proto(); !proto.Equal(s, got) {
		t.Fatalf("Expected errors to be identical - s: %v  got: %v", s, got)
	}
}

func TestFromNilProto(t *testing.T) {
	tests := []*Status{nil, FromProto(nil)}
	for _, s := range tests {
		if c := s.Code(); c != codes.OK {
			t.Errorf("s: %v - Expected s.Code() = OK; got %v", s, c)
		}
		if m := s.Message(); m != "" {
			t.Errorf("s: %v - Expected s.Message() = \"\"; got %q", s, m)
		}
		if p := s.Proto(); p != nil {
			t.Errorf("s: %v - Expected s.Proto() = nil; got %q", s, p)
		}
		if e := s.Err(); e != nil {
			t.Errorf("s: %v - Expected s.Err() = nil; got %v", s, e)
		}
	}
}

func TestError(t *testing.T) {
	err := Error(codes.Internal, "test description")
	if got, want := err.Error(), "rpc error: code = Internal desc = test description"; got != want {
		t.Fatalf("err.Error() = %q; want %q", got, want)
	}
	s, _ := FromError(err)
	if got, want := s.Code(), codes.Internal; got != want {
		t.Fatalf("err.Code() = %s; want %s", got, want)
	}
	if got, want := s.Message(), "test description"; got != want {
		t.Fatalf("err.Message() = %s; want %s", got, want)
	}
}

func TestErrorOK(t *testing.T) {
	err := Error(codes.OK, "foo")
	if err != nil {
		t.Fatalf("Error(codes.OK, _) = %p; want nil", err.(*statusError))
	}
}

func TestErrorProtoOK(t *testing.T) {
	s := &spb.Status{Code: int32(codes.OK)}
	if got := ErrorProto(s); got != nil {
		t.Fatalf("ErrorProto(%v) = %v; want nil", s, got)
	}
}

func TestFromError(t *testing.T) {
	code, message := codes.Internal, "test description"
	err := Error(code, message)
	s, ok := FromError(err)
	if !ok || s.Code() != code || s.Message() != message || s.Err() == nil {
		t.Fatalf("FromError(%v) = %v, %v; want <Code()=%s, Message()=%q, Err()!=nil>, true", err, s, ok, code, message)
	}
}

func TestFromErrorOK(t *testing.T) {
	code, message := codes.OK, ""
	s, ok := FromError(nil)
	if !ok || s.Code() != code || s.Message() != message || s.Err() != nil {
		t.Fatalf("FromError(nil) = %v, %v; want <Code()=%s, Message()=%q, Err=nil>, true", s, ok, code, message)
	}
}

type customError struct {
	Code    codes.Code
	Message string
	Details []*apb.Any
}

func (c customError) Error() string {
	return fmt.Sprintf("rpc error: code = %s desc = %s", c.Code, c.Message)
}

func (c customError) GRPCStatus() *Status {
	return &Status{
		s: &spb.Status{
			Code:    int32(c.Code),
			Message: c.Message,
			Details: c.Details,
		},
	}
}

func TestFromErrorImplementsInterface(t *testing.T) {
	code, message := codes.Internal, "test description"
	details := []*apb.Any{{
		TypeUrl: "testUrl",
		Value:   []byte("testValue"),
	}}
	err := customError{
		Code:    code,
		Message: message,
		Details: details,
	}
	s, ok := FromError(err)
	if !ok || s.Code() != code || s.Message() != message || s.Err() == nil {
		t.Fatalf("FromError(%v) = %v, %v; want <Code()=%s, Message()=%q, Err()!=nil>, true", err, s, ok, code, message)
	}
	pd := s.Proto().GetDetails()
	if len(pd) != 1 || !proto.Equal(pd[0], details[0]) {
		t.Fatalf("s.Proto.GetDetails() = %v; want <Details()=%s>", pd, details)
	}
}

func TestFromErrorUnknownError(t *testing.T) {
	code, message := codes.Unknown, "unknown error"
	err := errors.New("unknown error")
	s, ok := FromError(err)
	if ok || s.Code() != code || s.Message() != message {
		t.Fatalf("FromError(%v) = %v, %v; want <Code()=%s, Message()=%q>, false", err, s, ok, code, message)
	}
}

func TestConvertKnownError(t *testing.T) {
	code, message := codes.Internal, "test description"
	err := Error(code, message)
	s := Convert(err)
	if s.Code() != code || s.Message() != message {
		t.Fatalf("Convert(%v) = %v; want <Code()=%s, Message()=%q>", err, s, code, message)
	}
}

func TestConvertUnknownError(t *testing.T) {
	code, message := codes.Unknown, "unknown error"
	err := errors.New("unknown error")
	s := Convert(err)
	if s.Code() != code || s.Message() != message {
		t.Fatalf("Convert(%v) = %v; want <Code()=%s, Message()=%q>", err, s, code, message)
	}
}

func TestStatus_ErrorDetails(t *testing.T) {
	tests := []struct {
		code    codes.Code
		details []proto.Message
	}{
		{
			code:    codes.NotFound,
			details: nil,
		},
		{
			code: codes.NotFound,
			details: []proto.Message{
				&epb.ResourceInfo{
					ResourceType: "book",
					ResourceName: "projects/1234/books/5678",
					Owner:        "User",
				},
			},
		},
		{
			code: codes.Internal,
			details: []proto.Message{
				&epb.DebugInfo{
					StackEntries: []string{
						"first stack",
						"second stack",
					},
				},
			},
		},
		{
			code: codes.Unavailable,
			details: []proto.Message{
				&epb.RetryInfo{
					RetryDelay: &dpb.Duration{Seconds: 60},
				},
				&epb.ResourceInfo{
					ResourceType: "book",
					ResourceName: "projects/1234/books/5678",
					Owner:        "User",
				},
			},
		},
	}

	for _, tc := range tests {
		s, err := New(tc.code, "").WithDetails(tc.details...)
		if err != nil {
			t.Fatalf("(%v).WithDetails(%+v) failed: %v", str(s), tc.details, err)
		}
		details := s.Details()
		for i := range details {
			if !proto.Equal(details[i].(proto.Message), tc.details[i]) {
				t.Fatalf("(%v).Details()[%d] = %+v, want %+v", str(s), i, details[i], tc.details[i])
			}
		}
	}
}

func TestStatus_WithDetails_Fail(t *testing.T) {
	tests := []*Status{
		nil,
		FromProto(nil),
		New(codes.OK, ""),
	}
	for _, s := range tests {
		if s, err := s.WithDetails(); err == nil || s != nil {
			t.Fatalf("(%v).WithDetails(%+v) = %v, %v; want nil, non-nil", str(s), []proto.Message{}, s, err)
		}
	}
}

func TestStatus_ErrorDetails_Fail(t *testing.T) {
	tests := []struct {
		s *Status
		i []interface{}
	}{
		{
			nil,
			nil,
		},
		{
			FromProto(nil),
			nil,
		},
		{
			New(codes.OK, ""),
			[]interface{}{},
		},
		{
			FromProto(&spb.Status{
				Code: int32(cpb.Code_CANCELLED),
				Details: []*apb.Any{
					{
						TypeUrl: "",
						Value:   []byte{},
					},
					mustMarshalAny(&epb.ResourceInfo{
						ResourceType: "book",
						ResourceName: "projects/1234/books/5678",
						Owner:        "User",
					}),
				},
			}),
			[]interface{}{
				errors.New(`message type url "" is invalid`),
				&epb.ResourceInfo{
					ResourceType: "book",
					ResourceName: "projects/1234/books/5678",
					Owner:        "User",
				},
			},
		},
	}
	for _, tc := range tests {
		got := tc.s.Details()
		if !cmp.Equal(got, tc.i, cmp.Comparer(proto.Equal), cmp.Comparer(equalError)) {
			t.Errorf("(%v).Details() = %+v, want %+v", str(tc.s), got, tc.i)
		}
	}
}

func equalError(x, y error) bool {
	return x == y || (x != nil && y != nil && x.Error() == y.Error())
}

func str(s *Status) string {
	if s == nil {
		return "nil"
	}
	if s.s == nil {
		return "<Code=OK>"
	}
	return fmt.Sprintf("<Code=%v, Message=%q, Details=%+v>", codes.Code(s.s.GetCode()), s.s.GetMessage(), s.s.GetDetails())
}

// mustMarshalAny converts a protobuf message to an any.
func mustMarshalAny(msg proto.Message) *apb.Any {
	any, err := ptypes.MarshalAny(msg)
	if err != nil {
		panic(fmt.Sprintf("ptypes.MarshalAny(%+v) failed: %v", msg, err))
	}
	return any
}

func TestFromContextError(t *testing.T) {
	testCases := []struct {
		in   error
		want *Status
	}{
		{in: nil, want: New(codes.OK, "")},
		{in: context.DeadlineExceeded, want: New(codes.DeadlineExceeded, context.DeadlineExceeded.Error())},
		{in: context.Canceled, want: New(codes.Canceled, context.Canceled.Error())},
		{in: errors.New("other"), want: New(codes.Unknown, "other")},
	}
	for _, tc := range testCases {
		got := FromContextError(tc.in)
		if got.Code() != tc.want.Code() || got.Message() != tc.want.Message() {
			t.Errorf("FromContextError(%v) = %v; want %v", tc.in, got, tc.want)
		}
	}
}
