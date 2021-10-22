/*
 *
 * Copyright 2019 gRPC authors.
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

package status_test

import (
	"errors"
	"testing"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/grpc_testing"
)

func errWithDetails(t *testing.T, s *status.Status, details ...proto.Message) error {
	t.Helper()
	res, err := s.WithDetails(details...)
	if err != nil {
		t.Fatalf("(%v).WithDetails(%v) = %v, %v; want _, <nil>", s, details, res, err)
	}
	return res.Err()
}

func TestErrorIs(t *testing.T) {
	// Test errors.
	testErr := status.Error(codes.Internal, "internal server error")
	testErrWithDetails := errWithDetails(t, status.New(codes.Internal, "internal server error"), &grpc_testing.Empty{})

	// Test cases.
	testCases := []struct {
		err1, err2 error
		want       bool
	}{
		{err1: testErr, err2: nil, want: false},
		{err1: testErr, err2: status.Error(codes.Internal, "internal server error"), want: true},
		{err1: testErr, err2: status.Error(codes.Internal, "internal error"), want: false},
		{err1: testErr, err2: status.Error(codes.Unknown, "internal server error"), want: false},
		{err1: testErr, err2: errors.New("non-grpc error"), want: false},
		{err1: testErrWithDetails, err2: status.Error(codes.Internal, "internal server error"), want: false},
		{err1: testErrWithDetails, err2: errWithDetails(t, status.New(codes.Internal, "internal server error"), &grpc_testing.Empty{}), want: true},
		{err1: testErrWithDetails, err2: errWithDetails(t, status.New(codes.Internal, "internal server error"), &grpc_testing.Empty{}, &grpc_testing.Empty{}), want: false},
	}

	for _, tc := range testCases {
		isError, ok := tc.err1.(interface{ Is(target error) bool })
		if !ok {
			t.Errorf("(%v) does not implement is", tc.err1)
			continue
		}

		is := isError.Is(tc.err2)
		if is != tc.want {
			t.Errorf("(%v).Is(%v) = %t; want %t", tc.err1, tc.err2, is, tc.want)
		}
	}
}
