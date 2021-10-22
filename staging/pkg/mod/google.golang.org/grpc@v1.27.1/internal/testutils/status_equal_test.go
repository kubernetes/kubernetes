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

package testutils

import (
	"testing"

	anypb "github.com/golang/protobuf/ptypes/any"
	spb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var statusErr = status.ErrorProto(&spb.Status{
	Code:    int32(codes.DataLoss),
	Message: "error for testing",
	Details: []*anypb.Any{{
		TypeUrl: "url",
		Value:   []byte{6, 0, 0, 6, 1, 3},
	}},
})

func TestStatusErrEqual(t *testing.T) {
	tests := []struct {
		name      string
		err1      error
		err2      error
		wantEqual bool
	}{
		{"nil errors", nil, nil, true},
		{"equal OK status", status.New(codes.OK, "").Err(), status.New(codes.OK, "").Err(), true},
		{"equal status errors", statusErr, statusErr, true},
		{"different status errors", statusErr, status.New(codes.OK, "").Err(), false},
	}

	for _, test := range tests {
		if gotEqual := StatusErrEqual(test.err1, test.err2); gotEqual != test.wantEqual {
			t.Errorf("%v: StatusErrEqual(%v, %v) = %v, want %v", test.name, test.err1, test.err2, gotEqual, test.wantEqual)
		}
	}
}
