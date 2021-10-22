// Copyright 2018 Google LLC
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

package trace

import (
	"errors"
	"net/http"
	"testing"

	"cloud.google.com/go/internal/testutil"
	octrace "go.opencensus.io/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/genproto/googleapis/rpc/code"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestToStatus(t *testing.T) {
	for _, testcase := range []struct {
		input error
		want  octrace.Status
	}{
		{
			errors.New("some random error"),
			octrace.Status{Code: int32(code.Code_UNKNOWN), Message: "some random error"},
		},
		{
			&googleapi.Error{Code: http.StatusConflict, Message: "some specific googleapi http error"},
			octrace.Status{Code: int32(code.Code_ALREADY_EXISTS), Message: "some specific googleapi http error"},
		},
		{
			status.Error(codes.DataLoss, "some specific grpc error"),
			octrace.Status{Code: int32(code.Code_DATA_LOSS), Message: "some specific grpc error"},
		},
	} {
		got := toStatus(testcase.input)
		if r := testutil.Diff(got, testcase.want); r != "" {
			t.Errorf("got -, want +:\n%s", r)
		}
	}
}
