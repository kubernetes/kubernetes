// Copyright 2017 Google LLC
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

package internal

import (
	"errors"
	"testing"

	"google.golang.org/api/googleapi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const wantMessage = "prefix: msg"

func TestAnnotateGRPC(t *testing.T) {
	// grpc Status error
	err := status.Error(codes.NotFound, "msg")
	err = Annotate(err, "prefix")
	got, ok := status.FromError(err)
	if !ok {
		t.Fatalf("got %T, wanted a status", got)
	}
	if g, w := got.Code(), codes.NotFound; g != w {
		t.Errorf("got code %v, want %v", g, w)
	}
	if g, w := got.Message(), wantMessage; g != w {
		t.Errorf("got message %q, want %q", g, w)
	}
}

func TestAnnotateGoogleapi(t *testing.T) {
	// googleapi error
	var err error = &googleapi.Error{Code: 403, Message: "msg"}
	err = Annotate(err, "prefix")
	got2, ok := err.(*googleapi.Error)
	if !ok {
		t.Fatalf("got %T, wanted a googleapi.Error", got2)
	}
	if g, w := got2.Code, 403; g != w {
		t.Errorf("got code %d, want %d", g, w)
	}
	if g, w := got2.Message, wantMessage; g != w {
		t.Errorf("got message %q, want %q", g, w)
	}
}

func TestAnnotateUnknownError(t *testing.T) {
	err := Annotate(errors.New("msg"), "prefix")
	if g, w := err.Error(), wantMessage; g != w {
		t.Errorf("got message %q, want %q", g, w)
	}
}
