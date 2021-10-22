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
	"fmt"

	"google.golang.org/api/googleapi"
	"google.golang.org/grpc/status"
)

// Annotate prepends msg to the error message in err, attempting
// to preserve other information in err, like an error code.
//
// Annotate panics if err is nil.
//
// Annotate knows about these error types:
// - "google.golang.org/grpc/status".Status
// - "google.golang.org/api/googleapi".Error
// If the error is not one of these types, Annotate behaves
// like
//   fmt.Errorf("%s: %v", msg, err)
func Annotate(err error, msg string) error {
	if err == nil {
		panic("Annotate called with nil")
	}
	if s, ok := status.FromError(err); ok {
		p := s.Proto()
		p.Message = msg + ": " + p.Message
		return status.ErrorProto(p)
	}
	if g, ok := err.(*googleapi.Error); ok {
		g.Message = msg + ": " + g.Message
		return g
	}
	return fmt.Errorf("%s: %v", msg, err)
}

// Annotatef uses format and args to format a string, then calls Annotate.
func Annotatef(err error, format string, args ...interface{}) error {
	return Annotate(err, fmt.Sprintf(format, args...))
}
