// Copyright 2015 Google Inc. All Rights Reserved.
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

package bigquery

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	bq "google.golang.org/api/bigquery/v2"
)

func rowInsertionError(msg string) RowInsertionError {
	return RowInsertionError{Errors: []error{errors.New(msg)}}
}

func TestPutMultiErrorString(t *testing.T) {
	testCases := []struct {
		errs PutMultiError
		want string
	}{
		{
			errs: PutMultiError{},
			want: "0 row insertions failed",
		},
		{
			errs: PutMultiError{rowInsertionError("a")},
			want: "1 row insertion failed",
		},
		{
			errs: PutMultiError{rowInsertionError("a"), rowInsertionError("b")},
			want: "2 row insertions failed",
		},
	}

	for _, tc := range testCases {
		if tc.errs.Error() != tc.want {
			t.Errorf("PutMultiError string: got:\n%v\nwant:\n%v", tc.errs.Error(), tc.want)
		}
	}
}

func TestMultiErrorString(t *testing.T) {
	testCases := []struct {
		errs MultiError
		want string
	}{
		{
			errs: MultiError{},
			want: "(0 errors)",
		},
		{
			errs: MultiError{errors.New("a")},
			want: "a",
		},
		{
			errs: MultiError{errors.New("a"), errors.New("b")},
			want: "a (and 1 other error)",
		},
		{
			errs: MultiError{errors.New("a"), errors.New("b"), errors.New("c")},
			want: "a (and 2 other errors)",
		},
	}

	for _, tc := range testCases {
		if tc.errs.Error() != tc.want {
			t.Errorf("PutMultiError string: got:\n%v\nwant:\n%v", tc.errs.Error(), tc.want)
		}
	}
}

func TestErrorFromErrorProto(t *testing.T) {
	for _, test := range []struct {
		in   *bq.ErrorProto
		want *Error
	}{
		{nil, nil},
		{
			in:   &bq.ErrorProto{Location: "L", Message: "M", Reason: "R"},
			want: &Error{Location: "L", Message: "M", Reason: "R"},
		},
	} {
		if got := errorFromErrorProto(test.in); !reflect.DeepEqual(got, test.want) {
			t.Errorf("%v: got %v, want %v", test.in, got, test.want)
		}
	}
}

func TestErrorString(t *testing.T) {
	e := &Error{Location: "<L>", Message: "<M>", Reason: "<R>"}
	got := e.Error()
	if !strings.Contains(got, "<L>") || !strings.Contains(got, "<M>") || !strings.Contains(got, "<R>") {
		t.Errorf(`got %q, expected to see "<L>", "<M>" and "<R>"`, got)
	}
}
