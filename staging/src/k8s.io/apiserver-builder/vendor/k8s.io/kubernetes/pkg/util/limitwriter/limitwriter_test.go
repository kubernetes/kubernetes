/*
Copyright 2016 The Kubernetes Authors.

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

package limitwriter

import (
	"reflect"
	"testing"
)

type recordingWriter struct {
	Wrote [][]byte
}

func (r *recordingWriter) Write(data []byte) (int, error) {
	r.Wrote = append(r.Wrote, data)
	return len(data), nil
}

func TestLimitWriter(t *testing.T) {
	testcases := map[string]struct {
		Limit  int64
		Writes [][]byte

		ExpectedRecordedWrites [][]byte
		ExpectedReportedWrites []int
		ExpectedReportedErrors []error
	}{
		"empty": {},

		"empty write": {
			Limit:                  1000,
			Writes:                 [][]byte{{}},
			ExpectedRecordedWrites: [][]byte{{}},
			ExpectedReportedWrites: []int{0},
			ExpectedReportedErrors: []error{nil},
		},

		"unlimited write": {
			Limit:                  1000,
			Writes:                 [][]byte{[]byte(`foo`)},
			ExpectedRecordedWrites: [][]byte{[]byte(`foo`)},
			ExpectedReportedWrites: []int{3},
			ExpectedReportedErrors: []error{nil},
		},

		"limited write": {
			Limit:                  5,
			Writes:                 [][]byte{[]byte(``), []byte(`1`), []byte(`23`), []byte(`456789`), []byte(`10`), []byte(``)},
			ExpectedRecordedWrites: [][]byte{[]byte(``), []byte(`1`), []byte(`23`), []byte(`45`)},
			ExpectedReportedWrites: []int{0, 1, 2, 2, 0, 0},
			ExpectedReportedErrors: []error{nil, nil, nil, ErrMaximumWrite, ErrMaximumWrite, ErrMaximumWrite},
		},
	}

	for k, tc := range testcases {
		var reportedWrites []int
		var reportedErrors []error

		recordingWriter := &recordingWriter{}

		limitwriter := New(recordingWriter, tc.Limit)

		for _, w := range tc.Writes {
			n, err := limitwriter.Write(w)
			reportedWrites = append(reportedWrites, n)
			reportedErrors = append(reportedErrors, err)
		}

		if !reflect.DeepEqual(recordingWriter.Wrote, tc.ExpectedRecordedWrites) {
			t.Errorf("%s: expected recorded writes %v, got %v", k, tc.ExpectedRecordedWrites, recordingWriter.Wrote)
		}
		if !reflect.DeepEqual(reportedWrites, tc.ExpectedReportedWrites) {
			t.Errorf("%s: expected reported writes %v, got %v", k, tc.ExpectedReportedWrites, reportedWrites)
		}
		if !reflect.DeepEqual(reportedErrors, tc.ExpectedReportedErrors) {
			t.Errorf("%s: expected reported errors %v, got %v", k, tc.ExpectedReportedErrors, reportedErrors)
		}
	}
}
