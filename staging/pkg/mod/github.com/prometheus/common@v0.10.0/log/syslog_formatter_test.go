// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows,!nacl,!plan9

package log

import (
	"errors"
	"log/syslog"
	"testing"
)

func TestGetFacility(t *testing.T) {
	testCases := []struct {
		facility         string
		expectedPriority syslog.Priority
		expectedErr      error
	}{
		{"0", syslog.LOG_LOCAL0, nil},
		{"1", syslog.LOG_LOCAL1, nil},
		{"2", syslog.LOG_LOCAL2, nil},
		{"3", syslog.LOG_LOCAL3, nil},
		{"4", syslog.LOG_LOCAL4, nil},
		{"5", syslog.LOG_LOCAL5, nil},
		{"6", syslog.LOG_LOCAL6, nil},
		{"7", syslog.LOG_LOCAL7, nil},
		{"8", syslog.LOG_LOCAL0, errors.New("invalid local(8) for syslog")},
	}
	for _, tc := range testCases {
		priority, err := getFacility(tc.facility)
		if err != tc.expectedErr {
			if err.Error() != tc.expectedErr.Error() {
				t.Errorf("want %s, got %s", tc.expectedErr.Error(), err.Error())
			}
		}

		if priority != tc.expectedPriority {
			t.Errorf("want %q, got %q", tc.expectedPriority, priority)
		}
	}
}
