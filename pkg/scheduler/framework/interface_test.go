/*
Copyright 2019 The Kubernetes Authors.

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

package framework

import (
	"errors"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
)

var errorStatus = NewStatus(Error, "internal error")

func TestStatus(t *testing.T) {
	tests := []struct {
		name              string
		status            *Status
		expectedCode      Code
		expectedMessage   string
		expectedIsSuccess bool
		expectedAsError   error
	}{
		{
			name:              "success status",
			status:            NewStatus(Success, ""),
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedAsError:   nil,
		},
		{
			name:              "error status",
			status:            NewStatus(Error, "unknown error"),
			expectedCode:      Error,
			expectedMessage:   "unknown error",
			expectedIsSuccess: false,
			expectedAsError:   errors.New("unknown error"),
		},
		{
			name:              "nil status",
			status:            nil,
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedAsError:   nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.status.Code() != test.expectedCode {
				t.Errorf("expect status.Code() returns %v, but %v", test.expectedCode, test.status.Code())
			}

			if test.status.Message() != test.expectedMessage {
				t.Errorf("expect status.Message() returns %v, but %v", test.expectedMessage, test.status.Message())
			}

			if test.status.IsSuccess() != test.expectedIsSuccess {
				t.Errorf("expect status.IsSuccess() returns %v, but %v", test.expectedIsSuccess, test.status.IsSuccess())
			}

			if test.status.AsError() == test.expectedAsError {
				return
			}

			if test.status.AsError().Error() != test.expectedAsError.Error() {
				t.Errorf("expect status.AsError() returns %v, but %v", test.expectedAsError, test.status.AsError())
			}
		})
	}
}

// The String() method relies on the value and order of the status codes to function properly.
func TestStatusCodes(t *testing.T) {
	assertStatusCode(t, Success, 0)
	assertStatusCode(t, Error, 1)
	assertStatusCode(t, Unschedulable, 2)
	assertStatusCode(t, UnschedulableAndUnresolvable, 3)
	assertStatusCode(t, Wait, 4)
	assertStatusCode(t, Skip, 5)
}

func assertStatusCode(t *testing.T, code Code, value int) {
	if int(code) != value {
		t.Errorf("Status code %q should have a value of %v but got %v", code.String(), value, int(code))
	}
}

func TestPluginToStatusMerge(t *testing.T) {
	tests := []struct {
		name      string
		statusMap PluginToStatus
		wantCode  Code
	}{
		{
			name:      "merge Error and Unschedulable statuses",
			statusMap: PluginToStatus{"p1": NewStatus(Error), "p2": NewStatus(Unschedulable)},
			wantCode:  Error,
		},
		{
			name:      "merge Success and Unschedulable statuses",
			statusMap: PluginToStatus{"p1": NewStatus(Success), "p2": NewStatus(Unschedulable)},
			wantCode:  Unschedulable,
		},
		{
			name:      "merge Success, UnschedulableAndUnresolvable and Unschedulable statuses",
			statusMap: PluginToStatus{"p1": NewStatus(Success), "p2": NewStatus(UnschedulableAndUnresolvable), "p3": NewStatus(Unschedulable)},
			wantCode:  UnschedulableAndUnresolvable,
		},
		{
			name:     "merge nil status",
			wantCode: Success,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			gotStatus := test.statusMap.Merge()
			if test.wantCode != gotStatus.Code() {
				t.Errorf("wantCode %v, gotCode %v", test.wantCode, gotStatus.Code())
			}
		})
	}
}

func TestPreFilterResultMerge(t *testing.T) {
	tests := map[string]struct {
		receiver *PreFilterResult
		in       *PreFilterResult
		want     *PreFilterResult
	}{
		"all nil": {},
		"nil receiver empty input": {
			in:   &PreFilterResult{NodeNames: sets.NewString()},
			want: &PreFilterResult{NodeNames: sets.NewString()},
		},
		"empty receiver nil input": {
			receiver: &PreFilterResult{NodeNames: sets.NewString()},
			want:     &PreFilterResult{NodeNames: sets.NewString()},
		},
		"empty receiver empty input": {
			receiver: &PreFilterResult{NodeNames: sets.NewString()},
			in:       &PreFilterResult{NodeNames: sets.NewString()},
			want:     &PreFilterResult{NodeNames: sets.NewString()},
		},
		"nil receiver populated input": {
			in:   &PreFilterResult{NodeNames: sets.NewString("node1")},
			want: &PreFilterResult{NodeNames: sets.NewString("node1")},
		},
		"empty receiver populated input": {
			receiver: &PreFilterResult{NodeNames: sets.NewString()},
			in:       &PreFilterResult{NodeNames: sets.NewString("node1")},
			want:     &PreFilterResult{NodeNames: sets.NewString()},
		},

		"populated receiver nil input": {
			receiver: &PreFilterResult{NodeNames: sets.NewString("node1")},
			want:     &PreFilterResult{NodeNames: sets.NewString("node1")},
		},
		"populated receiver empty input": {
			receiver: &PreFilterResult{NodeNames: sets.NewString("node1")},
			in:       &PreFilterResult{NodeNames: sets.NewString()},
			want:     &PreFilterResult{NodeNames: sets.NewString()},
		},
		"populated receiver and input": {
			receiver: &PreFilterResult{NodeNames: sets.NewString("node1", "node2")},
			in:       &PreFilterResult{NodeNames: sets.NewString("node2", "node3")},
			want:     &PreFilterResult{NodeNames: sets.NewString("node2")},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := test.receiver.Merge(test.in)
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("unexpected diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestIsStatusEqual(t *testing.T) {
	tests := []struct {
		name string
		x, y *Status
		want bool
	}{
		{
			name: "two nil should be equal",
			x:    nil,
			y:    nil,
			want: true,
		},
		{
			name: "nil should be equal to success status",
			x:    nil,
			y:    NewStatus(Success),
			want: true,
		},
		{
			name: "nil should not be equal with status except success",
			x:    nil,
			y:    NewStatus(Error, "internal error"),
			want: false,
		},
		{
			name: "one status should be equal to itself",
			x:    errorStatus,
			y:    errorStatus,
			want: true,
		},
		{
			name: "same type statuses without reasons should be equal",
			x:    NewStatus(Success),
			y:    NewStatus(Success),
			want: true,
		},
		{
			name: "statuses with same message should be equal",
			x:    NewStatus(Unschedulable, "unschedulable"),
			y:    NewStatus(Unschedulable, "unschedulable"),
			want: true,
		},
		{
			name: "error statuses with same message should not be equal",
			x:    NewStatus(Error, "error"),
			y:    NewStatus(Error, "error"),
			want: false,
		},
		{
			name: "statuses with different reasons should not be equal",
			x:    NewStatus(Unschedulable, "unschedulable"),
			y:    NewStatus(Unschedulable, "unschedulable", "injected filter status"),
			want: false,
		},
		{
			name: "statuses with different codes should not be equal",
			x:    NewStatus(Error, "internal error"),
			y:    NewStatus(Unschedulable, "internal error"),
			want: false,
		},
		{
			name: "wrap error status should be equal with original one",
			x:    errorStatus,
			y:    AsStatus(fmt.Errorf("error: %w", errorStatus.AsError())),
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cmp.Equal(tt.x, tt.y); got != tt.want {
				t.Errorf("cmp.Equal() = %v, want %v", got, tt.want)
			}
		})
	}
}
