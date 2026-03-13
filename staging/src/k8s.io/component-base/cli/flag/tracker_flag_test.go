/*
Copyright 2025 The Kubernetes Authors.

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

package flag

import (
	"fmt"
	"testing"

	"github.com/spf13/pflag"
)

func TestNewTracker(t *testing.T) {
	tests := []struct {
		name     string
		value    pflag.Value
		provided *bool
		wantType string
	}{
		{
			name:     "non-bool-tracker",
			value:    &nonBoolFlagMockValue{val: "initial", typ: "string"},
			provided: new(bool),
			wantType: "string",
		},
		{
			name:     "bool-tracker",
			value:    &boolFlagMockValue{val: "false", typ: "bool", isBool: true},
			provided: new(bool),
			wantType: "bool",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := NewTracker(tt.value, tt.provided)

			if tracker.Type() != tt.wantType {
				t.Errorf("Want type %s, got %s", tt.wantType, tracker.Type())
			}

			if trackerValue, ok := tracker.(*TrackerValue); ok {
				if trackerValue.provided != tt.provided {
					t.Errorf("Provided pointer not stored correctly in TrackerValue")
				}
			} else if boolTrackerValue, ok := tracker.(*BoolTrackerValue); ok {
				if boolTrackerValue.provided != tt.provided {
					t.Errorf("Provided pointer not stored correctly in BoolTrackerValue")
				}
			}
		})
	}
}

func TestNewTrackerPanics(t *testing.T) {
	tests := []struct {
		name     string
		value    pflag.Value
		provided *bool
		panicMsg string
	}{
		{
			name:     "nil-value",
			value:    nil,
			provided: new(bool),
			panicMsg: "value must not be nil",
		},
		{
			name:     "nil-provided",
			value:    &boolFlagMockValue{val: "test"},
			provided: nil,
			panicMsg: "provided boolean must not be nil",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic, but did not panic")
				} else if r != tt.panicMsg {
					t.Errorf("expected panic message %q, got %q", tt.panicMsg, r)
				}
			}()
			NewTracker(tt.value, tt.provided)
		})
	}
}

func TestTrackerValue_String(t *testing.T) {
	testCases := []struct {
		name      string
		mockValue pflag.Value
		want      string
	}{
		{
			name:      "bool-flag",
			mockValue: &boolFlagMockValue{val: "bool-test"},
			want:      "bool-test",
		},
		{
			name:      "non-bool-flag",
			mockValue: &nonBoolFlagMockValue{val: "non-bool-test"},
			want:      "non-bool-test",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tracker := NewTracker(tc.mockValue, new(bool))
			result := tracker.String()
			if result != tc.want {
				t.Errorf("Want %q, but got %q", tc.want, result)
			}
		})
	}
}

func TestTrackerValue_Set(t *testing.T) {
	testCases := []struct {
		name         string
		mockValue    pflag.Value
		provided     *bool
		mockErr      error
		wantProvided bool
		wantErr      bool
	}{
		{
			name:         "success-bool-flag",
			mockValue:    &boolFlagMockValue{val: "bool-test"},
			provided:     new(bool),
			wantProvided: true,
			wantErr:      false,
		},
		{
			name:         "success-non-bool-flag",
			mockValue:    &nonBoolFlagMockValue{val: "bool-test"},
			provided:     new(bool),
			wantProvided: true,
			wantErr:      false,
		},
		{
			name:         "error-bool-flag",
			mockValue:    &boolFlagMockValue{val: "bool-test", err: fmt.Errorf("set error")},
			provided:     new(bool),
			wantProvided: false,
			wantErr:      true,
		},
		{
			name:         "error-non-bool-flag",
			mockValue:    &nonBoolFlagMockValue{val: "bool-test", err: fmt.Errorf("set error")},
			provided:     new(bool),
			wantProvided: false,
			wantErr:      true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tracker := NewTracker(tc.mockValue, tc.provided)
			err := tracker.Set("new value")

			if (err != nil) != tc.wantErr {
				t.Errorf("Want error: %v, got: %v", tc.wantErr, err != nil)
			}

			if *tc.provided != tc.wantProvided {
				t.Errorf("Want provided to be %v, got: %v", tc.wantProvided, *tc.provided)
			}
		})
	}
}

func TestTrackerValue_MultipleSetCalls(t *testing.T) {
	provided := false
	mock := &boolFlagMockValue{val: "initial"}
	tracker := NewTracker(mock, &provided)

	err := tracker.Set("new value")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if mock.val != "new value" {
		t.Errorf("Expected mock value to be 'new value', got '%s'", mock.val)
	}
	if !provided {
		t.Error("Expected 'provided' to be true, got false")
	}

	provided = false // reset
	mock.err = fmt.Errorf("set error")
	err = tracker.Set("failed set")

	if err == nil {
		t.Errorf("Expected an error, got nil")
	}
	if provided {
		t.Error("Expected 'provided' to be false after error, got true")
	}
}

func TestTrackerValue_Type(t *testing.T) {
	testCases := []struct {
		name      string
		mockValue pflag.Value
		want      string
	}{
		{
			name:      "success-bool-flag",
			mockValue: &boolFlagMockValue{typ: "mockBoolType"},
			want:      "mockBoolType",
		},
		{
			name:      "success-non-bool-flag",
			mockValue: &nonBoolFlagMockValue{typ: "mockNonBoolType"},
			want:      "mockNonBoolType",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tracker := NewTracker(tc.mockValue, new(bool))
			result := tracker.Type()
			if result != tc.want {
				t.Errorf("Want %q, but got %q", tc.want, result)
			}
		})
	}
}

type boolFlagMockValue struct {
	val    string
	typ    string
	isBool bool
	err    error
}

func (m *boolFlagMockValue) String() string {
	return m.val
}

func (m *boolFlagMockValue) Set(value string) error {
	m.val = value
	return m.err
}

func (m *boolFlagMockValue) Type() string {
	return m.typ
}

func (m *boolFlagMockValue) IsBoolFlag() bool {
	return m.isBool
}

type nonBoolFlagMockValue struct {
	val string
	typ string
	err error
}

func (m *nonBoolFlagMockValue) String() string {
	return m.val
}

func (m *nonBoolFlagMockValue) Set(value string) error {
	m.val = value
	return m.err
}

func (m *nonBoolFlagMockValue) Type() string {
	return m.typ
}
