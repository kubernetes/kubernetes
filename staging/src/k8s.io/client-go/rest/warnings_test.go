/*
Copyright 2024 The Kubernetes Authors.

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

package rest

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"k8s.io/klog/v2"
)

// TestNoWarningsHandler verifies that no warnings are logged when using the NoWarnings handler
func TestNoWarningsHandler(t *testing.T) {
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	klog.LogToStderr(false)
	defer klog.LogToStderr(true)

	handler := NoWarnings{}
	handler.HandleWarningHeader(299, "agent", "test warning")

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
	capturedOutput := buf.String()

	if capturedOutput != "" {
		t.Errorf("expected no warnings to be logged, got: %s", capturedOutput)
	}
}

// TestWarningLogger verifies that warnings are correctly logged by the WarningLogger handler
func TestWarningLogger(t *testing.T) {
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	klog.LogToStderr(false)
	defer klog.LogToStderr(true)

	handler := WarningLogger{}
	handler.HandleWarningHeader(299, "agent", "test warning")

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
	capturedOutput := buf.String()

	if !strings.Contains(capturedOutput, "test warning") {
		t.Errorf("expected warning to be logged, got: %s", capturedOutput)
	}
}

// TestWarningLoggerNon299 verifies that warnings with non-299 codes are not logged
func TestWarningLoggerNon299(t *testing.T) {
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	klog.LogToStderr(false)
	defer klog.LogToStderr(true)

	handler := WarningLogger{}
	handler.HandleWarningHeader(300, "agent", "test warning")

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
	capturedOutput := buf.String()

	if capturedOutput != "" {
		t.Errorf("expected no warnings to be logged for non-299 code, got: %s", capturedOutput)
	}
}

// TestWarningLoggerEmptyMessage verifies that empty warning messages are not logged
func TestWarningLoggerEmptyMessage(t *testing.T) {
	var buf bytes.Buffer
	klog.SetOutput(&buf)
	klog.LogToStderr(false)
	defer klog.LogToStderr(true)

	handler := WarningLogger{}
	handler.HandleWarningHeader(299, "agent", "")

	klog.Flush()
	klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
	capturedOutput := buf.String()

	if capturedOutput != "" {
		t.Errorf("expected no warnings to be logged for empty message, got: %s", capturedOutput)
	}
}

// TestNewWarningWriter verifies that a new warning writer is correctly created
func TestNewWarningWriter(t *testing.T) {
	var buf bytes.Buffer
	opts := WarningWriterOptions{Deduplicate: true, Color: true}
	writer := NewWarningWriter(&buf, opts)

	if writer == nil {
		t.Error("expected a new warning writer to be created, got nil")
	}

	if !writer.opts.Deduplicate || !writer.opts.Color {
		t.Error("expected options to be set correctly, but they were not")
	}
}

// TestWarningWriter verifies that warnings are correctly written to the output
func TestWarningWriter(t *testing.T) {
	testCases := []struct {
		name              string
		opts              WarningWriterOptions
		warnings          []string
		expectedOutput    string
		expectedWarnCount int
	}{
		{
			name: "single warning, no deduplication, no color",
			opts: WarningWriterOptions{Deduplicate: false, Color: false},
			warnings: []string{
				"test warning",
			},
			expectedOutput:    "Warning: test warning\n",
			expectedWarnCount: 1,
		},
		{
			name: "duplicate warnings, with deduplication",
			opts: WarningWriterOptions{Deduplicate: true, Color: false},
			warnings: []string{
				"test warning", "test warning",
			},
			expectedOutput:    "Warning: test warning\n",
			expectedWarnCount: 1,
		},
		{
			name: "multiple warnings, no deduplication",
			opts: WarningWriterOptions{Deduplicate: false, Color: false},
			warnings: []string{
				"test warning", "another warning",
			},
			expectedOutput:    "Warning: test warning\nWarning: another warning\n",
			expectedWarnCount: 2,
		},
		{
			name: "single warning, with color",
			opts: WarningWriterOptions{Deduplicate: false, Color: true},
			warnings: []string{
				"test warning",
			},
			expectedOutput:    "\u001b[33;1mWarning:\u001b[0m test warning\n",
			expectedWarnCount: 1,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var buf bytes.Buffer
			handler := NewWarningWriter(&buf, testCase.opts)

			for _, warning := range testCase.warnings {
				handler.HandleWarningHeader(299, "agent", warning)
			}

			if buf.String() != testCase.expectedOutput {
				t.Errorf("expected output %q, got: %q", testCase.expectedOutput, buf.String())
			}

			count := handler.WarningCount()
			if count != testCase.expectedWarnCount {
				t.Errorf("expected warning count to be %d, got: %d", testCase.expectedWarnCount, count)
			}
		})
	}
}

// TestHandleWarnings verifies that warnings are correctly handled and processed
func TestHandleWarnings(t *testing.T) {
	testCases := []struct {
		name           string
		headers        http.Header
		expectedOutput string
	}{
		{
			name: "multiple warnings",
			headers: http.Header{
				"Warning": []string{`299 - "test warning"`, `299 - "another warning"`},
			},
			expectedOutput: "Warning: test warning\nWarning: another warning\n",
		},
		{
			name: "single warning",
			headers: http.Header{
				"Warning": []string{`299 - "single warning"`},
			},
			expectedOutput: "Warning: single warning\n",
		},
		{
			name:           "no warnings",
			headers:        http.Header{},
			expectedOutput: "",
		},
		{
			name: "malformed warnings",
			headers: http.Header{
				"Warning": []string{`not a valid warning`},
			},
			expectedOutput: "",
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			var buf bytes.Buffer
			klog.SetOutput(&buf)
			klog.LogToStderr(false)
			defer klog.SetOutput(&bytes.Buffer{}) // prevent further writes into buf
			defer klog.LogToStderr(true)
			handler := NewWarningWriter(&buf, WarningWriterOptions{Deduplicate: false, Color: false})

			handleWarnings(testCase.headers, handler)

			if buf.String() != testCase.expectedOutput {
				t.Errorf("expected output %q, got: %q", testCase.expectedOutput, buf.String())
			}
		})
	}
}

// TestSetDefaultWarningHandler verifies that the default warning handler can be set and retrieved
func TestSetDefaultWarningHandler(t *testing.T) {
	handler := NoWarnings{}
	SetDefaultWarningHandler(handler)
	if getDefaultWarningHandler() != handler {
		t.Errorf("expected default handler to be set")
	}
}

// TestWarningWriterWarningCount verifies that the warning count is correctly incremented when deduplication is enabled
func TestWarningWriterWarningCount(t *testing.T) {
	var buf bytes.Buffer
	handler := NewWarningWriter(&buf, WarningWriterOptions{Deduplicate: true, Color: false})

	handler.HandleWarningHeader(299, "agent", "test warning")
	handler.HandleWarningHeader(299, "agent", "test warning")
	count := handler.WarningCount()
	if count != 1 {
		t.Errorf("expected warning count to be 1, got: %d", count)
	}
}
