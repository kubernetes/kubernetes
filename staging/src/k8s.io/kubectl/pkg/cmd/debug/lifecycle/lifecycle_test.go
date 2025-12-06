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

package lifecycle

import (
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestLifecycleOptionsValidate(t *testing.T) {
	tests := []struct {
		name        string
		opts        *LifecycleOptions
		expectError bool
	}{
		{
			name:        "valid options",
			opts:        &LifecycleOptions{PodName: "test-pod"},
			expectError: false,
		},
		{
			name:        "valid options with json output",
			opts:        &LifecycleOptions{PodName: "test-pod", OutputFormat: "json"},
			expectError: false,
		},
		{
			name:        "valid options with yaml output",
			opts:        &LifecycleOptions{PodName: "test-pod", OutputFormat: "yaml"},
			expectError: false,
		},
		{
			name:        "missing pod name",
			opts:        &LifecycleOptions{PodName: ""},
			expectError: true,
		},
		{
			name:        "invalid output format",
			opts:        &LifecycleOptions{PodName: "test-pod", OutputFormat: "xml"},
			expectError: true,
		},
		{
			name:        "invalid output format table",
			opts:        &LifecycleOptions{PodName: "test-pod", OutputFormat: "table"},
			expectError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.opts.Validate()
			if tc.expectError && err == nil {
				t.Error("expected error but got none")
			}
			if !tc.expectError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestNewLifecycleOptions(t *testing.T) {
	streams := genericiooptions.NewTestIOStreamsDiscard()
	opts := NewLifecycleOptions(streams)

	if opts == nil {
		t.Fatal("expected non-nil options")
	}

	if opts.IOStreams.Out != streams.Out {
		t.Error("expected IOStreams to be set correctly")
	}
}

func TestNewCmdLifecycle(t *testing.T) {
	streams := genericiooptions.NewTestIOStreamsDiscard()
	cmd := NewCmdLifecycle(nil, streams)

	if cmd == nil {
		t.Fatal("expected non-nil command")
	}

	if cmd.Use != "lifecycle POD" {
		t.Errorf("unexpected Use: %s", cmd.Use)
	}

	// Check flags exist
	showDetailsFlag := cmd.Flags().Lookup("show-details")
	if showDetailsFlag == nil {
		t.Error("expected show-details flag to exist")
	}

	outputFlag := cmd.Flags().Lookup("output")
	if outputFlag == nil {
		t.Error("expected output flag to exist")
	}

	// Verify short flag alias for output
	if outputFlag.Shorthand != "o" {
		t.Errorf("expected output shorthand 'o', got %s", outputFlag.Shorthand)
	}
}
