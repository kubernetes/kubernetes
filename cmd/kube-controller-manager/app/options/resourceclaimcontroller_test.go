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

package options

import (
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"

	resourceclaimconfig "k8s.io/kubernetes/pkg/controller/resourceclaim/config"
)

func TestResourceClaimControllerOptions_AddFlags(t *testing.T) {
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	opts := &ResourceClaimControllerOptions{
		&resourceclaimconfig.ResourceClaimControllerConfiguration{
			ConcurrentResourceClaimSyncs: 50,
		},
	}

	opts.AddFlags(fs)

	// Test that the flag was added
	flag := fs.Lookup("concurrent-resourceclaim-syncs")
	if flag == nil {
		t.Error("concurrent-resourceclaim-syncs flag was not added")
	}

	// Test that the flag has the correct default value
	if flag.DefValue != "50" {
		t.Errorf("expected default value 50, got %s", flag.DefValue)
	}

	// Test flag parsing
	args := []string{"--concurrent-resourceclaim-syncs=25"}
	if err := fs.Parse(args); err != nil {
		t.Errorf("failed to parse flags: %v", err)
	}

	if opts.ConcurrentResourceClaimSyncs != 25 {
		t.Errorf("expected ConcurrentResourceClaimSyncs to be 25, got %d", opts.ConcurrentResourceClaimSyncs)
	}
}

func TestResourceClaimControllerOptions_AddFlags_Nil(t *testing.T) {
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	var opts *ResourceClaimControllerOptions

	// Should not panic when options is nil
	opts.AddFlags(fs)

	// Flag should not be added
	flag := fs.Lookup("concurrent-resourceclaim-syncs")
	if flag != nil {
		t.Error("concurrent-resourceclaim-syncs flag should not be added when options is nil")
	}
}

func TestResourceClaimControllerOptions_ApplyTo(t *testing.T) {
	opts := &ResourceClaimControllerOptions{
		&resourceclaimconfig.ResourceClaimControllerConfiguration{
			ConcurrentResourceClaimSyncs: 75,
		},
	}

	cfg := &resourceclaimconfig.ResourceClaimControllerConfiguration{}

	err := opts.ApplyTo(cfg)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if cfg.ConcurrentResourceClaimSyncs != 75 {
		t.Errorf("expected ConcurrentResourceClaimSyncs to be 75, got %d", cfg.ConcurrentResourceClaimSyncs)
	}
}

func TestResourceClaimControllerOptions_ApplyTo_Nil(t *testing.T) {
	var opts *ResourceClaimControllerOptions
	cfg := &resourceclaimconfig.ResourceClaimControllerConfiguration{
		ConcurrentResourceClaimSyncs: 50,
	}

	err := opts.ApplyTo(cfg)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Configuration should remain unchanged
	if cfg.ConcurrentResourceClaimSyncs != 50 {
		t.Errorf("expected ConcurrentResourceClaimSyncs to remain 50, got %d", cfg.ConcurrentResourceClaimSyncs)
	}
}

func TestResourceClaimControllerOptions_Validate(t *testing.T) {
	testCases := []struct {
		name                         string
		concurrentResourceClaimSyncs int32
		expectErrors                 bool
		expectedErrorSubString       string
	}{
		{
			name:                         "valid concurrent syncs",
			concurrentResourceClaimSyncs: 50,
			expectErrors:                 false,
		},
		{
			name:                         "valid minimum concurrent syncs",
			concurrentResourceClaimSyncs: 1,
			expectErrors:                 false,
		},
		{
			name:                         "invalid zero concurrent syncs",
			concurrentResourceClaimSyncs: 0,
			expectErrors:                 true,
			expectedErrorSubString:       "concurrent-resourceclaim-syncs must be greater than 0",
		},
		{
			name:                         "invalid negative concurrent syncs",
			concurrentResourceClaimSyncs: -5,
			expectErrors:                 true,
			expectedErrorSubString:       "concurrent-resourceclaim-syncs must be greater than 0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			opts := &ResourceClaimControllerOptions{
				&resourceclaimconfig.ResourceClaimControllerConfiguration{
					ConcurrentResourceClaimSyncs: tc.concurrentResourceClaimSyncs,
				},
			}

			errs := opts.Validate()

			if tc.expectErrors && len(errs) == 0 {
				t.Error("expected validation errors, but got none")
			}

			if !tc.expectErrors && len(errs) > 0 {
				t.Errorf("expected no validation errors, but got: %v", errs)
			}

			if tc.expectErrors && len(errs) > 0 {
				gotErr := utilerrors.NewAggregate(errs).Error()
				if !strings.Contains(gotErr, tc.expectedErrorSubString) {
					t.Errorf("expected error to contain %q, but got %q", tc.expectedErrorSubString, gotErr)
				}
			}
		})
	}
}

func TestResourceClaimControllerOptions_Validate_Nil(t *testing.T) {
	var opts *ResourceClaimControllerOptions

	errs := opts.Validate()
	if len(errs) != 0 {
		t.Errorf("expected no validation errors for nil options, but got: %v", errs)
	}
}

func TestResourceClaimControllerOptions_Integration(t *testing.T) {
	// Test the complete workflow: create options, set flags, apply to config
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	opts := &ResourceClaimControllerOptions{
		&resourceclaimconfig.ResourceClaimControllerConfiguration{
			ConcurrentResourceClaimSyncs: 50,
		},
	}

	// Add flags
	opts.AddFlags(fs)

	// Parse flags with custom value
	args := []string{"--concurrent-resourceclaim-syncs=100"}
	if err := fs.Parse(args); err != nil {
		t.Fatalf("failed to parse flags: %v", err)
	}

	// Validate
	errs := opts.Validate()
	if len(errs) > 0 {
		t.Fatalf("validation failed: %v", errs)
	}

	// Apply to config
	cfg := &resourceclaimconfig.ResourceClaimControllerConfiguration{}
	if err := opts.ApplyTo(cfg); err != nil {
		t.Fatalf("failed to apply options: %v", err)
	}

	// Verify final configuration
	expected := &resourceclaimconfig.ResourceClaimControllerConfiguration{
		ConcurrentResourceClaimSyncs: 100,
	}

	if !reflect.DeepEqual(cfg, expected) {
		t.Errorf("expected config %+v, got %+v", expected, cfg)
	}
}