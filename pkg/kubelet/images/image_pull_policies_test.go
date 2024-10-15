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

package images

import (
	"reflect"
	"testing"
)

func TestNeverVerifyPreloadedPullPolicy(t *testing.T) {
	tests := []struct {
		name              string
		imageRecordsExist bool
		want              bool
	}{
		{
			name:              "there are no records about the image being pulled",
			imageRecordsExist: false,
			want:              false,
		},
		{
			name:              "there are records about the image being pulled",
			imageRecordsExist: true,
			want:              true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NeverVerifyPreloadedPullPolicy()("test-image", tt.imageRecordsExist); got != tt.want {
				t.Errorf("NeverVerifyPreloadedPullPolicy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewNeverVerifyAllowListedPullPolicy(t *testing.T) {
	tests := []struct {
		name              string
		imageRecordsExist bool
		allowlist         []string
		expectedAbsolutes int
		expectedWildcards int
		want              bool
		wantErr           bool
	}{
		{
			name:              "there are no records about the image being pulled, not in allowlist",
			imageRecordsExist: false,
			want:              true,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/image3"},
			expectedAbsolutes: 3,
		},
		{
			name:              "there are records about the image being pulled, not in allowlist",
			imageRecordsExist: true,
			want:              true,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image3", "test.io/test/image2", "test.io/test/image3"},
			expectedAbsolutes: 3,
		},
		{
			name:              "there are no records about the image being pulled, appears in allowlist",
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/test-image", "test.io/test/image3"},
			expectedAbsolutes: 4,
		},
		{
			name:              "there are records about the image being pulled, appears in allowlist",
			imageRecordsExist: true,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/test-image", "test.io/test/image3"},
			expectedAbsolutes: 4,
		},
		{
			name:      "invalid allowlist pattern - wildcard in the middle",
			wantErr:   true,
			allowlist: []string{"image.repo/pokus*/imagename"},
		},
		{
			name:      "invalid allowlist pattern - trailing non-segment wildcard middle",
			wantErr:   true,
			allowlist: []string{"image.repo/pokus*"},
		},
		{
			name:      "invalid allowlist pattern - wildcard path segment in the middle",
			wantErr:   true,
			allowlist: []string{"image.repo/*/imagename"},
		},
		{
			name:      "invalid allowlist pattern - only wildcard segment",
			wantErr:   true,
			allowlist: []string{"/*"},
		},
		{
			name:      "invalid allowlist pattern - ends with a '/'",
			wantErr:   true,
			allowlist: []string{"image.repo/"},
		},
		{
			name:      "invalid allowlist pattern - empty",
			wantErr:   true,
			allowlist: []string{""},
		},
		{
			name:      "invalid allowlist pattern - asterisk",
			wantErr:   true,
			allowlist: []string{"*"},
		},
		{
			name:      "invalid allowlist pattern - image with a tag",
			wantErr:   true,
			allowlist: []string{"test.io/test/image1:tagged"},
		},
		{
			name:      "invalid allowlist pattern - image with a digest",
			wantErr:   true,
			allowlist: []string{"test.io/test/image1@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd"},
		},
		{
			name:      "invalid allowlist pattern - trailing whitespace",
			wantErr:   true,
			allowlist: []string{"test.io/test/image1 "},
		},
		{
			name:              "there are no records about the image being pulled, not in allowlist - different repo wildcard",
			imageRecordsExist: false,
			want:              true,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "different.repo/test/*"},
			expectedAbsolutes: 2,
			expectedWildcards: 1,
		},
		{
			name:              "there are no records about the image being pulled, not in allowlist - matches org wildcard",
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/*"},
			expectedAbsolutes: 2,
			expectedWildcards: 1,
		},
		{
			name:              "there are no records about the image being pulled, not in allowlist - matches repo wildcard",
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/*"},
			expectedAbsolutes: 2,
			expectedWildcards: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			policyEnforcer, err := NewNeverVerifyAllowListedPullPolicy(tt.allowlist)
			if tt.wantErr != (err != nil) {
				t.Fatalf("wanted error: %t, got: %v", tt.wantErr, err)
			}

			if err != nil {
				return
			}

			if len(policyEnforcer.absoluteURLs) != tt.expectedAbsolutes {
				t.Errorf("expected %d of absolute image URLs in the allowlist policy, got %d: %v", tt.expectedAbsolutes, len(policyEnforcer.absoluteURLs), policyEnforcer.absoluteURLs)
			}

			if len(policyEnforcer.prefixes) != tt.expectedWildcards {
				t.Errorf("expected %d of wildcard image URLs in the allowlist policy, got %d: %v", tt.expectedWildcards, len(policyEnforcer.prefixes), policyEnforcer.prefixes)
			}

			got := policyEnforcer.RequireCredentialVerificationForImage("test.io/test/test-image", tt.imageRecordsExist)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewNeverVerifyAllowListedPullPolicy() = %v, want %v", got, tt.want)
			}
		})
	}
}
