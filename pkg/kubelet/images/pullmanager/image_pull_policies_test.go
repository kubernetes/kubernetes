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

package pullmanager

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
			if got := NeverVerifyPreloadedPullPolicy()("test-image:sometag", tt.imageRecordsExist); got != tt.want {
				t.Errorf("NeverVerifyPreloadedPullPolicy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewNeverVerifyAllowListedPullPolicy(t *testing.T) {
	const basicTestImage = "test.io/test/test-image:tag"

	tests := []struct {
		name              string
		imageRecordsExist bool
		inputImage        string
		allowlist         []string
		expectedAbsolutes int
		expectedWildcards int
		want              bool
		wantErr           bool
	}{
		{
			name:              "there are no records about the image being pulled, not in allowlist",
			imageRecordsExist: false,
			inputImage:        basicTestImage,
			want:              true,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/image3"},
			expectedAbsolutes: 3,
		},
		{
			name:              "there are records about the image being pulled, not in allowlist",
			imageRecordsExist: true,
			inputImage:        basicTestImage,
			want:              true,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image3", "test.io/test/image2", "test.io/test/image3"},
			expectedAbsolutes: 3,
		},
		{
			name:              "there are no records about the image being pulled, appears in allowlist",
			imageRecordsExist: false,
			inputImage:        basicTestImage,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/test-image", "test.io/test/image3"},
			expectedAbsolutes: 4,
		},
		{
			name:              "there are records about the image being pulled, appears in allowlist",
			imageRecordsExist: true,
			inputImage:        basicTestImage,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/test-image", "test.io/test/image3"},
			expectedAbsolutes: 4,
		},
		{
			name:       "invalid allowlist pattern - wildcard in the middle",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"image.repo/pokus*/imagename"},
		},
		{
			name:       "invalid allowlist pattern - trailing non-segment wildcard middle",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"image.repo/pokus*"},
		},
		{
			name:       "invalid allowlist pattern - wildcard path segment in the middle",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"image.repo/*/imagename"},
		},
		{
			name:       "invalid allowlist pattern - only wildcard segment",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"/*"},
		},
		{
			name:       "invalid allowlist pattern - ends with a '/'",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"image.repo/"},
		},
		{
			name:       "invalid allowlist pattern - empty",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{""},
		},
		{
			name:       "invalid allowlist pattern - asterisk",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"*"},
		},
		{
			name:       "invalid allowlist pattern - image with a tag",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"test.io/test/image1:tagged"},
		},
		{
			name:       "invalid allowlist pattern - image with a digest",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"test.io/test/image1@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd"},
		},
		{
			name:       "invalid allowlist pattern - trailing whitespace",
			inputImage: basicTestImage,
			wantErr:    true,
			allowlist:  []string{"test.io/test/image1 "},
		},
		{
			name:              "there are no records about the image being pulled, not in allowlist - different repo wildcard",
			inputImage:        basicTestImage,
			imageRecordsExist: false,
			want:              true,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "different.repo/test/*"},
			expectedAbsolutes: 2,
			expectedWildcards: 1,
		},
		{
			name:              "there are no records about the image being pulled, not in allowlist - matches org wildcard",
			inputImage:        basicTestImage,
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/test/*"},
			expectedAbsolutes: 2,
			expectedWildcards: 1,
		},
		{
			name:              "there are no records about the image being pulled, not in allowlist - matches repo wildcard",
			inputImage:        basicTestImage,
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"test.io/test/image1", "test.io/test/image2", "test.io/*"},
			expectedAbsolutes: 2,
			expectedWildcards: 1,
		},
		{
			name:              "'familiar' image name with docker-disambiguated fqdn in the allowlist - not a match",
			inputImage:        "ubuntu",
			imageRecordsExist: false,
			want:              true,
			allowlist:         []string{"docker.io/library/ubuntu"},
			expectedAbsolutes: 1,
		},
		{
			name:              "'familiar' image name with docker-disambiguated wildcard in the allowlist - not a match",
			inputImage:        "ubuntu",
			imageRecordsExist: false,
			want:              true,
			allowlist:         []string{"docker.io/library/*"},
			expectedWildcards: 1,
		},
		{
			name:              "'familiar' image name with 'familiar' image matching it in the allowlist",
			inputImage:        "ubuntu",
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"ubuntu"},
			expectedAbsolutes: 1,
		},
		{
			name:              "'familiar' image name with a tag with 'familiar' image matching it in the allowlist",
			inputImage:        "ubuntu:tagged",
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"ubuntu"},
			expectedAbsolutes: 1,
		},
		{
			name:              "'familiar' image name with a digest with 'familiar' image matching it in the allowlist",
			inputImage:        "ubuntu@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			imageRecordsExist: false,
			want:              false,
			allowlist:         []string{"ubuntu"},
			expectedAbsolutes: 1,
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

			got := policyEnforcer.RequireCredentialVerificationForImage(tt.inputImage, tt.imageRecordsExist)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewNeverVerifyAllowListedPullPolicy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTrimImageTagDigest(t *testing.T) {
	tests := []struct {
		name    string
		image   string
		want    string
		wantErr bool
	}{
		{
			name:  "'familiar' image name, no tag or digest",
			image: "myimage",
			want:  "myimage",
		},
		{
			name:  "'familiar' image name with tag",
			image: "myimage:latest",
			want:  "myimage",
		},
		{
			name:  "'familiar' image name with digest",
			image: "myimage@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "myimage",
		},
		{
			name:  "'familiar' image name with tag and digest",
			image: "myimage:v1@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "myimage",
		},
		{
			name:  "single path segment, no tag",
			image: "registry.io/myimage",
			want:  "registry.io/myimage",
		},
		{
			name:  "single path segment with tag",
			image: "registry.io/myimage:v1.0",
			want:  "registry.io/myimage",
		},
		{
			name:  "single path segment with digest",
			image: "registry.io/myimage@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "registry.io/myimage",
		},
		{
			name:  "single path segment with tag and digest",
			image: "registry.io/myimage:latest@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "registry.io/myimage",
		},
		{
			name:  "domain and two path segments, no tag",
			image: "registry.io/myorg/myimage",
			want:  "registry.io/myorg/myimage",
		},
		{
			name:  "domain and two path segments with tag",
			image: "registry.io/myorg/myimage:v2",
			want:  "registry.io/myorg/myimage",
		},
		{
			name:  "domain and two path segments with digest",
			image: "registry.io/myorg/myimage@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "registry.io/myorg/myimage",
		},
		{
			name:  "domain and two path segments with tag and digest",
			image: "registry.io/myorg/myimage:v2@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "registry.io/myorg/myimage",
		},
		{
			name:  "deep path and tag",
			image: "registry.io/a/b/c/myimage:latest",
			want:  "registry.io/a/b/c/myimage",
		},
		{
			name:    "image name ends with slash",
			image:   "registry.io/myorg/",
			wantErr: true,
		},
		{
			name:    "last segment is only a tag",
			image:   "registry.io/myorg/:tag",
			wantErr: true,
		},
		{
			name:    "last segment is only a digest",
			image:   "registry.io/myorg/@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			wantErr: true,
		},
		{
			name:  "domain with port and tag",
			image: "localhost:5000/myimage:v1",
			want:  "localhost:5000/myimage",
		},
		{
			name:  "domain with port, org, and tag",
			image: "myregistry:5000/myorg/myimage:v1",
			want:  "myregistry:5000/myorg/myimage",
		},
		{
			name:  "multiple occurrences of tag",
			image: "docker.io/taghere/image:taghere",
			want:  "docker.io/taghere/image",
		},
		{
			name:  "multiple occurrences of digest",
			image: "registry.io/38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd/myorg/image@sha256:38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd",
			want:  "registry.io/38a8906435c4dd5f4258899d46621bfd8eea3ad6ff494ee3c2f17ef0321625bd/myorg/image",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := trimImageTagDigest(tt.image)
			if (err != nil) != tt.wantErr {
				t.Fatalf("removeTagDigest(%q) error = %v, wantErr %v", tt.image, err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("removeTagDigest(%q) = %q, want %q", tt.image, got, tt.want)
			}
		})
	}
}
