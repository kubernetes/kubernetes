/*
Copyright The Kubernetes Authors.

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

package gce

import (
	"reflect"
	"strings"
	"testing"
)

func TestPickNewestImage(t *testing.T) {
	img := func(name, family, ts string) gceImage {
		return gceImage{Name: name, Family: family, CreationTimestamp: ts}
	}
	tests := []struct {
		name        string
		images      []gceImage
		imageRegex  string
		imageFamily string
		want        string
		wantErr     string
	}{
		{
			name: "newest of the family wins",
			images: []gceImage{
				img("img-old", "fam", "2026-08-01T10:00:00Z"),
				img("img-new", "fam", "2026-08-03T10:00:00Z"),
				img("img-mid", "fam", "2026-08-02T10:00:00Z"),
			},
			imageFamily: "fam",
			want:        "img-new",
		},
		{
			name: "images of other families are ignored",
			images: []gceImage{
				img("other-newer", "other", "2026-08-09T10:00:00Z"),
				img("fam-new", "fam", "2026-08-03T10:00:00Z"),
				img("fam-old", "fam", "2026-08-01T10:00:00Z"),
			},
			imageFamily: "fam",
			want:        "fam-new",
		},
		{
			name: "regex keeps only matching names, even a newer non-match is skipped",
			images: []gceImage{
				img("keep-v2", "fam", "2026-08-03T10:00:00Z"),
				img("skip-v1", "fam", "2026-08-09T10:00:00Z"),
			},
			imageFamily: "fam",
			imageRegex:  "keep-.*",
			want:        "keep-v2",
		},
		{
			name: "regex without a family",
			images: []gceImage{
				img("keep-old", "", "2026-08-01T10:00:00Z"),
				img("skip-newest", "", "2026-08-09T10:00:00Z"),
				img("keep-new", "", "2026-08-03T10:00:00Z"),
			},
			imageRegex: "keep-.*",
			want:       "keep-new",
		},
		{
			name: "no match returns an error",
			images: []gceImage{
				img("other", "other", "2026-08-01T10:00:00Z"),
			},
			imageFamily: "fam",
			wantErr:     "found zero images",
		},
		{
			name: "a malformed timestamp returns an error",
			images: []gceImage{
				img("fam-x", "fam", "not-a-timestamp"),
			},
			imageFamily: "fam",
			wantErr:     "failed to parse instance creation timestamp",
		},
		{
			name: "an invalid image_regex returns an error",
			images: []gceImage{
				img("img", "fam", "2026-08-01T10:00:00Z"),
			},
			imageRegex: "[",
			wantErr:    "failed to compile image_regex",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := pickNewestImage(tc.images, tc.imageRegex, tc.imageFamily, "proj")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("pickNewestImage() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("pickNewestImage() unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("pickNewestImage() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestGCEImageListArgs(t *testing.T) {
	tests := []struct {
		name        string
		project     string
		imageFamily string
		want        []string
	}{
		{
			name:        "a family adds a server-side filter",
			project:     "proj",
			imageFamily: "fam",
			want:        []string{"compute", "images", "list", "--format=json", "--project=proj", "--filter=family=fam"},
		},
		{
			name:    "no family adds no filter",
			project: "proj",
			want:    []string{"compute", "images", "list", "--format=json", "--project=proj"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := gceImageListArgs(tc.project, tc.imageFamily); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("gceImageListArgs() = %q, want %q", got, tc.want)
			}
		})
	}
}
