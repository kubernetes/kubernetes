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
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"k8s.io/kubernetes/test/e2e_node/remote"
)

func TestGetGCEImage(t *testing.T) {
	image := func(name, family, created string) gceImage {
		return gceImage{Name: name, Family: family, CreationTimestamp: created}
	}
	images := []gceImage{
		image("ubuntu-v20260801", "family", "2026-08-03T10:00:00Z"),
		image("ubuntu-v20260801-cgroupsv1", "family", "2026-08-09T10:00:00Z"),
		image("ubuntu-newer", "other-family", "2026-08-10T10:00:00Z"),
	}
	listErr := errors.New("fixture list error")
	tests := []struct {
		name      string
		images    []gceImage
		include   string
		exclude   string
		family    string
		response  []byte
		listErr   error
		want      string
		wantErr   string
		wantCause error
		wantCalls int
	}{
		{
			name:      "exclude keeps a valid name ending in 1",
			images:    images,
			exclude:   "-cgroupsv1$",
			family:    "family",
			want:      "ubuntu-v20260801",
			wantCalls: 1,
		},
		{
			name: "exclude can match within an image name",
			images: []gceImage{
				image("ubuntu-stable", "family", "2026-08-03T10:00:00Z"),
				image("ubuntu-cgroupsv1-preview", "family", "2026-08-09T10:00:00Z"),
			},
			exclude:   "cgroupsv1",
			family:    "family",
			want:      "ubuntu-stable",
			wantCalls: 1,
		},
		{
			name:      "include and exclude both apply",
			images:    images,
			include:   "^ubuntu-v",
			exclude:   "-cgroupsv1$",
			want:      "ubuntu-v20260801",
			wantCalls: 1,
		},
		{
			name:      "empty exclude retains include behavior",
			images:    images,
			include:   "^ubuntu-v",
			family:    "family",
			want:      "ubuntu-v20260801-cgroupsv1",
			wantCalls: 1,
		},
		{
			name:      "family matching is exact",
			images:    images,
			family:    "family",
			want:      "ubuntu-v20260801-cgroupsv1",
			wantCalls: 1,
		},
		{
			name: "newest timestamp wins regardless of input order",
			images: []gceImage{
				image("new", "family", "2026-08-09T10:00:00Z"),
				image("old", "family", "2026-08-03T10:00:00Z"),
			},
			family:    "family",
			want:      "new",
			wantCalls: 1,
		},
		{
			name: "excluded timestamps are not parsed",
			images: []gceImage{
				image("ubuntu-cgroupsv1", "family", "invalid"), images[0],
			},
			exclude:   "-cgroupsv1$",
			family:    "family",
			want:      "ubuntu-v20260801",
			wantCalls: 1,
		},
		{
			name: "other family timestamps are not parsed",
			images: []gceImage{
				image("other", "other-family", "invalid"), images[0],
			},
			family:    "family",
			want:      "ubuntu-v20260801",
			wantCalls: 1,
		},
		{
			name:      "empty list returns an error",
			family:    "family",
			wantErr:   "found zero images",
			wantCalls: 1,
		},
		{
			name:      "all excluded returns an error",
			images:    images,
			exclude:   ".*",
			family:    "family",
			wantErr:   "found zero images",
			wantCalls: 1,
		},
		{
			name:    "invalid include fails before listing",
			include: "[",
			wantErr: "failed to compile image_regex",
		},
		{
			name:    "invalid exclude fails before listing",
			exclude: "[",
			wantErr: "failed to compile image_exclude_regex",
		},
		{
			name:      "list errors retain their cause",
			listErr:   listErr,
			wantErr:   "failed to list images",
			wantCause: listErr,
			wantCalls: 1,
		},
		{
			name:      "invalid JSON returns an error",
			response:  []byte("{"),
			wantErr:   "failed to parse images",
			wantCalls: 1,
		},
		{
			name:      "matching invalid timestamp returns an error",
			images:    []gceImage{image("ubuntu", "family", "invalid")},
			family:    "family",
			wantErr:   "failed to parse instance creation timestamp",
			wantCalls: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response := test.response
			if response == nil {
				var err error
				response, err = json.Marshal(test.images)
				if err != nil {
					t.Fatal(err)
				}
			}
			calls := stubGCEImageLister(t, response, test.listErr)
			got, err := (&GCERunner{}).getGCEImage(test.include, test.exclude, test.family, "fixture-project")
			if *calls != test.wantCalls {
				t.Errorf("list calls = %d, want %d", *calls, test.wantCalls)
			}
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("error = %v, want %q", err, test.wantErr)
				}
				if test.wantCause != nil && !errors.Is(err, test.wantCause) {
					t.Errorf("error = %v, want cause %v", err, test.wantCause)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Errorf("image = %q, want %q", got, test.want)
			}
		})
	}
}

func TestPrepareGCEImages(t *testing.T) {
	listErr := errors.New("fixture list error")
	tests := []struct {
		name      string
		selectors string
		listErr   error
		want      string
		wantErr   string
		wantCause error
		wantCalls int
	}{
		{
			name:      "YAML exclude is applied",
			selectors: "image_family: family\n    image_exclude_regex: '-cgroupsv1$'",
			want:      "ubuntu-v20260801",
			wantCalls: 1,
		},
		{
			name:      "explicit image bypasses invalid filters",
			selectors: "image: pinned-image\n    image_regex: '['\n    image_exclude_regex: '['",
			want:      "pinned-image",
		},
		{
			name:      "explicit image needs no positive selector",
			selectors: "image: pinned-image\n    image_exclude_regex: '['",
			want:      "pinned-image",
		},
		{
			name:      "invalid YAML include fails before listing",
			selectors: "image_regex: '['",
			wantErr:   "failed to compile image_regex",
		},
		{
			name:      "invalid YAML exclude fails before listing",
			selectors: "image_family: family\n    image_exclude_regex: '['",
			wantErr:   "failed to compile image_exclude_regex",
		},
		{
			name:      "exclude without a positive selector is rejected",
			selectors: "image_exclude_regex: '-cgroupsv1$'",
			wantErr:   "image_exclude_regex requires image_regex or image_family",
		},
		{
			name:      "include without family can refine exclude",
			selectors: "image_regex: '^ubuntu-v'\n    image_exclude_regex: '-cgroupsv1$'",
			want:      "ubuntu-v20260801",
			wantCalls: 1,
		},
		{
			name:      "config without exclude retains its selection",
			selectors: "image_family: family\n    image_regex: '^ubuntu-v'",
			want:      "ubuntu-v20260801-cgroupsv1",
			wantCalls: 1,
		},
		{
			name:      "listing error remains wrapped through config loading",
			selectors: "image_family: family",
			listErr:   listErr,
			wantErr:   "could not retrieve an image",
			wantCause: listErr,
			wantCalls: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			originalProject, originalZone, originalMetadata := *project, *zone, *instanceMetadata
			t.Cleanup(func() {
				*project, *zone, *instanceMetadata = originalProject, originalZone, originalMetadata
			})
			*project, *zone, *instanceMetadata = "fixture-project", "fixture-zone", ""
			response := []byte(`[
{"name":"ubuntu-v20260801","family":"family","creationTimestamp":"2026-08-03T10:00:00Z"},
{"name":"ubuntu-v20260801-cgroupsv1","family":"family","creationTimestamp":"2026-08-09T10:00:00Z"}
]`)
			calls := stubGCEImageLister(t, response, test.listErr)
			path := filepath.Join(t.TempDir(), "images.yaml")
			config := "images:\n  ubuntu:\n    project: fixture-project\n    " + test.selectors + "\n"
			if err := os.WriteFile(path, []byte(config), 0600); err != nil {
				t.Fatal(err)
			}
			runner := &GCERunner{cfg: remote.Config{ImageConfigFile: path}}
			got, err := runner.prepareGceImages()
			if *calls != test.wantCalls {
				t.Errorf("list calls = %d, want %d", *calls, test.wantCalls)
			}
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("error = %v, want %q", err, test.wantErr)
				}
				if test.wantCause != nil && !errors.Is(err, test.wantCause) {
					t.Errorf("error = %v, want cause %v", err, test.wantCause)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(got.images) != 1 || got.images["ubuntu"].image != test.want {
				t.Errorf("images = %v, want ubuntu image %q", got.images, test.want)
			}
		})
	}
}

func stubGCEImageLister(t *testing.T, response []byte, listErr error) *int {
	t.Helper()
	original := gceImageLister
	t.Cleanup(func() { gceImageLister = original })
	calls := 0
	gceImageLister = func(args ...string) ([]byte, error) {
		calls++
		want := []string{"compute", "images", "list", "--format=json", "--project=fixture-project"}
		if !slices.Equal(args, want) {
			t.Errorf("list arguments = %q, want %q", args, want)
		}
		return response, listErr
	}
	return &calls
}
