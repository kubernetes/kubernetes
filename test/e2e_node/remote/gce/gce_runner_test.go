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
	"errors"
	"reflect"
	"strings"
	"testing"

	"sigs.k8s.io/yaml"
)

func TestPickNewestImage(t *testing.T) {
	img := func(name, family, ts string) gceImage {
		return gceImage{Name: name, Family: family, CreationTimestamp: ts}
	}
	tests := []struct {
		name              string
		images            []gceImage
		imageRegex        string
		imageExcludeRegex string
		imageFamily       string
		want              string
		wantErr           string
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
			name: "exclude drops the -cgroupsv1 suffix but keeps a newer name ending in 1",
			images: []gceImage{
				img("ubuntu-v20260801", "fam", "2026-08-03T10:00:00Z"),
				img("ubuntu-v20260801-cgroupsv1", "fam", "2026-08-09T10:00:00Z"),
			},
			imageFamily:       "fam",
			imageExcludeRegex: "-cgroupsv1$",
			want:              "ubuntu-v20260801",
		},
		{
			name: "exclude is applied together with the include regex",
			images: []gceImage{
				img("keep-a", "fam", "2026-08-03T10:00:00Z"),
				img("keep-b-cgroupsv1", "fam", "2026-08-09T10:00:00Z"),
				img("skip-c", "fam", "2026-08-05T10:00:00Z"),
			},
			imageFamily:       "fam",
			imageRegex:        "keep-.*",
			imageExcludeRegex: "-cgroupsv1$",
			want:              "keep-a",
		},
		{
			name: "exclude removes every candidate",
			images: []gceImage{
				img("ubuntu-a-cgroupsv1", "fam", "2026-08-03T10:00:00Z"),
				img("ubuntu-b-cgroupsv1", "fam", "2026-08-09T10:00:00Z"),
			},
			imageFamily:       "fam",
			imageExcludeRegex: "-cgroupsv1$",
			wantErr:           `found zero images based on regex "", exclude regex "-cgroupsv1$"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			selector, err := compileImageSelector(tc.imageRegex, tc.imageExcludeRegex, tc.imageFamily)
			if err != nil {
				t.Fatalf("compileImageSelector() unexpected error: %v", err)
			}
			got, err := pickNewestImage(tc.images, selector, "proj")
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

func TestCompileImageSelector(t *testing.T) {
	tests := []struct {
		name              string
		imageRegex        string
		imageExcludeRegex string
		wantInclude       bool
		wantExclude       bool
		wantErr           string
	}{
		{name: "both empty leaves both filters unset"},
		{name: "include only", imageRegex: "keep-.*", wantInclude: true},
		{name: "exclude only", imageExcludeRegex: "-cgroupsv1$", wantExclude: true},
		{name: "a bad include regex returns an error", imageRegex: "[", wantErr: "failed to compile image_regex"},
		{name: "a bad exclude regex returns an error", imageExcludeRegex: "[", wantErr: "failed to compile image_exclude_regex"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, err := compileImageSelector(tc.imageRegex, tc.imageExcludeRegex, "fam")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("compileImageSelector() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("compileImageSelector() unexpected error: %v", err)
			}
			if (s.include != nil) != tc.wantInclude {
				t.Errorf("include non-nil = %v, want %v", s.include != nil, tc.wantInclude)
			}
			if (s.exclude != nil) != tc.wantExclude {
				t.Errorf("exclude non-nil = %v, want %v", s.exclude != nil, tc.wantExclude)
			}
		})
	}
}

func TestGetGCEImage(t *testing.T) {
	const twoImages = `[
		{"name":"ubuntu-v2","family":"fam","creationTimestamp":"2026-08-03T10:00:00Z"},
		{"name":"ubuntu-v2-cgroupsv1","family":"fam","creationTimestamp":"2026-08-09T10:00:00Z"}
	]`
	tests := []struct {
		name              string
		imageRegex        string
		imageExcludeRegex string
		listOut           string
		listErr           error
		wantCalls         int
		want              string
		wantErr           string
	}{
		{
			name:       "a bad include regex fails before any gcloud call",
			imageRegex: "[",
			wantCalls:  0,
			wantErr:    "failed to compile image_regex",
		},
		{
			name:              "a bad exclude regex fails before any gcloud call",
			imageExcludeRegex: "[",
			wantCalls:         0,
			wantErr:           "failed to compile image_exclude_regex",
		},
		{
			name:      "a listing error is wrapped",
			listErr:   errors.New("gcloud boom"),
			wantCalls: 1,
			wantErr:   "failed to list images",
		},
		{
			name:              "the newest non-excluded image is returned",
			imageExcludeRegex: "-cgroupsv1$",
			listOut:           twoImages,
			wantCalls:         1,
			want:              "ubuntu-v2",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			orig := gceImageLister
			gceImageLister = func(args ...string) ([]byte, error) {
				calls++
				return []byte(tc.listOut), tc.listErr
			}
			defer func() { gceImageLister = orig }()

			got, err := (&GCERunner{}).getGCEImage(tc.imageRegex, tc.imageExcludeRegex, "fam", "proj")
			if calls != tc.wantCalls {
				t.Errorf("gcloud invocation count = %d, want %d", calls, tc.wantCalls)
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("getGCEImage() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("getGCEImage() unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("getGCEImage() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestValidateImageSelector(t *testing.T) {
	tests := []struct {
		name    string
		config  GCEImage
		wantErr string
	}{
		{
			name:   "explicit image is allowed",
			config: GCEImage{Image: "ubuntu-x"},
		},
		{
			name:   "image regex is a valid selector",
			config: GCEImage{ImageRegex: "keep-.*"},
		},
		{
			name:   "image family is a valid selector",
			config: GCEImage{ImageFamily: "fam"},
		},
		{
			name:   "exclude with a family is allowed",
			config: GCEImage{ImageFamily: "fam", ImageExcludeRegex: "-cgroupsv1$"},
		},
		{
			name:   "exclude with an include regex is allowed",
			config: GCEImage{ImageRegex: "keep-.*", ImageExcludeRegex: "-cgroupsv1$"},
		},
		{
			name:    "exclude alone is rejected",
			config:  GCEImage{ImageExcludeRegex: "-cgroupsv1$"},
			wantErr: "image_exclude_regex requires image_regex or image_family",
		},
		{
			name: "explicit image takes precedence and ignores dynamic selectors",
			config: GCEImage{
				Image:             "ubuntu-x",
				ImageRegex:        "keep-.*",
				ImageExcludeRegex: "-cgroupsv1$",
				ImageFamily:       "fam",
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateImageSelector("cos-example", tc.config)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("validateImageSelector() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("validateImageSelector() unexpected error: %v", err)
			}
		})
	}
}

func TestGCEImageConfigUnmarshalImageExcludeRegex(t *testing.T) {
	const data = `
images:
  ubuntu-example:
    image_regex: "ubuntu-.*"
    image_exclude_regex: "-cgroupsv1$"
    image_family: "ubuntu-fam"
    project: "ubuntu-proj"
`
	cfg := GCEImageConfig{Images: make(map[string]GCEImage)}
	if err := yaml.Unmarshal([]byte(data), &cfg); err != nil {
		t.Fatalf("yaml.Unmarshal() unexpected error: %v", err)
	}
	got, ok := cfg.Images["ubuntu-example"]
	if !ok {
		t.Fatalf("image %q missing from decoded config", "ubuntu-example")
	}
	if got.ImageExcludeRegex != "-cgroupsv1$" {
		t.Errorf("ImageExcludeRegex = %q, want %q", got.ImageExcludeRegex, "-cgroupsv1$")
	}
}
