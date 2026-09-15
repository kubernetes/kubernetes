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

package validators

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeProfile writes body to a temp file and returns its path.
func writeProfile(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "profile.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing profile: %v", err)
	}
	return path
}

func TestLoadProfileEmptyPath(t *testing.T) {
	profile, err := LoadProfile("")
	if err != nil {
		t.Fatalf("LoadProfile(\"\") = %v, want nil", err)
	}
	if profile != nil {
		t.Errorf("LoadProfile(\"\") = %+v, want nil", profile)
	}
	// The nil profile must be usable without nil-checking.
	if got := profile.formats(); len(got) != 0 {
		t.Errorf("(nil).formats() = %v, want empty", got)
	}
}

func TestLoadProfileErrors(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr string
	}{{
		name:    "not yaml",
		body:    "formats: [",
		wantErr: "parsing profile",
	}, {
		name:    "unknown field",
		body:    "formats:\n  - name: x\n    rexeg: y\n    docs: d\n",
		wantErr: "unknown field",
	}, {
		name:    "missing name",
		body:    "formats:\n  - docs: d\n    regex: x\n    message: m\n",
		wantErr: "formats[0]: name is required",
	}, {
		name:    "missing docs",
		body:    "formats:\n  - name: project-uri\n    regex: x\n    message: m\n",
		wantErr: `formats[0] ("project-uri"): docs is required`,
	}, {
		name:    "no regex",
		body:    "formats:\n  - name: project-uri\n    docs: d\n    message: m\n",
		wantErr: "regex is required",
	}, {
		name:    "regex that does not compile",
		body:    "formats:\n  - name: a\n    docs: d\n    regex: \"[unterminated\"\n    message: m\n",
		wantErr: `regex "[unterminated" does not compile`,
	}, {
		// Reporting the pattern at users is not explaining the rule to them.
		name:    "no message",
		body:    "formats:\n  - name: a\n    docs: d\n    regex: \"^x$\"\n",
		wantErr: "message is required",
	}, {
		name:    "name is not in the conventional shape",
		body:    "formats:\n  - name: Project_URI\n    docs: d\n    regex: x\n    message: m\n",
		wantErr: "name must be lower-case alphanumerics with dashes",
	}, {
		name:    "duplicate format",
		body:    "formats:\n  - name: a\n    docs: d\n    regex: x\n    message: m\n  - name: a\n    docs: d\n    regex: y\n    message: m\n",
		wantErr: `format "a" was already defined by formats[0]`,
	}, {
		// A profile extends the profile; letting it redefine a built-in
		// would make the same tag mean different things per repository.
		name:    "shadows a built-in format",
		body:    "formats:\n  - name: k8s-short-name\n    docs: d\n    regex: x\n    message: m\n",
		wantErr: `"k8s-short-name" is a built-in format and cannot be redefined`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadProfile(writeProfile(t, tc.body))
			if err == nil {
				t.Fatalf("LoadProfile() = nil, want an error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("LoadProfile() = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}

// TestFormatGoIdent pins the mapping from a format name to the variable-name
// fragment it generates. The mapping must be injective: two names that collided
// would share one variable, silently giving one format the other's pattern.
func TestFormatGoIdent(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{"x", "X"},
		{"a1", "A1"},
		{"a-1", "A_1"},
		{"ab", "Ab"},
		{"a-b", "A_b"},
		{"project-uri", "Project_uri"},
		{"project-uri-v2", "Project_uri_v2"},
	}

	seen := map[string]string{}
	for _, tc := range cases {
		if got := formatGoIdent(tc.name); got != tc.want {
			t.Errorf("formatGoIdent(%q) = %q, want %q", tc.name, got, tc.want)
		}
		if prev, dup := seen[tc.want]; dup {
			t.Errorf("formats %q and %q both map to %q", prev, tc.name, tc.want)
		}
		seen[tc.want] = tc.name
	}
}
