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

// write puts body in a temp file named name and returns its path.
func write(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing %s: %v", name, err)
	}
	return path
}

func format(name, pattern string) string {
	return "  - name: " + name + "\n    docs: d\n    pattern: " + pattern + "\n    message: m\n"
}

func TestLoadExtensions(t *testing.T) {
	t.Run("no files", func(t *testing.T) {
		ext, err := LoadExtensions(nil)
		if err != nil || ext != nil {
			t.Fatalf("LoadExtensions(nil) = %+v, %v, want nil, nil", ext, err)
		}
		// format.go calls formats() without nil-checking first.
		if got := ext.formats(); len(got) != 0 {
			t.Errorf("(nil).formats() = %v, want empty", got)
		}
	})

	t.Run("across files", func(t *testing.T) {
		a := write(t, "a.yaml", "formats:\n"+format("a-one", "^a$"))
		b := write(t, "b.yaml", "formats:\n"+format("b-one", "^b$"))
		ext, err := LoadExtensions([]string{a, b})
		if err != nil {
			t.Fatalf("LoadExtensions() = %v", err)
		}
		if got := ext.formats(); len(got) != 2 || got["a-one"].Pattern != "^a$" || got["b-one"].Pattern != "^b$" {
			t.Errorf("formats() = %+v, want a-one and b-one", got)
		}
	})

	t.Run("across documents in one file", func(t *testing.T) {
		path := write(t, "multi.yaml", "formats:\n"+format("doc-one", "^1$")+"---\nformats:\n"+format("doc-two", "^2$"))
		ext, err := LoadExtensions([]string{path})
		if err != nil {
			t.Fatalf("LoadExtensions() = %v", err)
		}
		if got := ext.formats(); len(got) != 2 {
			t.Errorf("formats() = %+v, want both documents", got)
		}
	})

	t.Run("duplicate across files names both", func(t *testing.T) {
		a := write(t, "a.yaml", "formats:\n"+format("dup", "^a$"))
		b := write(t, "b.yaml", "formats:\n"+format("dup", "^b$"))
		_, err := LoadExtensions([]string{a, b})
		if err == nil {
			t.Fatal("LoadExtensions() = nil, want a duplicate error")
		}
		for _, want := range []string{"dup", a, b} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %v does not mention %q", err, want)
			}
		}
	})
}

func TestLoadExtensionsErrors(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		wantErr string
	}{{
		name:    "unknown field",
		body:    "formats:\n  - name: x\n    patern: y\n    docs: d\n",
		wantErr: "unknown field",
	}, {
		name:    "missing name",
		body:    "formats:\n  - docs: d\n    pattern: x\n    message: m\n",
		wantErr: "name is required",
	}, {
		name:    "missing docs",
		body:    "formats:\n  - name: project-uri\n    pattern: x\n    message: m\n",
		wantErr: `formats[0] ("project-uri"): docs is required`,
	}, {
		name:    "no pattern",
		body:    "formats:\n  - name: project-uri\n    docs: d\n    message: m\n",
		wantErr: "pattern is required",
	}, {
		name:    "pattern that does not compile",
		body:    "formats:\n  - name: a\n    docs: d\n    pattern: \"[unterminated\"\n    message: m\n",
		wantErr: `pattern "[unterminated" does not compile`,
	}, {
		name:    "no message",
		body:    "formats:\n  - name: a\n    docs: d\n    pattern: \"^x$\"\n",
		wantErr: "message is required",
	}, {
		name:    "name is not in the conventional shape",
		body:    "formats:\n  - name: Project_URI\n    docs: d\n    pattern: x\n    message: m\n",
		wantErr: "name must be lower-case alphanumerics with dashes",
	}, {
		name:    "duplicate format",
		body:    "formats:\n" + format("a", "x") + format("a", "y"),
		wantErr: `format "a" was already defined in`,
	}, {
		name:    "shadows a built-in format",
		body:    "formats:\n" + format("k8s-short-name", "x"),
		wantErr: `"k8s-short-name" is a built-in format and cannot be redefined`,
	}, {
		name:    "reserved k8s- prefix",
		body:    "formats:\n" + format("k8s-not-yet-a-thing", "x"),
		wantErr: `cannot use the reserved "k8s-" prefix`,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := LoadExtensions([]string{write(t, "cv.yaml", tc.body)})
			if err == nil {
				t.Fatalf("LoadExtensions() = nil, want an error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("LoadExtensions() = %v, want an error containing %q", err, tc.wantErr)
			}
		})
	}
}
