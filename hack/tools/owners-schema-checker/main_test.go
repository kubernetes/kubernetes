package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidateFile(t *testing.T) {
	tests := []struct {
		name    string
		base    string
		content string
		wantErr string
	}{
		{
			name: "valid owners",
			base: "OWNERS",
			content: `approvers:
  - alice
reviewers:
  - bob
options:
  no_parent_owners: true
`,
		},
		{
			name: "invalid owners key",
			base: "OWNERS",
			content: `approvers:
  - alice
emeritus_aprovers:
  - bob
`,
			wantErr: "unknown field",
		},
		{
			name: "duplicate owners key",
			base: "OWNERS",
			content: `approvers:
  - alice
approvers:
  - bob
`,
			wantErr: "key \"approvers\" already set",
		},
		{
			name: "comments only",
			base: "OWNERS",
			content: `# See the OWNERS docs

# still only comments
`,
			wantErr: "only comments",
		},
		{
			name: "valid aliases",
			base: "OWNERS_ALIASES",
			content: `aliases:
  sig-foo:
    - alice
`,
		},
		{
			name: "invalid aliases key",
			base: "OWNERS_ALIASES",
			content: `alias:
  sig-foo:
    - alice
`,
			wantErr: "unknown field",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, tc.base)
			if err := os.WriteFile(path, []byte(tc.content), 0o644); err != nil {
				t.Fatalf("write %s: %v", path, err)
			}

			err := validateFile(path)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("validateFile(%s) unexpected error: %v", path, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("validateFile(%s) error = nil, want %q", path, tc.wantErr)
			}
			if got := err.Error(); !strings.Contains(got, tc.wantErr) {
				t.Fatalf("validateFile(%s) error = %q, want substring %q", path, got, tc.wantErr)
			}
		})
	}
}

func TestIsCommentsOnly(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{name: "empty", content: "", want: true},
		{name: "comments only", content: "# comment\n\n  # comment\n", want: true},
		{name: "yaml content", content: "approvers:\n  - alice\n", want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isCommentsOnly([]byte(tc.content)); got != tc.want {
				t.Fatalf("isCommentsOnly() = %t, want %t", got, tc.want)
			}
		})
	}
}
