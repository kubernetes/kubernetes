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

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMatchesWithWildcard(t *testing.T) {
	tests := []struct {
		name     string
		pattern  []string
		text     []string
		expected bool
	}{
		{
			name:     "exact-match",
			pattern:  []string{"line1", "line2", "line3"},
			text:     []string{"line1", "line2", "line3"},
			expected: true,
		},
		{
			name:     "no-match",
			pattern:  []string{"line1", "line2"},
			text:     []string{"line1", "different"},
			expected: false,
		},
		{
			name:     "wildcard-at-end",
			pattern:  []string{"line1", "..."},
			text:     []string{"line1", "line2", "line3"},
			expected: true,
		},
		{
			name:     "wildcard-in-middle",
			pattern:  []string{"line1", "...", "line5"},
			text:     []string{"line1", "line2", "line3", "line4", "line5"},
			expected: true,
		},
		{
			name:     "wildcard-at-start",
			pattern:  []string{"...", "line3"},
			text:     []string{"line1", "line2", "line3"},
			expected: true,
		},
		{
			name:     "multiple-wildcards",
			pattern:  []string{"line1", "...", "line3", "...", "line5"},
			text:     []string{"line1", "line2", "line3", "line4", "line5"},
			expected: true,
		},
		{
			name:     "wildcard-no-match-after",
			pattern:  []string{"line1", "...", "missing"},
			text:     []string{"line1", "line2", "line3"},
			expected: false,
		},
		{
			name:     "empty-pattern-and-text",
			pattern:  []string{},
			text:     []string{},
			expected: true,
		},
		{
			name:     "only-wildcard",
			pattern:  []string{"..."},
			text:     []string{"line1", "line2"},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := matchesWithWildcard(tt.pattern, tt.text)
			if result != tt.expected {
				t.Errorf("matchesWithWildcard() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestFindFirstHeadingPosition(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected int
	}{
		{
			name:     "heading-at-start",
			content:  "# Heading\nContent",
			expected: 0,
		},
		{
			name:     "heading-after-text",
			content:  "Some text\n# Heading",
			expected: 10,
		},
		{
			name:     "no-heading",
			content:  "Just some text\nNo heading here",
			expected: -1,
		},
		{
			name:     "empty-file",
			content:  "",
			expected: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := findFirstHeadingPosition([]byte(tt.content))
			if result != tt.expected {
				t.Errorf("findFirstHeadingPosition() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestExtractFirstHeadingAndCodeBlocks(t *testing.T) {
	tests := []struct {
		name             string
		markdown         string
		expectFound      bool
		expectCodeBlocks []string
	}{
		{
			name: "single-code-block",
			markdown: `# First Heading

` + "```" + `
code block 1
` + "```" + `

## Second Heading`,
			expectFound:      true,
			expectCodeBlocks: []string{"code block 1\n"},
		},
		{
			name: "multiple-code-blocks",
			markdown: `# First Heading

` + "```" + `
code block 1
` + "```" + `

Some text

` + "```" + `
code block 2
` + "```" + `

## Second Heading`,
			expectFound:      true,
			expectCodeBlocks: []string{"code block 1\n", "code block 2\n"},
		},
		{
			name: "no-code-blocks",
			markdown: `# First Heading

Just text content

## Second Heading`,
			expectFound:      true,
			expectCodeBlocks: []string{},
		},
		{
			name:             "no-heading",
			markdown:         "Just some text",
			expectFound:      false,
			expectCodeBlocks: []string{},
		},
		{
			name: "code-block-in-second-heading-not-captured",
			markdown: `# First Heading

Text

# Second Heading

` + "```" + `
code block
` + "```" + ``,
			expectFound:      true,
			expectCodeBlocks: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			found, codeBlocks := extractInitialCodeBlocks(tt.markdown)

			if found != tt.expectFound {
				t.Errorf("extractFirstHeadingAndCodeBlocks() found = %v, expected %v", found, tt.expectFound)
			}

			if len(codeBlocks) != len(tt.expectCodeBlocks) {
				t.Errorf("extractFirstHeadingAndCodeBlocks() returned %d code blocks, expected %d",
					len(codeBlocks), len(tt.expectCodeBlocks))
				return
			}

			for i, block := range codeBlocks {
				if block != tt.expectCodeBlocks[i] {
					t.Errorf("code block %d = %q, expected %q", i, block, tt.expectCodeBlocks[i])
				}
			}
		})
	}
}

func TestInsertHeading(t *testing.T) {
	defaultTitle := "Replace with a short title"
	defaultDescription := "Replace this text with a short summary of the change\nand how users of the package can deal with this breaking\nchange. If users are not expected to be affected, then\ninstead explain why. If the changes are too long,\nyou may shorten them by replacing multiple lines\nwith three dots (...)."

	tests := []struct {
		name           string
		content        string
		changes        string
		title          string
		description    string
		expectContains []string
	}{
		{
			name:        "insert-before-existing-heading",
			content:     "# Existing Heading\n\nContent",
			changes:     "test changes",
			title:       defaultTitle,
			description: defaultDescription,
			expectContains: []string{
				"### Replace with a short title",
				"test changes",
				"# Existing Heading",
			},
		},
		{
			name:        "append-to-empty-file",
			content:     "",
			changes:     "new changes",
			title:       defaultTitle,
			description: defaultDescription,
			expectContains: []string{
				"### Replace with a short title",
				"new changes",
			},
		},
		{
			name:        "append-to-file-without-headings",
			content:     "Some text without headings",
			changes:     "api changes",
			title:       defaultTitle,
			description: defaultDescription,
			expectContains: []string{
				"Some text without headings",
				"### Replace with a short title",
				"api changes",
			},
		},
		{
			name:        "custom-title-and-description",
			content:     "# Existing\n",
			changes:     "changes",
			title:       "Custom Title",
			description: "Custom description text",
			expectContains: []string{
				"### Custom Title",
				"Custom description text",
				"changes",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := insertHeading([]byte(tt.content), tt.changes, tt.title, tt.description)
			if err != nil {
				t.Fatalf("insertHeading() error = %v", err)
			}

			for _, expected := range tt.expectContains {
				if !strings.Contains(result, expected) {
					t.Errorf("insertHeading() result does not contain %q", expected)
				}
			}

			if tt.content != "" && strings.HasPrefix(tt.content, "#") {
				if strings.Index(result, "###") >= strings.Index(result, tt.content) {
					t.Error("insertHeading() did not insert before existing heading")
				}
			}
		})
	}
}

func TestRun(t *testing.T) {
	tempDir := t.TempDir()

	t.Run("insert-mode", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "insert-test.md")
		if err := os.WriteFile(changelogPath, []byte("# Existing\n"), 0644); err != nil {
			t.Fatal(err)
		}

		err := run(changelogPath, false, true, "test changes", "Replace with a short title", "Replace this text with a short summary of the change\nand how users of the package can deal with this breaking\nchange. If users are not expected to be affected, then\ninstead explain why. If the changes are too long,\nyou may shorten them by replacing multiple lines\nwith three dots (...).")
		if err != nil {
			t.Errorf("run() insert mode error = %v", err)
		}

		content, err := os.ReadFile(changelogPath)
		if err != nil {
			t.Fatal(err)
		}

		if !strings.Contains(string(content), "```\ntest changes\n```") {
			t.Errorf("run() insert mode did not insert changes, got instead:\n%s", string(content))
		}
	})

	t.Run("verify-mode-success", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "verify-success.md")
		markdown := `### Breaking Change

Description

` + "```" + `
line1
line2
` + "```" + `
`
		if err := os.WriteFile(changelogPath, []byte(markdown), 0644); err != nil {
			t.Fatal(err)
		}

		err := run(changelogPath, true, false, "line1\nline2\n", "", "")
		if err != nil {
			t.Errorf("run() verify mode error = %v, expected success", err)
		}
	})

	t.Run("verify-mode-with-wildcard", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "verify-wildcard.md")
		markdown := `### Breaking Change

Description

` + "```" + `
line1
...
line4
` + "```" + `
`
		if err := os.WriteFile(changelogPath, []byte(markdown), 0644); err != nil {
			t.Fatal(err)
		}

		err := run(changelogPath, true, false, "line1\nline2\nline3\nline4\n", "", "")
		if err != nil {
			t.Errorf("run() verify mode with wildcard error = %v, expected success", err)
		}
	})

	t.Run("verify-mode-failure", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "verify-fail.md")
		markdown := `### Breaking Change

Description

` + "```" + `
different content
` + "```" + `
`
		if err := os.WriteFile(changelogPath, []byte(markdown), 0644); err != nil {
			t.Fatal(err)
		}

		err := run(changelogPath, true, false, "expected content", "", "")
		if err != verificationFailErr {
			t.Errorf("run() verify mode error = %v, expected verificationFailErr", err)
		}
	})

	t.Run("missing-changes-flag", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "test.md")
		err := run(changelogPath, true, false, "", "", "")
		if err == nil {
			t.Error("run() with empty changes should return error")
		}
	})

	t.Run("both-flags-set", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "test.md")
		err := run(changelogPath, true, true, "changes", "", "")
		if err != operationErr {
			t.Errorf("run() with both flags error = %v, expected operationErr", err)
		}
	})

	t.Run("no-flags-set", func(t *testing.T) {
		changelogPath := filepath.Join(tempDir, "test.md")
		err := run(changelogPath, false, false, "changes", "", "")
		if err != operationErr {
			t.Errorf("run() with no flags error = %v, expected operationErr", err)
		}
	})
}
