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
	testTitle := "Replace with a short title"
	testDescription := "Replace this text with a short summary of the change"

	tests := []struct {
		name           string
		content        string
		changes        string
		title          string
		description    string
		expectContains []string // Each entry is a line without newline, order as in the expected result.
	}{
		{
			name:        "insert-before-existing-heading",
			content:     "# Existing Heading\n\nContent",
			changes:     "test changes",
			title:       testTitle,
			description: testDescription,
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
			title:       testTitle,
			description: testDescription,
			expectContains: []string{
				"### Replace with a short title",
				"new changes",
			},
		},
		{
			name:        "append-to-file-without-headings",
			content:     "Some text without headings",
			changes:     "api changes",
			title:       testTitle,
			description: testDescription,
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
		{
			name:        "not-a-heading",
			content:     "some text with hash # in the middle\n",
			changes:     "changes",
			title:       "Custom Title",
			description: "Custom description text",
			expectContains: []string{
				"some text with hash # in the middle",
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

			previousIndex := -1
			for _, expected := range tt.expectContains {
				index := strings.Index(result, expected+"\n")
				if index == -1 {
					t.Fatalf("insertHeading() result does not contain line %q, got:\n%s", expected, result)
				}
				if index < previousIndex {
					t.Fatalf("insertHeading() result has line %q before previous expected line:\n%s", expected, result)
				}
				previousIndex = index
			}

			if tt.content != "" && strings.HasPrefix(tt.content, "#") {
				if strings.Index(result, "###") >= strings.Index(result, tt.content) {
					t.Error("insertHeading() did not insert before existing heading")
				}
			}
		})
	}
}

func TestFilterChanges(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		exclude  bool
		expected string
	}{
		{
			name: "exclude-matching-lines",
			input: `- ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
- ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
- some other change that should not be filtered
- ./informers/autoscaling.Interface.V2beta1: removed
- package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed
- another unfiltered change`,
			exclude: true,
			expected: `- some other change that should not be filtered
- another unfiltered change
`,
		},
		{
			name: "include-only-matching-lines",
			input: `- ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
- ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
- some other change that should not be filtered
- ./informers/autoscaling.Interface.V2beta1: removed
- package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed
- another unfiltered change`,
			exclude: false,
			expected: `- ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
- ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
- ./informers/autoscaling.Interface.V2beta1: removed
- package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed
`,
		},
		{
			name:     "exclude-empty-input",
			input:    "",
			exclude:  true,
			expected: "",
		},
		{
			name:     "include-empty-input",
			input:    "",
			exclude:  false,
			expected: "",
		},
		{
			name: "exclude-no-matches",
			input: `- some other change
- another change
- yet another change`,
			exclude: true,
			expected: `- some other change
- another change
- yet another change
`,
		},
		{
			name: "include-no-matches",
			input: `- some other change
- another change
- yet another change`,
			exclude:  false,
			expected: "",
		},
		{
			name: "exclude-all-matches",
			input: `- ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
- ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
- package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed`,
			exclude:  true,
			expected: "",
		},
		{
			name: "include-all-matches",
			input: `- ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
- ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
- package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed`,
			exclude: false,
			expected: `- ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
- ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
- package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := filterChanges(tt.input, tt.exclude)
			if err != nil {
				t.Fatalf("filterChanges() error = %v", err)
			}

			if got != tt.expected {
				t.Errorf("filterChanges() output mismatch\nGot:\n%s\nExpected:\n%s", got, tt.expected)
			}
		})
	}
}

func TestOutputName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{input: "./", expected: "__.out"},
		{input: "./staging/src/k8s.io/client-go", expected: "__staging_src_k8s_io_client-go.out"},
		{input: ".", expected: "_.out"},
		{input: "simple", expected: "simple.out"},
		{input: "with spaces", expected: "with_spaces.out"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := outputName(tt.input)
			if got != tt.expected {
				t.Errorf("outputName(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestSplitApidiffSections(t *testing.T) {
	tests := []struct {
		name             string
		lines            []string
		expectedPreamble string
		expectedIncompat []string
		expectedCompat   []string
	}{
		{
			name: "both-sections",
			lines: []string{
				"Incompatible changes:",
				"- pkg1: removed",
				"- pkg2: changed",
				"Compatible changes:",
				"- pkg3: added",
			},
			expectedPreamble: "",
			expectedIncompat: []string{"- pkg1: removed", "- pkg2: changed"},
			expectedCompat:   []string{"- pkg3: added"},
		},
		{
			name: "incompatible-only",
			lines: []string{
				"Incompatible changes:",
				"- pkg1: removed",
			},
			expectedPreamble: "",
			expectedIncompat: []string{"- pkg1: removed"},
			expectedCompat:   nil,
		},
		{
			name: "compatible-only",
			lines: []string{
				"Compatible changes:",
				"- pkg1: added",
			},
			expectedPreamble: "",
			expectedIncompat: nil,
			expectedCompat:   []string{"- pkg1: added"},
		},
		{
			name: "sorting",
			lines: []string{
				"Incompatible changes:",
				"- zzz: removed",
				"- aaa: changed",
			},
			expectedPreamble: "",
			expectedIncompat: []string{"- aaa: changed", "- zzz: removed"},
			expectedCompat:   nil,
		},
		{
			name:             "empty",
			lines:            []string{},
			expectedPreamble: "",
			expectedIncompat: nil,
			expectedCompat:   nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			preamble, incompat, compat := splitApidiffSections(tt.lines)
			if preamble != tt.expectedPreamble {
				t.Errorf("preamble = %q, want %q", preamble, tt.expectedPreamble)
			}
			if len(incompat) != len(tt.expectedIncompat) {
				t.Errorf("incompatible count = %d, want %d: %v", len(incompat), len(tt.expectedIncompat), incompat)
			} else {
				for i := range incompat {
					if incompat[i] != tt.expectedIncompat[i] {
						t.Errorf("incompatible[%d] = %q, want %q", i, incompat[i], tt.expectedIncompat[i])
					}
				}
			}
			if len(compat) != len(tt.expectedCompat) {
				t.Errorf("compatible count = %d, want %d: %v", len(compat), len(tt.expectedCompat), compat)
			} else {
				for i := range compat {
					if compat[i] != tt.expectedCompat[i] {
						t.Errorf("compatible[%d] = %q, want %q", i, compat[i], tt.expectedCompat[i])
					}
				}
			}
		})
	}
}
