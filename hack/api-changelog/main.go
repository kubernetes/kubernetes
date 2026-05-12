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

// Package main provides a command-line tool for managing API changelog entries.
// It can insert new changelog entries with breaking changes and verify that
// existing entries contain expected API changes.
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	changelogFile := flag.String("changelog", "CHANGELOG.md", "path to the CHANGELOG.md file")
	verify := flag.Bool("verify", false, "verify that the first heading contains a code block with the changes")
	insert := flag.Bool("insert", false, "insert a new heading with the changes")
	changes := flag.String("changes", "", "expected changes content for verification or insertion")
	title := flag.String("title", "Replace with a short title", "heading title when inserting changes")
	description := flag.String("description", "Replace this text with a short summary of the change\nand how users of the package can deal with this breaking\nchange. If users are not expected to be affected, then\ninstead explain why. If the changes are too long,\nyou may shorten them by replacing multiple lines\nwith three dots (...).", "first paragraph when inserting changes")
	flag.Parse()

	if err := run(*changelogFile, *verify, *insert, *changes, *title, *description); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		if errors.Is(err, verificationFailErr) {
			os.Exit(2)
		}
		os.Exit(1)
	}
}

var (
	// operationErr is returned when neither or both of -insert and -verify are specified.
	operationErr = errors.New("exactly one of -insert or -verify must be selected")
	// verificationFailErr is returned when verification fails because expected changes are not found.
	verificationFailErr = errors.New("changes not found in code blocks of first heading")
)

// run executes the main logic for either inserting or verifying changelog entries.
// Returns verificationFailErr (exit code 2) if verification fails, or other errors (exit code 1).
func run(changelogFile string, verify bool, insert bool, changes, title, description string) error {
	if !verify && !insert ||
		verify && insert {
		return operationErr
	}
	if changes == "" {
		return errors.New("no changes specified")
	}

	// Below we assume that changes contain complete, i.e. that each
	// line is terminated with a line break. For the input we accept also
	// a missing line break at the end and just add it here.
	if changes[len(changes)-1] != '\n' {
		changes += "\n"
	}

	content, err := os.ReadFile(changelogFile)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	switch {
	case insert:
		newContent, err := insertHeading(content, changes, title, description)
		if err != nil {
			return err
		}
		if err := os.WriteFile(changelogFile, []byte(newContent), 0644); err != nil {
			return fmt.Errorf("failed to write file: %w", err)
		}
	case verify:
		found, codeBlocks := extractInitialCodeBlocks(string(content))
		if !found {
			return fmt.Errorf("no heading found in changelog")
		}

		found = false
		expectedLines := strings.Split(changes, "\n")
		for _, block := range codeBlocks {
			blockLines := strings.Split(block, "\n")
			if matchesWithWildcard(blockLines, expectedLines) {
				found = true
				break
			}
		}
		if !found {
			return verificationFailErr
		}
		fmt.Println("Verification successful")
	default:
		// Not reached in practice, added for the sake of completeness.
		return operationErr
	}

	return nil
}

// extractInitialCodeBlocks parses the markdown content and extracts code blocks
// from the first section (identified by # at the start of a line).
func extractInitialCodeBlocks(content string) (bool, []string) {
	lines := strings.Split(content, "\n")
	var codeBlocks []string
	var haveFirstHeading bool
	var inFirstSection bool
	var firstHeadingLevel int
	var inCodeBlock bool
	var currentCodeBlock strings.Builder

	for _, line := range lines {
		// Check for heading (# at the start of the line).
		if strings.HasPrefix(line, "#") {
			// Count the heading level.
			level := 0
			for i := 0; i < len(line) && line[i] == '#'; i++ {
				level++
			}

			if !haveFirstHeading {
				// Found the first heading.
				haveFirstHeading = true
				firstHeadingLevel = level
				inFirstSection = true
				continue
			}

			// If we encounter another heading at the same or higher level, we're done with the first section.
			if level <= firstHeadingLevel {
				break
			}
		}

		if inFirstSection {
			// Check for code block delimiters.
			if strings.HasPrefix(line, "```") {
				if inCodeBlock {
					// End of code block.
					codeBlocks = append(codeBlocks, currentCodeBlock.String())
					currentCodeBlock.Reset()
					inCodeBlock = false
				} else {
					// Start of code block.
					inCodeBlock = true
				}
			} else if inCodeBlock {
				// Add line to current code block.
				currentCodeBlock.WriteString(line)
				currentCodeBlock.WriteString("\n")
			}
		}
	}

	return haveFirstHeading, codeBlocks
}

// insertHeading creates a new level 3 heading with a template description and code block
// containing the changes. If the file has no headings, appends to the end. Otherwise,
// inserts before the first existing heading.
func insertHeading(content []byte, changes, title, description string) (string, error) {
	lines := string(content)

	newHeading := `### ` + title + `

` + description + `

` + "```" + `
` + changes +
		"```" + `
`

	firstHeadingPos := findFirstHeadingPosition(content)

	if firstHeadingPos == -1 {
		return lines + "\n" + newHeading, nil
	}

	return lines[:firstHeadingPos] + newHeading + "\n" + lines[firstHeadingPos:], nil
}

// findFirstHeadingPosition returns the byte position of the first '#' character,
// which indicates the start of a markdown heading. Returns -1 if no heading is found.
func findFirstHeadingPosition(content []byte) int {
	lines := string(content)
	for i := 0; i < len(lines); i++ {
		if lines[i] == '#' {
			return i
		}
	}
	return -1
}

// matchesWithWildcard checks if text matches pattern, where pattern can contain lines
// with "..." as a wildcard matching any number of lines in text.
func matchesWithWildcard(pattern, text []string) bool {
	pi := 0
	ti := 0

	for pi < len(pattern) && ti < len(text) {
		if strings.TrimSpace(pattern[pi]) == "..." {
			pi++
			if pi >= len(pattern) {
				return true
			}
			for ti < len(text) {
				if matchesWithWildcard(pattern[pi:], text[ti:]) {
					return true
				}
				ti++
			}
			return false
		}

		if pattern[pi] != text[ti] {
			return false
		}
		pi++
		ti++
	}

	for pi < len(pattern) && strings.TrimSpace(pattern[pi]) == "..." {
		pi++
	}

	return pi == len(pattern) && ti == len(text)
}
