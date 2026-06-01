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

// Package main provides a command-line tool for comparing API changes across
// modules. It runs apidiff to detect API changes between two git revisions,
// and inserts or verifies changelog entries for incompatible changes.
//
// It returns:
// - 0: no incompatible changes found
// - 1: unexpected error
// - 2: incompatible changes found and at least one wasn't documented
// - 3: incompatible changes found and all of them were documented
package main

import (
	_ "embed"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

//go:embed api-changes-allowlist
var apiChangesAllowlist string

// changelogFilename is the default name for changelog files.
const changelogFilename = "CHANGELOG.md"

// errVerificationFail is returned when verification fails because expected changes are not found.
var errVerificationFail = errors.New("changes not found in code blocks of first heading")

// exitError is returned to main() to influce the exist code.
type exitError struct {
	exitCode int
}

func (e exitError) Error() string {
	return ""
}

// runDiffOptions configures the API diff operation.
type runDiffOptions struct {
	base            string // base git revision (required)
	target          string // target git revision; empty means the working tree
	changelogName   string // changelog filename, e.g. "CHANGELOG.md"
	updateChangelog bool   // update changelog files in place instead of printing a patch
	mergeCommit     string // when non-empty, uses this merge commit to populate new changelog sections
}

func main() {
	base := flag.String("base", "", "base git revision (required)")
	target := flag.String("target", "", "target git revision; default is the working tree")
	updateChangelog := flag.Bool("update-changelog", false, "update CHANGELOG.md files in place when incompatible changes are found")
	mergeCommit := flag.String("merge-commit", "", "when set, populates new changelog sections with PR information from this merge commit")
	flag.Parse()

	if *base == "" {
		fmt.Fprintln(os.Stderr, "Error: -base is required")
		os.Exit(1)
	}
	opts := runDiffOptions{
		base:            *base,
		target:          *target,
		changelogName:   changelogFilename,
		updateChangelog: *updateChangelog,
		mergeCommit:     *mergeCommit,
	}
	if err := runDiff(opts, flag.Args()); err != nil {
		if errExit, ok := errors.AsType[exitError](err); ok {
			// Don't include "Error", that might be too strong.
			fmt.Fprintf(os.Stderr, "%v\n", err)
			os.Exit(errExit.exitCode)
		}
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// runDiff runs the full API diff workflow: dumps API state for the base and target
// revisions, compares each module, reports incompatibilities, and optionally updates
// or patches CHANGELOG.md files.
// Returns a non-nil error if any module has unresolved incompatible changes.
func runDiff(opts runDiffOptions, dirs []string) error {
	if opts.changelogName == "" {
		opts.changelogName = changelogFilename
	}

	tempDir, err := os.MkdirTemp("", "apidiff-")
	if err != nil {
		return fmt.Errorf("creating temp directory: %w", err)
	}
	defer func() { _ = os.RemoveAll(tempDir) }()

	repoRoot, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("getting working directory: %w", err)
	}

	afterDir := filepath.Join(tempDir, "after")
	beforeDir := filepath.Join(tempDir, "before")

	// Dump API state for the target revision (or the working tree).
	if opts.target == "" {
		if err := runApidiff(repoRoot, dirs, afterDir); err != nil {
			return fmt.Errorf("running apidiff on working tree: %w", err)
		}
	} else {
		targetWorktree := filepath.Join(tempDir, "target")
		cleanup, err := setupWorktree(targetWorktree, opts.target)
		if err != nil {
			return err
		}
		defer cleanup()
		if err := runApidiff(targetWorktree, dirs, afterDir); err != nil {
			return fmt.Errorf("running apidiff on target %s: %w", opts.target, err)
		}
	}

	// Dump API state for the base revision.
	baseWorktree := filepath.Join(tempDir, "base")
	cleanup, err := setupWorktree(baseWorktree, opts.base)
	if err != nil {
		return err
	}
	defer cleanup()
	if err := runApidiff(baseWorktree, dirs, beforeDir); err != nil {
		return fmt.Errorf("running apidiff on base %s: %w", opts.base, err)
	}

	// Retrieve merge commit info when populating changelogs from a PR merge.
	var mcTitle, mcDescription string
	if opts.mergeCommit != "" {
		mcTitle, mcDescription, err = getMergeCommitInfo(opts.mergeCommit)
		if err != nil {
			return err
		}
	}

	const defaultTitle = "Replace with a short title"
	const defaultDescription = `Replace this text with a short summary of the change
and how users of the package can deal with this breaking
change. If users are not expected to be affected, then
instead explain why. If the changes are too long,
you may shorten them by replacing multiple lines
with three dots (...).`

	type changelogEntry struct {
		original string
		updated  string
	}
	var failures []string
	var changelogEntries []changelogEntry
	documentedFailures := 0
	updatedFailures := 0

	fmt.Println()
	for _, dir := range dirs {
		beforeFile := filepath.Join(beforeDir, outputName(dir))
		afterFile := filepath.Join(afterDir, outputName(dir))

		incompatible, err := compareApidiff(dir, beforeFile, afterFile)
		if err != nil {
			return fmt.Errorf("comparing %s: %w", dir, err)
		}
		if incompatible == "" {
			continue
		}

		// Check for a CHANGELOG.md in this directory.
		changelog := filepath.Join(dir, opts.changelogName)
		if _, err := os.Stat(changelog); os.IsNotExist(err) {
			// No changelog file; just mark as a failure.
			failures = append(failures, dir)
			continue
		}

		// Verify whether the incompatible change is already documented.
		verifyErr := verifyChangelog(changelog, incompatible)
		if verifyErr == nil {
			// Already documented; not a failure.
			documentedFailures++
			continue
		}
		if !errors.Is(verifyErr, errVerificationFail) {
			return fmt.Errorf("verifying changelog for %s: %w", dir, verifyErr)
		}

		// Determine the title and description for the new changelog section.
		title, description := defaultTitle, defaultDescription
		if opts.mergeCommit != "" {
			title, description = mcTitle, mcDescription
		}

		if opts.updateChangelog {
			// Insert the entry directly into the original changelog.
			if err := insertChangelog(changelog, incompatible, title, description); err != nil {
				return fmt.Errorf("inserting changelog entry for %s: %w", dir, err)
			}
			updatedFailures++
		} else {
			// Write the entry into a temp copy for patch generation later.
			tempChangelog := filepath.Join(tempDir, changelog)
			if err := os.MkdirAll(filepath.Dir(tempChangelog), 0755); err != nil {
				return fmt.Errorf("creating temp changelog directory: %w", err)
			}
			content, err := os.ReadFile(changelog)
			if err != nil {
				return fmt.Errorf("reading changelog %s: %w", changelog, err)
			}
			if err := os.WriteFile(tempChangelog, content, 0644); err != nil {
				return fmt.Errorf("writing temp changelog: %w", err)
			}
			if err := insertChangelog(tempChangelog, incompatible, title, description); err != nil {
				return fmt.Errorf("inserting changelog entry for %s: %w", dir, err)
			}
			changelogEntries = append(changelogEntries, changelogEntry{original: changelog, updated: tempChangelog})
		}
		failures = append(failures, dir)
	}

	if len(failures) == 0 {
		return nil
	}

	fmt.Print(`
Detected incompatible changes on modules:
`)
	for _, f := range failures {
		fmt.Println(f)
	}
	fmt.Print(`
For more information about incompatible changes, see
https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/go_api_changes.md

`)

	if len(changelogEntries) > 0 {
		dirArgs := strings.Join(dirs, " ")
		targetFlag := ""
		if opts.target != "" {
			targetFlag = fmt.Sprintf("-t %s ", opts.target)
		}
		fmt.Printf(`
Run the following command to add the incompatible changes to
the %s file(s), edit the modified file(s) to
replace the template text in the new section at the top
with and explanation of the changes, then include the result
in the pull request for review:

    hack/update-go-apidocs.sh

Under the hood this will run:

    hack/apidiff.sh -u -r %s %s%s

If running these commands is impossible, then you can
apply the following diff instead by piping it into "patch -p0":

vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
`, opts.changelogName, opts.base, targetFlag, dirArgs)
		for _, entry := range changelogEntries {
			// diff exits 1 when differences are found; ignore the exit code.
			diffCmd := exec.Command("diff", "-c", entry.original, entry.updated)
			out, _ := diffCmd.CombinedOutput()
			fmt.Print(string(out))
		}
		fmt.Print(`^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`)
	}

	// 3 indicates to apidiff.sh that it should consider the check a success, despite having
	// some incompatible changes.
	if len(failures) == updatedFailures {
		return fmt.Errorf("Documented changes in %d module(s)%w", len(failures), exitError{3})
	}
	if len(failures) == documentedFailures {
		return fmt.Errorf("Found documentation of changes in %d module(s)%w", len(failures), exitError{3})
	}
	if len(failures) == updatedFailures+documentedFailures {
		return fmt.Errorf("Found documentation of changes or updated documentation in %d module(s)%w", len(failures), exitError{3})
	}

	// 2 indicates that there's missing documentation.
	return fmt.Errorf("Error: incompatible changes in %d module(s)%w", len(failures), exitError{2})
}

// setupWorktree creates a git worktree at worktreePath for the specified revision.
// Returns a cleanup function that removes the worktree.
func setupWorktree(worktreePath, rev string) (func(), error) {
	if _, err := os.Stat(worktreePath); os.IsNotExist(err) {
		cmd := exec.Command("git", "worktree", "add", "-f", "-d", worktreePath, rev)
		if out, err := cmd.CombinedOutput(); err != nil {
			return nil, fmt.Errorf("git worktree add %s: %w\n%s", rev, err, out)
		}
	}
	return func() {
		_ = exec.Command("git", "worktree", "remove", "-f", worktreePath).Run()
	}, nil
}

// runApidiff runs apidiff in parallel for each target directory, writing the
// module API state to files in outDir.
func runApidiff(repoRoot string, dirs []string, outDir string) error {
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("creating output directory: %w", err)
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errs []error
	for _, dir := range dirs {
		absDir := filepath.Join(repoRoot, dir)
		if _, err := os.Stat(absDir); os.IsNotExist(err) {
			fmt.Printf("module %s does not exist, skipping ...\n", dir)
			continue
		}
		wg.Add(1)
		go func(d, absD string) {
			defer wg.Done()
			outFile := filepath.Join(outDir, outputName(d))
			cmd := exec.Command("apidiff", "-m", "-w", outFile, ".")
			cmd.Dir = absD
			if out, err := cmd.CombinedOutput(); err != nil {
				mu.Lock()
				errs = append(errs, fmt.Errorf("apidiff in %s: %w\n%s", d, err, out))
				mu.Unlock()
			}
		}(dir, absDir)
	}
	wg.Wait()
	return errors.Join(errs...)
}

// outputName converts a directory path to a safe filename for apidiff output.
func outputName(path string) string {
	re := regexp.MustCompile(`[^a-zA-Z0-9_-]`)
	return re.ReplaceAllString(path, "_") + ".out"
}

// getMergeCommitInfo extracts the PR title and description from a GitHub merge commit.
// The merge commit body is expected in the format produced by GitHub:
// first line "Merge pull request #<number> from <branch>", last line is the PR title.
func getMergeCommitInfo(commit string) (title, description string, err error) {
	cmd := exec.Command("git", "show", "--no-patch", "--format=%B", commit)
	out, err := cmd.Output()
	if err != nil {
		return "", "", fmt.Errorf("git show %s: %w", commit, err)
	}
	lines := trimTrailingEmpty(strings.Split(strings.TrimRight(string(out), "\n"), "\n"))
	// First line: "Merge pull request #<number> from <branch>".
	words := strings.Fields(lines[0])
	if len(words) < 4 {
		return "", "", fmt.Errorf("unexpected merge commit format: %q", lines[0])
	}
	prNum := strings.TrimPrefix(words[3], "#")
	title = lines[len(lines)-1]
	description = fmt.Sprintf("See [PR #%s](https://github.com/kubernetes/kubernetes/pull/%s).", prNum, prNum)
	return title, description, nil
}

// trimTrailingEmpty removes trailing empty strings from a slice.
func trimTrailingEmpty(lines []string) []string {
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// compareApidiff runs apidiff on two module state files, prints the formatted
// comparison report to stdout, and returns the incompatible (non-tolerated)
// changes for changelog verification. Returns an empty string if there are none.
func compareApidiff(what, beforeFile, afterFile string) (string, error) {
	if _, err := os.Stat(beforeFile); os.IsNotExist(err) {
		fmt.Printf("## %s\ncan not compare changes, module didn't exist before or after\n\n", what)
		return "", nil
	}
	if _, err := os.Stat(afterFile); os.IsNotExist(err) {
		fmt.Printf("## %s\ncan not compare changes, module didn't exist before or after\n\n", what)
		return "", nil
	}

	// apidiff exits non-zero when it finds incompatible changes, so ignore the exit code.
	cmd := exec.Command("apidiff", "-m", beforeFile, afterFile)
	out, _ := cmd.CombinedOutput()

	// Filter out "Ignoring internal package" lines.
	var filtered []string
	for line := range strings.SplitSeq(string(out), "\n") {
		if !strings.HasPrefix(line, "Ignoring internal package") {
			filtered = append(filtered, line)
		}
	}
	filtered = trimTrailingEmpty(filtered)

	fmt.Printf("## %s\n", what)
	if len(filtered) == 0 {
		fmt.Println("no changes")
		return "", nil
	}

	preamble, incompatibleLines, compatibleLines := splitApidiffSections(filtered)

	// Filter incompatible changes through the allowlist.
	incompatibleStr := strings.Join(incompatibleLines, "\n")
	tolerated, err := filterChanges(incompatibleStr, false)
	if err != nil {
		return "", fmt.Errorf("filtering tolerated changes: %w", err)
	}
	incompatibleStr, err = filterChanges(incompatibleStr, true)
	if err != nil {
		return "", fmt.Errorf("filtering incompatible changes: %w", err)
	}

	// Print preamble (matches shell's `echo "$changes"`, even when empty, producing a blank line).
	fmt.Println(preamble)
	if incompatibleStr != "" {
		fmt.Println("Incompatible changes:")
		fmt.Print(incompatibleStr)
	}
	if tolerated != "" {
		fmt.Println("Acceptable incompatible changes:")
		fmt.Print(tolerated)
	}
	if len(compatibleLines) > 0 {
		fmt.Println("Compatible changes:")
		fmt.Println(strings.Join(compatibleLines, "\n"))
	}
	fmt.Println()

	return incompatibleStr, nil
}

// splitApidiffSections parses lines of apidiff output into preamble, incompatible,
// and compatible sections. Lines within each section are sorted.
func splitApidiffSections(lines []string) (preamble string, incompatible, compatible []string) {
	// Find and extract the "Compatible changes:" section first.
	for i, line := range lines {
		if line == "Compatible changes:" {
			compatible = append([]string{}, lines[i+1:]...)
			sort.Strings(compatible)
			lines = trimTrailingEmpty(lines[:i])
			break
		}
	}
	// Find and extract the "Incompatible changes:" section.
	for i, line := range lines {
		if line == "Incompatible changes:" {
			incompatible = append([]string{}, lines[i+1:]...)
			sort.Strings(incompatible)
			lines = trimTrailingEmpty(lines[:i])
			break
		}
	}
	preamble = strings.Join(lines, "\n")
	return
}

// filterChanges filters API change lines against the allowlist.
// If exclude is true, returns lines not matching any allowlist pattern (real incompatible changes).
// If exclude is false, returns lines matching at least one allowlist pattern (tolerated changes).
func filterChanges(changes string, exclude bool) (string, error) {
	patterns, err := compileAllowlistPatterns()
	if err != nil {
		return "", err
	}
	var result strings.Builder
	for line := range strings.SplitSeq(changes, "\n") {
		if line == "" {
			continue
		}
		matched := false
		for _, p := range patterns {
			if p.MatchString(line) {
				matched = true
				break
			}
		}
		if matched != exclude {
			result.WriteString(line)
			result.WriteByte('\n')
		}
	}
	return result.String(), nil
}

// compileAllowlistPatterns compiles the regular expressions from the embedded allowlist.
func compileAllowlistPatterns() ([]*regexp.Regexp, error) {
	var patterns []*regexp.Regexp
	for line := range strings.SplitSeq(apiChangesAllowlist, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		re, err := regexp.Compile(line)
		if err != nil {
			return nil, fmt.Errorf("failed to compile regex %q: %w", line, err)
		}
		patterns = append(patterns, re)
	}
	return patterns, nil
}

// verifyChangelog checks that the first heading of file contains a code block with changes.
// Returns errVerificationFail if changes are not found.
func verifyChangelog(file, changes string) error {
	if changes == "" {
		return errors.New("no changes specified")
	}
	if changes[len(changes)-1] != '\n' {
		changes += "\n"
	}
	content, err := os.ReadFile(file)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}
	found, codeBlocks := extractInitialCodeBlocks(string(content))
	if !found {
		return fmt.Errorf("no heading found in changelog")
	}
	expectedLines := strings.Split(changes, "\n")
	for _, block := range codeBlocks {
		if matchesWithWildcard(strings.Split(block, "\n"), expectedLines) {
			return nil
		}
	}
	return errVerificationFail
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
		if strings.HasPrefix(line, "#") {
			level := 0
			for i := 0; i < len(line) && line[i] == '#'; i++ {
				level++
			}
			if !haveFirstHeading {
				haveFirstHeading = true
				firstHeadingLevel = level
				inFirstSection = true
				continue
			}
			if level <= firstHeadingLevel {
				break
			}
		}
		if inFirstSection {
			if strings.HasPrefix(line, "```") {
				if inCodeBlock {
					codeBlocks = append(codeBlocks, currentCodeBlock.String())
					currentCodeBlock.Reset()
					inCodeBlock = false
				} else {
					inCodeBlock = true
				}
			} else if inCodeBlock {
				currentCodeBlock.WriteString(line)
				currentCodeBlock.WriteString("\n")
			}
		}
	}
	return haveFirstHeading, codeBlocks
}

// matchesWithWildcard checks if text matches pattern, where pattern can contain lines
// with "..." as a wildcard matching any number of lines in text.
func matchesWithWildcard(pattern, text []string) bool {
	pi, ti := 0, 0
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

// insertChangelog inserts a new heading with a code block containing changes into file.
func insertChangelog(file, changes, title, description string) error {
	if changes == "" {
		return errors.New("no changes specified")
	}
	if changes[len(changes)-1] != '\n' {
		changes += "\n"
	}
	content, err := os.ReadFile(file)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}
	newContent, err := insertHeading(content, changes, title, description)
	if err != nil {
		return err
	}
	if err := os.WriteFile(file, []byte(newContent), 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	return nil
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
	for i := 0; i < len(content); i++ {
		if content[i] == '#' {
			return i
		}
	}
	return -1
}
