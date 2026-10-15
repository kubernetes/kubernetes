/*
Copyright 2018 The Kubernetes Authors.

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

// Package golangcilint verifies hack/verify-golangci-lint.sh, golangci-lint
// and its configuration files by running it against the test cases in
// hack/testdata/golangci-lint, once with hack/golangci.yaml and once with
// hack/golangci-hints.yaml. The test case source documents what it expects to
// be reported through "// <linter> issue: <regexp>" and
// "// <linter> hint: <regexp>" comments. An "issue" is reported by both configs,
// a "hint" only by the latter. They get matched against the linter name and
// text of the actual findings.
//
// The test is run as an integration test instead of a simple unit test because
// building and running golangci-lint might need more time than normal for a
// unit test.
//
// Test cases live under hack/testdata/golangci-lint so that they are outside
// of any "test"- or "testing"-named directory. Some linter rules (for example
// depguard's test-only-dependencies) are keyed off that convention and could
// not be triggered if the test cases themselves lived in such a directory.
//
// Some enabled linters cannot be triggered because their rules are scoped to other
// paths, so there are no test cases for them:
//   - kubeapilinter: restricted to staging/src/k8s.io/api/.*.
//   - sorted: restricted to a fixed list of feature gate files.
//
// Some test cases exercise patterns that the configuration explicitly
// excludes from being reported (for example revive's "exported" rule, which
// is excluded everywhere except cmd/kubeadm). Those are included without a
// "// <linter> issue|hint: <regexp>" comment: if the exclusion ever broke,
// the resulting unexpected finding would fail the test.
//
// Updating files opened by this test (the configs, test case source) automatically
// invalidates the Go test cache. Updating hack/tools/golangci-lint or the
// verify script doesn't, so the "-count=1" trick must be used to force running
// the test in that case.
package golangcilint

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"

	"k8s.io/ktesting"
)

// expectation is one expected finding, derived from a "// <linter> issue: <regexp>" or
// "// <linter> hint: <regexp>" comment in one of the test case source files.
type expectation struct {
	linter  string // golangci-lint linter name, e.g. "govet"
	kind    string // "issue" or "hint"
	file    string // relative to the repository root, e.g. hack/testdata/golangci-lint/golangcilint.go
	line    int
	pattern *regexp.Regexp
}

// lintIssue is the subset of golangci-lint's JSON output that this test cares about.
type lintIssue struct {
	FromLinter string
	Text       string
	Pos        struct {
		Filename string
		Line     int
	}
}

type lintResult struct {
	Issues []lintIssue
}

var markerRE = regexp.MustCompile(`//\s*(\S+)\s+(issue|hint):\s*(.+?)\s*$`)

func TestGolangciLint(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx = tCtx.WithTimeout(5*time.Minute, "running hack/verify-golangci-lint.sh took too long")

	repoRoot := repoRoot(tCtx)
	expectations := collectExpectations(tCtx, repoRoot)
	if len(expectations) == 0 {
		tCtx.Fatal("no expectations found, test cases are missing or broken")
	}

	tmp := t.TempDir()
	noHintsConfig := filepath.Join(tmp, "no-hints.yaml")
	writeConfig(tCtx, filepath.Join(repoRoot, "hack/golangci.yaml"), noHintsConfig)
	hintsConfig := filepath.Join(tmp, "hints.yaml")
	writeConfig(tCtx, filepath.Join(repoRoot, "hack/golangci-hints.yaml"), hintsConfig)

	noHintIssues := runLint(tCtx, repoRoot, noHintsConfig)
	hintIssues := runLint(tCtx, repoRoot, hintsConfig)

	var wantIssuesOnly, wantIssuesAndHints []expectation
	for _, e := range expectations {
		wantIssuesAndHints = append(wantIssuesAndHints, e)
		if e.kind == "issue" {
			wantIssuesOnly = append(wantIssuesOnly, e)
		}
	}

	tCtx.Run("without-hints", func(tCtx ktesting.TContext) {
		checkIssues(tCtx, wantIssuesOnly, noHintIssues)
	})
	tCtx.Run("with-hints", func(tCtx ktesting.TContext) {
		checkIssues(tCtx, wantIssuesAndHints, hintIssues)
	})
}

// repoRoot determines the repository root by looking at the location of this
// source file: it is three directories below the root
// (test/integration/golangci-lint).
func repoRoot(tCtx ktesting.TContext) string {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		tCtx.Fatal("could not determine source file location")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(thisFile), "..", "..", ".."))
}

// collectExpectations scans hack/testdata/golangci-lint for Go source files
// and extracts "// <linter> issue: <regexp>" and "// <linter> hint: <regexp>"
// comments from them.
func collectExpectations(tCtx ktesting.TContext, repoRoot string) []expectation {
	testdataDir := filepath.Join(repoRoot, "hack", "testdata", "golangci-lint")

	var expectations []expectation
	entries, err := os.ReadDir(testdataDir)
	tCtx.ExpectNoError(err, "read test case directory")
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			continue
		}
		found, err := parseExpectations(repoRoot, filepath.Join(testdataDir, entry.Name()))
		tCtx.ExpectNoError(err, "parse %s", entry.Name())
		expectations = append(expectations, found...)
	}
	return expectations
}

// parseExpectations extracts expectations from a single Go source file. The
// marker comment either trails the code that is expected to trigger the
// finding ("someCode() // govet issue: some text") or stands on its own line
// right before that code.
func parseExpectations(repoRoot, path string) ([]expectation, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	relPath, err := filepath.Rel(repoRoot, path)
	if err != nil {
		return nil, err
	}
	relPath = filepath.ToSlash(relPath)

	var expectations []expectation
	lines := strings.Split(string(content), "\n")
	for i, line := range lines {
		match := markerRE.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		lineNo := i + 1
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			// Comment stands on its own line: it describes the next line.
			lineNo++
		}
		pattern, err := regexp.Compile(match[3])
		if err != nil {
			return nil, fmt.Errorf("%s:%d: invalid regexp %q in marker comment: %w", relPath, i+1, match[3], err)
		}
		expectations = append(expectations, expectation{
			linter:  match[1],
			kind:    match[2],
			file:    relPath,
			line:    lineNo,
			pattern: pattern,
		})
	}
	return expectations, nil
}

// writeConfig derives a temporary golangci-lint configuration:
//   - the exclusion for hack/testdata/golangci-lint gets disabled so that
//     it actually gets linted
//   - the output format gets switched from text to JSON, written to stdout
//     for easier parsing
func writeConfig(tCtx ktesting.TContext, from, to string) {
	content, err := os.ReadFile(from)
	tCtx.ExpectNoError(err, "read %s", from)
	config := string(content)

	const excludePath = "path: hack/testdata/golangci-lint/"
	if !strings.Contains(config, excludePath) {
		tCtx.Fatalf("%s: did not find expected exclusion %q", from, excludePath)
	}
	config = strings.Replace(config, excludePath, "path: no-such-directory", 1)

	const textOutput = "output:\n  formats:\n    text:\n      path: stderr\n"
	const jsonOutput = "output:\n  formats:\n    json:\n      path: stdout\n"
	if !strings.Contains(config, textOutput) {
		tCtx.Fatalf("%s: did not find expected output configuration %q", from, textOutput)
	}
	config = strings.Replace(config, textOutput, jsonOutput, 1)

	tCtx.ExpectNoError(os.WriteFile(to, []byte(config), 0644), "write %s", to)
}

// runLint invokes hack/verify-golangci-lint.sh with the given configuration,
// restricted to the golangci-lint test case packages, and parses the JSON
// output it prints on success and failure alike.
func runLint(tCtx ktesting.TContext, repoRoot, config string) []lintIssue {
	cmd := exec.CommandContext(tCtx, "bash", "hack/verify-golangci-lint.sh",
		"-c", config, "./hack/testdata/golangci-lint/...")
	cmd.Dir = repoRoot
	cmd.Env = os.Environ()
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	// The script intentionally returns a non-zero exit code when it finds
	// issues, which is expected here, so the error gets ignored as long as
	// the output can be parsed below.
	err := cmd.Run()

	issues, parseErr := parseLintOutput(stdout.Bytes())
	if parseErr != nil {
		tCtx.Logf("stdout:\n%s", stdout.String())
		tCtx.Logf("stderr:\n%s", stderr.String())
		tCtx.Fatalf("running %v: %v; failed to parse JSON output: %v", cmd.Args, err, parseErr)
	}
	return issues
}

// parseLintOutput extracts the single JSON object that golangci-lint writes
// among the other, human-readable output of hack/verify-golangci-lint.sh.
func parseLintOutput(output []byte) ([]lintIssue, error) {
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 || line[0] != '{' {
			continue
		}
		var result lintResult
		if err := json.Unmarshal(line, &result); err != nil {
			return nil, err
		}
		return result.Issues, nil
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return nil, errors.New("no JSON output line found")
}

// checkIssues matches each expectation against the actual issues, failing
// the test for every expectation that isn't matched by any issue and for
// every issue that isn't matched by any expectation.
func checkIssues(tCtx ktesting.TContext, expected []expectation, actual []lintIssue) {
	remaining := append([]lintIssue{}, actual...)
	for _, exp := range expected {
		found := -1
		for i, act := range remaining {
			if act.FromLinter == exp.linter && act.Pos.Filename == exp.file && act.Pos.Line == exp.line && exp.pattern.MatchString(act.Text) {
				found = i
				break
			}
		}
		if found < 0 {
			tCtx.Errorf("expected %s from %s at %s:%d matching %q was not reported", exp.kind, exp.linter, exp.file, exp.line, exp.pattern)
			continue
		}
		remaining = append(remaining[:found], remaining[found+1:]...)
	}
	for _, act := range remaining {
		tCtx.Errorf("unexpected issue reported by %s at %s:%d: %s", act.FromLinter, act.Pos.Filename, act.Pos.Line, act.Text)
	}
}
