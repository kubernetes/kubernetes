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
// The test also determines, via "golangci-lint linters", which linters are
// enabled by either configuration and fails if one of them never reported
// anything in either sub-test: an enabled linter that never fires is either
// dead configuration or untested. Linters which genuinely cannot be triggered
// from test cases (for example because their rules are scoped to other paths)
// must be documented with a "// <linter> untested: <free form explanation>"
// comment instead, which suppresses that failure.
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

// untestedMarker records a "// <linter> untested: <free form explanation>"
// comment, documenting that a linter cannot be triggered by any test case and
// thus is allowed to never show up in the actual findings.
type untestedMarker struct {
	linter      string
	explanation string
	file        string
	line        int
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

// lintersResult is the subset of the JSON output of "golangci-lint linters
// --json" that this test cares about.
type lintersResult struct {
	Enabled []struct {
		Name string
	}
}

var markerRE = regexp.MustCompile(`//\s*(\S+)\s+(issue|hint|untested):\s*(.+?)\s*$`)

func TestGolangciLint(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx = tCtx.WithTimeout(5*time.Minute, "running hack/verify-golangci-lint.sh took too long")

	repoRoot := repoRoot(tCtx)
	wantReports, expectUntested := collectExpectations(tCtx, repoRoot)
	if len(wantReports) == 0 || len(expectUntested) == 0 {
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
	for _, report := range wantReports {
		wantIssuesAndHints = append(wantIssuesAndHints, report)
		if report.kind == "issue" {
			wantIssuesOnly = append(wantIssuesOnly, report)
		}
	}

	tCtx.Run("without-hints", func(tCtx ktesting.TContext) {
		checkIssues(tCtx, wantIssuesOnly, noHintIssues)
	})
	tCtx.Run("with-hints", func(tCtx ktesting.TContext) {
		checkIssues(tCtx, wantIssuesAndHints, hintIssues)
	})

	// Determine which linters are actually enabled by either configuration
	// and fail for each of them which never showed up among the issues
	// reported by either sub-test above, unless it was explicitly marked
	// as untested.
	enabledLinters := listEnabledLinters(tCtx, repoRoot, noHintsConfig)
	for linter := range listEnabledLinters(tCtx, repoRoot, hintsConfig) {
		enabledLinters[linter] = true
	}
	reportedLinters := map[string]bool{}
	for _, issue := range noHintIssues {
		reportedLinters[issue.FromLinter] = true
	}
	for _, issue := range hintIssues {
		reportedLinters[issue.FromLinter] = true
	}
	untestedLinters := map[string]untestedMarker{}
	for _, untested := range expectUntested {
		untestedLinters[untested.linter] = untested
	}
	for linter := range enabledLinters {
		if reportedLinters[linter] {
			continue
		}
		if _, ok := untestedLinters[linter]; ok {
			continue // documented as untested.
		}
		tCtx.Errorf("linter %q is enabled, but did not report any issue or hint in either sub-test; either add a test case or document why with a %q comment", linter, fmt.Sprintf("// %s untested: <explanation>", linter))
	}
	for linter, u := range untestedLinters {
		if !enabledLinters[linter] {
			tCtx.Errorf("%s:%d: %q is marked as untested, but is not enabled by either configuration", u.file, u.line, linter)
		} else if reportedLinters[linter] {
			tCtx.Errorf("%s:%d: %q is marked as untested, but did report an issue or hint, remove the marker comment", u.file, u.line, linter)
		}
	}
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
// comments from them, together with "// <linter> untested: <explanation>"
// markers.
func collectExpectations(tCtx ktesting.TContext, repoRoot string) ([]expectation, []untestedMarker) {
	testdataDir := filepath.Join(repoRoot, "hack", "testdata", "golangci-lint")

	var expectations []expectation
	var untested []untestedMarker
	entries, err := os.ReadDir(testdataDir)
	tCtx.ExpectNoError(err, "read test case directory")
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			continue
		}
		found, foundUntested, err := parseExpectations(repoRoot, filepath.Join(testdataDir, entry.Name()))
		tCtx.ExpectNoError(err, "parse %s", entry.Name())
		expectations = append(expectations, found...)
		untested = append(untested, foundUntested...)
	}
	return expectations, untested
}

// parseExpectations extracts expectations and untested markers from a single
// Go source file. The marker comment either trails the code that is expected
// to trigger the finding ("someCode() // govet issue: some text") or stands
// on its own line right before that code. "untested" markers typically stand
// on their own, not tied to a specific line of code.
func parseExpectations(repoRoot, path string) ([]expectation, []untestedMarker, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	relPath, err := filepath.Rel(repoRoot, path)
	if err != nil {
		return nil, nil, err
	}
	relPath = filepath.ToSlash(relPath)

	var expectations []expectation
	var untested []untestedMarker
	lines := strings.Split(string(content), "\n")
	for i, line := range lines {
		match := markerRE.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		lineNo := i + 1
		linter, kind, text := match[1], match[2], match[3]
		if kind == "untested" {
			untested = append(untested, untestedMarker{
				linter:      linter,
				explanation: text,
				file:        relPath,
				line:        lineNo,
			})
			continue
		}
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			// Comment stands on its own line: it describes the next line.
			lineNo++
		}
		pattern, err := regexp.Compile(text)
		if err != nil {
			return nil, nil, fmt.Errorf("%s:%d: invalid regexp %q in marker comment: %w", relPath, i+1, text, err)
		}
		expectations = append(expectations, expectation{
			linter:  linter,
			kind:    kind,
			file:    relPath,
			line:    lineNo,
			pattern: pattern,
		})
	}
	return expectations, untested, nil
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

// golangciLintBinary returns the path at which hack/verify-golangci-lint.sh
// builds the golangci-lint binary (see kube::golang::setup_env in
// hack/lib/init.sh). runLint must have been called at least once before this
// so that the binary actually exists.
func golangciLintBinary(repoRoot string) string {
	subpath := os.Getenv("KUBE_OUTPUT_SUBPATH")
	if subpath == "" {
		subpath = "_output/local"
	}
	return filepath.Join(repoRoot, subpath, "bin", "golangci-lint")
}

// listEnabledLinters runs "golangci-lint linters" with the given
// configuration and returns the set of linters it reports as enabled.
func listEnabledLinters(tCtx ktesting.TContext, repoRoot, config string) map[string]bool {
	binary := golangciLintBinary(repoRoot)
	cmd := exec.CommandContext(tCtx, binary, "linters", "-c", config, "--json")
	cmd.Dir = repoRoot
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		tCtx.Logf("stdout:\n%s", stdout.String())
		tCtx.Logf("stderr:\n%s", stderr.String())
		tCtx.Fatalf("running %v: %v", cmd.Args, err)
	}

	var result lintersResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		tCtx.Fatalf("parsing output of %v: %v\noutput:\n%s", cmd.Args, err, stdout.String())
	}

	enabled := make(map[string]bool, len(result.Enabled))
	for _, linter := range result.Enabled {
		enabled[linter.Name] = true
	}
	return enabled
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
