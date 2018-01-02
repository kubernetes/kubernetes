package main

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"regexp"
)

var testFuncRegexp *regexp.Regexp

func init() {
	testFuncRegexp = regexp.MustCompile(`(?m)^\s*func\s+\(\w*\s*\*(\w+Suite)\)\s+(Test\w+)`)
}

func enumerateTestsForBytes(b []byte) ([]string, error) {
	var tests []string
	submatches := testFuncRegexp.FindAllSubmatch(b, -1)
	for _, submatch := range submatches {
		if len(submatch) == 3 {
			tests = append(tests, fmt.Sprintf("%s.%s$", submatch[1], submatch[2]))
		}
	}
	return tests, nil
}

// enumerateTests enumerates valid `-check.f` strings for all the test functions.
// Note that we use regexp rather than parsing Go files for performance reason.
// (Try `TESTFLAGS=-check.list make test-integration-cli` to see the slowness of parsing)
// The files needs to be `gofmt`-ed
//
// The result will be as follows, but unsorted ('$' is appended because they are regexp for `-check.f`):
//  "DockerAuthzSuite.TestAuthZPluginAPIDenyResponse$"
//  "DockerAuthzSuite.TestAuthZPluginAllowEventStream$"
//  ...
//  "DockerTrustedSwarmSuite.TestTrustedServiceUpdate$"
func enumerateTests(wd string) ([]string, error) {
	testGoFiles, err := filepath.Glob(filepath.Join(wd, "integration-cli", "*_test.go"))
	if err != nil {
		return nil, err
	}
	var allTests []string
	for _, testGoFile := range testGoFiles {
		b, err := ioutil.ReadFile(testGoFile)
		if err != nil {
			return nil, err
		}
		tests, err := enumerateTestsForBytes(b)
		if err != nil {
			return nil, err
		}
		allTests = append(allTests, tests...)
	}
	return allTests, nil
}
