//go:build coverage

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

// Package coverage provides tools for coverage-instrumented binaries to collect and
// flush coverage information.
package coverage

import (
	"flag"
	"fmt"
	"os"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

var coverageFile string

// tempCoveragePath returns a temporary file to write coverage information to.
// The file is in the same directory as the destination, ensuring os.Rename will work.
func tempCoveragePath() string {
	return coverageFile + ".tmp"
}

// InitCoverage is called from the dummy unit test to prepare Go's coverage framework.
// Clients should never need to call it.
func InitCoverage(name string) {
	// We read the coverage destination in from the KUBE_COVERAGE_FILE env var,
	// or if it's empty we just use a default in /tmp
	coverageFile = os.Getenv("KUBE_COVERAGE_FILE")
	if coverageFile == "" {
		coverageFile = "/tmp/k8s-" + name + ".cov"
	}
	fmt.Println("Dumping coverage information to " + coverageFile)

	flushInterval := 5 * time.Second
	requestedInterval := os.Getenv("KUBE_COVERAGE_FLUSH_INTERVAL")
	if requestedInterval != "" {
		if duration, err := time.ParseDuration(requestedInterval); err == nil {
			flushInterval = duration
		} else {
			panic("Invalid KUBE_COVERAGE_FLUSH_INTERVAL value; try something like '30s'.")
		}
	}

	// Set up the unit test framework with the required arguments to activate test coverage.
	flag.CommandLine.Parse([]string{"-test.coverprofile", tempCoveragePath()})

	// Begin periodic logging
	go wait.Forever(FlushCoverage, flushInterval)
}

// FlushCoverage flushes collected coverage information to disk.
// The destination file is configured at startup and cannot be changed.
// Calling this function also sends a line like "coverage: 5% of statements" to stdout.
func FlushCoverage() {
	// We're not actually going to run any tests, but we need Go to think we did so it writes
	// coverage information to disk. To achieve this, we create a bunch of empty test suites and
	// have it "run" them.
	tests := []testing.InternalTest{}
	benchmarks := []testing.InternalBenchmark{}
	examples := []testing.InternalExample{}
	fuzztargets := []testing.InternalFuzzTarget{}

	var deps fakeTestDeps

	dummyRun := testing.MainStart(deps, tests, benchmarks, fuzztargets, examples)
	dummyRun.Run()

	// Once it writes to the temporary path, we move it to the intended path.
	// This gets us atomic updates from the perspective of another process trying to access
	// the file.
	if err := os.Rename(tempCoveragePath(), coverageFile); err != nil {
		klog.Errorf("Couldn't move coverage file from %s to %s", coverageFile, tempCoveragePath())
	}
}
