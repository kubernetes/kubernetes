/*
Copyright 2017 The Kubernetes Authors.

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

package validation_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/ginkgo/config"
	. "github.com/onsi/ginkgo/types"
	. "github.com/onsi/gomega"

	"fmt"
	"testing"
)

func TestOpenapi(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecsWithDefaultAndCustomReporters(t, "Openapi Suite", []Reporter{newlineReporter{}})
}

// Print a newline after the default newlineReporter due to issue
// https://github.com/jstemmer/go-junit-report/issues/31
type newlineReporter struct{}

func (newlineReporter) SpecSuiteWillBegin(config GinkgoConfigType, summary *SuiteSummary) {}

func (newlineReporter) BeforeSuiteDidRun(setupSummary *SetupSummary) {}

func (newlineReporter) AfterSuiteDidRun(setupSummary *SetupSummary) {}

func (newlineReporter) SpecWillRun(specSummary *SpecSummary) {}

func (newlineReporter) SpecDidComplete(specSummary *SpecSummary) {}

// SpecSuiteDidEnd Prints a newline between "35 Passed | 0 Failed | 0 Pending | 0 Skipped" and "--- PASS:"
func (newlineReporter) SpecSuiteDidEnd(summary *SuiteSummary) { fmt.Printf("\n") }
