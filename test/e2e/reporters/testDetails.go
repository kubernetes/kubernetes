/*
Copyright 2019 The Kubernetes Authors.

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

package reporters

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/types"
	"k8s.io/klog/v2"
)

// DetailsReporter is a ginkgo reporter which dumps information regarding the tests which is difficult to get
// via an AST app. This allows us to leverage the existing ginkgo logic to walk the tests and such while then following
// up with an custom app which leverages AST to generate conformance documentation.
type DetailsReporter struct {
	Writer io.Writer
}

// NewDetailsReporterWithWriter returns a reporter which will write the SpecSummary objects as tests
// complete to the given writer.
func NewDetailsReporterWithWriter(w io.Writer) *DetailsReporter {
	return &DetailsReporter{
		Writer: w,
	}
}

// NewDetailsReporterFile returns a reporter which will create the file given and dump the specs
// to it as they complete.
func NewDetailsReporterFile(filename string) *DetailsReporter {
	absPath, err := filepath.Abs(filename)
	if err != nil {
		klog.Errorf("%#v\n", err)
		panic(err)
	}
	f, err := os.Create(absPath)
	if err != nil {
		klog.Errorf("%#v\n", err)
		panic(err)
	}
	return NewDetailsReporterWithWriter(f)
}

// SpecSuiteWillBegin is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *DetailsReporter) SpecSuiteWillBegin(cfg config.GinkgoConfigType, summary *types.SuiteSummary) {
}

// SpecSuiteDidEnd is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *DetailsReporter) SpecSuiteDidEnd(summary *types.SuiteSummary) {}

// SpecDidComplete is invoked by Ginkgo each time a spec is completed (including skipped specs).
func (reporter *DetailsReporter) SpecDidComplete(specSummary *types.SpecSummary) {
	b, err := json.Marshal(specSummary)
	if err != nil {
		klog.Errorf("Error in detail reporter: %v", err)
		return
	}
	_, err = reporter.Writer.Write(b)
	if err != nil {
		klog.Errorf("Error saving test details in detail reporter: %v", err)
		return
	}
	// Printing newline between records for easier viewing in various tools.
	_, err = fmt.Fprintln(reporter.Writer, "")
	if err != nil {
		klog.Errorf("Error saving test details in detail reporter: %v", err)
		return
	}
}

// SpecWillRun is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *DetailsReporter) SpecWillRun(specSummary *types.SpecSummary) {}

// BeforeSuiteDidRun is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *DetailsReporter) BeforeSuiteDidRun(setupSummary *types.SetupSummary) {}

// AfterSuiteDidRun is implemented as a noop to satisfy the reporter interface for ginkgo.
func (reporter *DetailsReporter) AfterSuiteDidRun(setupSummary *types.SetupSummary) {
	if c, ok := reporter.Writer.(io.Closer); ok {
		c.Close()
	}
}
