/*
Copyright 2022 The Kubernetes Authors.

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

package junit

import (
	"encoding/xml"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

// WriteJUnitReport generates a JUnit file that is shorter than the one
// normally written by `ginkgo --junit-report`. This is needed because the full
// report can become too large for tools like Spyglass
// (https://github.com/kubernetes/kubernetes/issues/111510).
func WriteJUnitReport(report ginkgo.Report, filename string) error {
	config := reporters.JunitReportConfig{
		// Remove details for specs where we don't care.
		OmitTimelinesForSpecState: types.SpecStatePassed | types.SpecStateSkipped,

		// Don't write <failure message="summary">. The same text is
		// also in the full text for the failure. If we were to write
		// both, then tools like kettle and spyglass would concatenate
		// the two strings and thus show duplicated information.
		OmitFailureMessageAttr: true,

		// All labels are also part of the spec texts in inline [] tags,
		// so we don't need to write them separately.
		OmitSpecLabels: true,
	}

	// Sort specs by full name. The default is by start (or completion?) time,
	// which is less useful in spyglass because those times are not shown
	// and thus tests seem to be listed with no apparent order.
	slices.SortFunc(report.SpecReports, func(a, b types.SpecReport) int {
		res := strings.Compare(a.FullText(), b.FullText())
		if res == 0 {
			// Use start time as tie-breaker in the unlikely
			// case that two specs have the same full name.
			return a.StartTime.Compare(b.StartTime)
		}
		return res
	})

	detectDataRaces(report)

	if err := reporters.GenerateJUnitReportWithConfig(report, filename, config); err != nil {
		return err
	}

	// Ginkgo's JUnit reporter has no support for writing out the report
	// entries recorded via ginkgo.AddReportEntry
	// (https://github.com/onsi/ginkgo/issues/1431). We work around that by
	// loading the file that was just written, adding the entries as
	// <property> elements to their corresponding <testcase>, then writing
	// the file again.
	return addReportEntries(report, filename, config)
}

// junitTestSuites, junitTestSuite and junitTestCase mirror the corresponding
// reporters.JUnit* types (down to enough attributes to decode and re-encode a
// report unchanged) except that junitTestCase additionally supports
// <property> child elements, which is what we need to store report entries
// in a way that other JUnit XML consumers should recognize.
type junitTestSuites struct {
	XMLName    xml.Name         `xml:"testsuites"`
	Tests      int              `xml:"tests,attr"`
	Disabled   int              `xml:"disabled,attr"`
	Errors     int              `xml:"errors,attr"`
	Failures   int              `xml:"failures,attr"`
	Time       float64          `xml:"time,attr"`
	TestSuites []junitTestSuite `xml:"testsuite"`
}

type junitTestSuite struct {
	Name       string                    `xml:"name,attr"`
	Package    string                    `xml:"package,attr"`
	Tests      int                       `xml:"tests,attr"`
	Disabled   int                       `xml:"disabled,attr"`
	Skipped    int                       `xml:"skipped,attr"`
	Errors     int                       `xml:"errors,attr"`
	Failures   int                       `xml:"failures,attr"`
	Time       float64                   `xml:"time,attr"`
	Timestamp  string                    `xml:"timestamp,attr"`
	Properties reporters.JUnitProperties `xml:"properties"`
	TestCases  []junitTestCase           `xml:"testcase"`
}

type junitTestCase struct {
	Name       string                  `xml:"name,attr"`
	Classname  string                  `xml:"classname,attr"`
	Status     string                  `xml:"status,attr"`
	Time       float64                 `xml:"time,attr"`
	Owner      string                  `xml:"owner,attr,omitempty"`
	Skipped    *reporters.JUnitSkipped `xml:"skipped,omitempty"`
	Error      *reporters.JUnitError   `xml:"error,omitempty"`
	Failure    *reporters.JUnitFailure `xml:"failure,omitempty"`
	SystemOut  string                  `xml:"system-out,omitempty"`
	SystemErr  string                  `xml:"system-err,omitempty"`
	Properties *junitProperties        `xml:"properties,omitempty"`
}

type junitProperties struct {
	Properties []junitProperty `xml:"property"`
}

type junitProperty struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

// addReportEntries loads the JUnit file previously written by
// reporters.GenerateJUnitReportWithConfig, copies all report entries
// (ginkgo.AddReportEntry) from report into the corresponding <testcase> as
// <property> elements, then writes the file back.
func addReportEntries(report ginkgo.Report, filename string, config reporters.JunitReportConfig) error {
	specs := report.SpecReports
	if config.OmitSuiteSetupNodes {
		filtered := make([]types.SpecReport, 0, len(specs))
		for _, spec := range specs {
			if spec.LeafNodeType == types.NodeTypeIt {
				filtered = append(filtered, spec)
			}
		}
		specs = filtered
	}

	// No entries anywhere, so the file doesn't need to be touched.
	hasEntries := false
	for _, spec := range specs {
		if len(spec.ReportEntries) > 0 {
			hasEntries = true
			break
		}
	}
	if !hasEntries {
		return nil
	}

	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	var suites junitTestSuites
	if err := xml.Unmarshal(data, &suites); err != nil {
		return fmt.Errorf("parsing %s: %w", filename, err)
	}
	if len(suites.TestSuites) != 1 {
		return fmt.Errorf("expected exactly one JUnit test suite in %s, got %d", filename, len(suites.TestSuites))
	}
	suite := &suites.TestSuites[0]
	if len(suite.TestCases) != len(specs) {
		return fmt.Errorf("expected %d JUnit test cases in %s, got %d", len(specs), filename, len(suite.TestCases))
	}

	for i, spec := range specs {
		if len(spec.ReportEntries) == 0 {
			continue
		}
		properties := &junitProperties{}
		for _, entry := range spec.ReportEntries {
			properties.Properties = append(properties.Properties, junitProperty{
				Name:  entry.Name,
				Value: entry.StringRepresentation(),
			})
		}
		suite.TestCases[i].Properties = properties
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	if _, err := f.WriteString(xml.Header); err != nil {
		f.Close()
		return err
	}
	encoder := xml.NewEncoder(f)
	encoder.Indent("  ", "    ")
	if err := encoder.Encode(suites); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
