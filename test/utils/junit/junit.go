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

// Package junit provides data structures to allow easy XML encoding
// and decoding of JUnit test results.
package junit

import (
	"encoding/xml"
	"time"
)

// TestSuite is a top-level test suite containing test cases.
type TestSuite struct {
	XMLName xml.Name `xml:"testsuite"`

	Name      string    `xml:"name,attr"`
	Tests     int       `xml:"tests,attr"`
	Disabled  int       `xml:"disabled,attr,omitempty"`
	Errors    int       `xml:"errors,attr"`
	Failures  int       `xml:"failures,attr"`
	Skipped   int       `xml:"skipped,attr,omitempty"`
	Time      float64   `xml:"time,attr"`
	Timestamp time.Time `xml:"timestamp,attr"`
	ID        int       `xml:"id,attr,omitempty"`
	Package   string    `xml:"package,attr,omitempty"`
	Hostname  string    `xml:"hostname,attr"`

	Properties []*Property `xml:"properties,omitempty"`
	TestCases  []*TestCase `xml:"testcase"`

	SystemOut string `xml:"system-out,omitempty"`
	SystemErr string `xml:"system-err,omitempty"`
}

// Update iterates through the TestCases and updates Tests, Errors,
// Failures, and Skipped top level attributes.
func (t *TestSuite) Update() {
	t.Tests = len(t.TestCases)
	for _, tc := range t.TestCases {
		t.Errors += len(tc.Errors)
		t.Failures += len(tc.Failures)
		if len(tc.Skipped) > 0 {
			t.Skipped++
		}
	}
}

// Property is a simple key-value property that can be attached to a TestSuite.
type Property struct {
	XMLName xml.Name `xml:"property"`

	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

// Error represents the errors in a test case.
type Error struct {
	XMLName xml.Name `xml:"error"`

	Message string `xml:"message,attr,omitempty"`
	Type    string `xml:"type,attr"`

	Value string `xml:",cdata"`
}

// Failure represents the failures in a test case.
type Failure struct {
	XMLName xml.Name `xml:"failure"`

	Message string `xml:"message,attr,omitempty"`
	Type    string `xml:"type,attr"`

	Value string `xml:",cdata"`
}

// TestCase represents a single test case within a suite.
type TestCase struct {
	XMLName xml.Name `xml:"testcase"`

	Name       string  `xml:"name,attr"`
	Classname  string  `xml:"classname,attr"`
	Status     string  `xml:"status,attr,omitempty"`
	Assertions int     `xml:"assertions,attr,omitempty"`
	Time       float64 `xml:"time,attr"`

	Skipped string `xml:"skipped,omitempty"`

	Errors   []*Error   `xml:"error,omitempty"`
	Failures []*Failure `xml:"failure,omitempty"`
}
