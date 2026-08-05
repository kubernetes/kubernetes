/*
Copyright 2020 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"os"

	"k8s.io/kubernetes/test/e2e/framework"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
)

func DebugStorageTestsuite(format string, args ...any) {
	if _, ok := os.LookupEnv("DEBUG_STORAGE_TESTSUITE"); !ok {
		return
	}
	fmt.Printf(format, args...)
}

// TestSuite represents an interface for a set of tests which works with TestDriver.
// Each testsuite should implement this interface.
// All the functions except GetTestSuiteInfo() should not be called directly. Instead,
// use RegisterTests() to register the tests in a more standard way.
type TestSuite interface {
	GetTestSuiteInfo() TestSuiteInfo
	// DefineTests defines tests of the testpattern for the driver.
	// Called inside a Ginkgo context that reflects the current driver and test pattern,
	// so the test suite can define tests directly with ginkgo.It.
	DefineTests(TestDriver, TestPattern)
	// SkipUnsupportedTests will return a reason why the the given combination of TestPattern
	// and TestDriver cannot be tested. No tests get registered.
	//
	// Testsuite should check if the given pattern and driver works for the "whole testsuite"
	// Testcase specific check should happen inside defineTests. Unsupported
	// tests should not even get defined.
	SkipUnsupportedTests(TestDriver, TestPattern) string
}

// RegisterTests register the driver + pattern combination to the inside TestSuite
// This function actually register tests inside testsuite
func RegisterTests(suite TestSuite, driver TestDriver, pattern TestPattern) {
	tsInfo := suite.GetTestSuiteInfo()
	var args []interface{}
	args = append(args, fmt.Sprintf("[Testpattern: %s]", pattern.Name))
	args = append(args, pattern.TestTags...)
	args = append(args, tsInfo.Name)
	args = append(args, tsInfo.TestTags...)
	args = append(args, func() {
		reason := SkipInvalidDriverPatternCombination(driver, pattern)
		if reason == "" {
			reason = suite.SkipUnsupportedTests(driver, pattern)
		}
		if reason != "" {
			DebugStorageTestsuite("Skipping registration: [Testpattern: %s] %s [Driver: %s] - %s\n", pattern.Name, tsInfo.Name, driver.GetDriverInfo().Name, reason)
			return
		}

		// actually define the tests
		// at this step the testsuite should not worry about if the pattern and driver
		// does not fit for the whole testsuite. But driver&pattern check
		// might still needed for specific independent test cases.
		suite.DefineTests(driver, pattern)
	})
	framework.Context(args...)
}

// DefineTestSuites defines tests for all testpatterns and all testSuites for a driver
func DefineTestSuites(driver TestDriver, tsInits []func() TestSuite) {
	for _, testSuiteInit := range tsInits {
		suite := testSuiteInit()
		for _, pattern := range suite.GetTestSuiteInfo().TestPatterns {
			RegisterTests(suite, driver, pattern)
		}
	}
}

// TestSuiteInfo represents a set of parameters for TestSuite
type TestSuiteInfo struct {
	Name               string              // name of the TestSuite
	TestTags           []interface{}       // additional parameters for framework.It, like framework.WithDisruptive()
	TestPatterns       []TestPattern       // Slice of TestPattern for the TestSuite
	SupportedSizeRange e2evolume.SizeRange // Size range supported by the test suite
}

// SkipInvalidDriverPatternCombination will skip tests if the combination of driver, and testpattern
// is not compatible to be tested. This function will be called in the RegisterTests() to make
// sure all the testsuites we defined are valid.
//
// Whether it needs to be skipped is checked by following steps:
// 0. Check with driver SkipUnsupportedTest
// 1. Check if volType is supported by driver from its interface
// 2. Check if fsType is supported
//
// Test suites can also skip tests inside their own skipUnsupportedTests function or in
// individual tests.
func SkipInvalidDriverPatternCombination(driver TestDriver, pattern TestPattern) string {
	dInfo := driver.GetDriverInfo()
	var isSupported bool

	// 0. Check with driver specific logic
	if reason := driver.SkipUnsupportedTest(pattern); reason != "" {
		return reason
	}

	// 1. Check if Whether volType is supported by driver from its interface
	switch pattern.VolType {
	case InlineVolume:
		_, isSupported = driver.(InlineVolumeTestDriver)
	case PreprovisionedPV:
		_, isSupported = driver.(PreprovisionedPVTestDriver)
	case DynamicPV, GenericEphemeralVolume:
		_, isSupported = driver.(DynamicPVTestDriver)
	case CSIInlineVolume:
		_, isSupported = driver.(EphemeralTestDriver)
	default:
		isSupported = false
	}

	if !isSupported {
		return fmt.Sprintf("Driver %s doesn't support %v", dInfo.Name, pattern.VolType)
	}

	// 2. Check if fsType is supported
	if !dInfo.SupportedFsType.Has(pattern.FsType) {
		return fmt.Sprintf("Driver %s doesn't support %v", dInfo.Name, pattern.FsType)
	}
	if pattern.FsType == "xfs" && framework.NodeOSDistroIs("windows") {
		return "Distro doesn't support xfs"
	}
	if pattern.FsType == "ntfs" && !framework.NodeOSDistroIs("windows") {
		return fmt.Sprintf("Distro %s doesn't support ntfs", framework.TestContext.NodeOSDistro)
	}

	return ""
}
