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

	"github.com/onsi/ginkgo/v2"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
)

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
	// SkipUnsupportedTests will skip the test suite based on the given TestPattern, TestDriver
	// Testsuite should check if the given pattern and driver works for the "whole testsuite"
	// Testcase specific check should happen inside defineTests
	SkipUnsupportedTests(TestDriver, TestPattern)
}

// RegisterTests register the driver + pattern combination to the inside TestSuite
// This function actually register tests inside testsuite
func RegisterTests(suite TestSuite, driver TestDriver, pattern TestPattern) {
	tsInfo := suite.GetTestSuiteInfo()
	testName := fmt.Sprintf("[Testpattern: %s]%s %s%s", pattern.Name, pattern.FeatureTag, tsInfo.Name, tsInfo.FeatureTag)
	ginkgo.Context(testName, func() {
		ginkgo.BeforeEach(func() {
			// skip all the invalid combination of driver and pattern
			SkipInvalidDriverPatternCombination(driver, pattern)
			// skip the unsupported test pattern and driver combination specific for this TestSuite
			suite.SkipUnsupportedTests(driver, pattern)
		})
		// actually define the tests
		// at this step the testsuite should not worry about if the pattern and driver
		// does not fit for the whole testsuite. But driver&pattern check
		// might still needed for specific independent test cases.
		suite.DefineTests(driver, pattern)
	})
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
	FeatureTag         string              // featureTag for the TestSuite
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
func SkipInvalidDriverPatternCombination(driver TestDriver, pattern TestPattern) {
	dInfo := driver.GetDriverInfo()
	var isSupported bool

	// 0. Check with driver specific logic
	driver.SkipUnsupportedTest(pattern)

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
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
	}

	// 2. Check if fsType is supported
	if !dInfo.SupportedFsType.Has(pattern.FsType) {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.FsType)
	}
	if pattern.FsType == "xfs" && framework.NodeOSDistroIs("windows") {
		e2eskipper.Skipf("Distro doesn't support xfs -- skipping")
	}
	if pattern.FsType == "ntfs" && !framework.NodeOSDistroIs("windows") {
		e2eskipper.Skipf("Distro %s doesn't support ntfs -- skipping", framework.TestContext.NodeOSDistro)
	}
}
