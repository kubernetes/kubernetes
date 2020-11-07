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

// Package testsuites_test is used intentionally to ensure that the
// code below only has access to exported names. It doesn't have any
// actual test. That the custom volume test suite defined below
// compile is the test.
//
// It's needed because we don't have any in-tree volume test
// suite implementations that aren't in the "testuites" package itself.
// We don't need this for the "TestDriver" interface because there
// we have implementations in a separate package.
package testsuites_test

import (
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
)

// test suite will have to explicitly set TestSuiteInterface in their stuct.
// In order to define tests, they will have to use TestSuiteHandler
type fakeSuite struct {
	testsuites.TestSuiteInterface
}

//lint:ignore U1000 we dont need this to be called because the custom volume test suite
// defined below compile is the test.
func (f *fakeSuite) getTestSuiteInfo() testsuites.TestSuiteInfo {
	return testsuites.TestSuiteInfo{
		Name:               "fake",
		FeatureTag:         "",
		TestPatterns:       []testpatterns.TestPattern{testpatterns.DefaultFsDynamicPV},
		SupportedSizeRange: e2evolume.SizeRange{Min: "1Mi", Max: "1Gi"},
	}
}

//lint:ignore U1000 we dont need this to be called because the custom volume test suite
// defined below compile is the test.
func (f *fakeSuite) defineTests(testsuites.TestDriver, testpatterns.TestPattern) {
}

//lint:ignore U1000 we dont need this to be called because the custom volume test suite
// defined below compile is the test.
func (f *fakeSuite) skipUnsupportedTests(testsuites.TestDriver, testpatterns.TestPattern) {
}

var _ testsuites.TestSuiteInterface = &fakeSuite{}
