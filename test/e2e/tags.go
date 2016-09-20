/*
Copyright 2015 The Kubernetes Authors.

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

package e2e

import (
	"strings"
)

// The use of ginkgo tags is encouraged for describing tests.
// This file defines test tags and annotations.
// Each annotation should have a corresponding description.
// Tags can possibly have different metadata over time.
type Tag struct {
	Name string
}

// Conformance tests are tests we expect to pass on **any** Kubernetes cluster.
// Does not supersede any other labels.
// Test policies are a work-in-progress; see #18162.
var TagConformanceTest = Tag{"[Conformance]"}

// If a test takes more than five minutes to run (by itself or in parallel with many other tests), it is labeled `[Slow]`.
// This partition allows us to run almost all of our tests quickly in parallel, without waiting for the stragglers to finish.
var TagSlowTest = Tag{"[Slow]"}

// If a test cannot be run in parallel with other tests (e.g. it takes too many resources or restarts nodes),
// it is labeled `[Serial]`, and should be run in serial as part of a separate suite.
var TagSerialTest = Tag{"[Serial]"}

// If a test restarts components that might cause other tests to fail or break the cluster completely, it is labeled
// `[Disruptive]`. Any `[Disruptive]` test is also assumed to qualify for the `[Serial]` label, but need not be labeled
// as both.  These tests are not run against soak clusters to avoid restarting components.
var TagDisruptive = Tag{"[Disruptive]"}

// If a test is found to be flaky and we have decided that it's too hard to fix in the short term
// (e.g. it's going to take a full engineer-week), it receives the [Flaky] label until it is fixed.
// The [Flaky] label should be used very sparingly, and should be accompanied with a reference to the issue for
// de-flaking the test, because while a test remains labeled since otherwise these aren't monitored closely in CI.
// These are by default not run, unless a `focus` or `skip` argument is explicitly given.
var TagFlakey = Tag{"[Flaky]"}

// TagFeature is for things that can be turned on/off at deployment time (eg: Volumes, Ingress)
func TagFeature(f string) string {
	return "[Feature:" + f + "]"
}

// TagMeasurement is for things that measure things like performance, scale, etc.
func TagMeasurement(f string) string {
	return "[Measurement:" + f + "]"
}

// TagExperimental is for Catch-all for tests which are off the critical path, but not necessarily slow/flaky.
func TagQuirk(f string) string {
	return "[Quirk:" + f + "]"
}

// Function for detecting bad tags.  We can post-process Its with this, and report them.
func getBadTags(description string) []string {
	badTags := []string{}
	validTags := []string{
		"Feature",
		"Measurement",
		"Quirk",
		TagConformanceTest.Name,
		TagDisruptive.Name,
		TagFlakey.Name,
		TagSerialTest.Name,
		TagSlowTest.Name,
	}
	for _, proposedTag := range splitBetweenChars(description, '[', ']') {
		valid := false
		for _, aValidTag := range validTags {
			// If this tag has one of the validTokens, its ok.
			if strings.Contains(proposedTag, aValidTag) {
				valid = true
			}
		}
		if !valid {
			badTags = append(badTags, proposedTag)
		}
	}
	return badTags
}

// splitBetweenChars used to split out tags i.e. "It [Feature:blah] [Flaky]"->string[]{"Feature:blah","Flaky"}
func splitBetweenChars(s string, start byte, end byte) []string {
	entries := []string{}

	indices := func(tmpStr string, border byte) []int {
		ints := []int{}
		for index, i := range tmpStr {
			if byte(i) == border {
				ints = append(ints, index)
			}
		}
		return ints
	}

	starts := indices(s, '[')
	ends := indices(s, ']')

	for i, startLoc := range starts {
		tag := s[startLoc+1 : ends[i]]
		entries = append(entries, tag)
	}
	return entries
}
