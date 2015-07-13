/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

const unversionedWarningTag = "UNVERSIONED_WARNING"

var beginUnversionedWarning = beginMungeTag(unversionedWarningTag)
var endUnversionedWarning = endMungeTag(unversionedWarningTag)

const unversionedWarning = `
<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->
`

// inserts/updates a warning for unversioned docs
func updateUnversionedWarning(file string, markdown []byte) ([]byte, error) {
	lines := splitLines(markdown)
	if hasLine(lines, "<!-- TAG IS_VERSIONED -->") {
		// No warnings on release branches
		return markdown, nil
	}
	if !hasMacroBlock(lines, beginUnversionedWarning, endUnversionedWarning) {
		lines = append([]string{beginUnversionedWarning, endUnversionedWarning}, lines...)
	}
	return updateMacroBlock(lines, beginUnversionedWarning, endUnversionedWarning, unversionedWarning)
}
