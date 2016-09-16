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

package main

import (
	"fmt"
	"strings"
)

const unversionedWarningTag = "UNVERSIONED_WARNING"

const unversionedWarningPre = `
<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.`

const unversionedWarningLink = `

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/%s/%s).`

const unversionedWarningPost = `

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

`

func makeUnversionedWarning(fileName string, linkToReleaseDoc bool) mungeLines {
	var insert string
	if linkToReleaseDoc {
		insert = unversionedWarningPre + fmt.Sprintf(unversionedWarningLink, latestReleaseBranch, fileName) + unversionedWarningPost
	} else {
		insert = unversionedWarningPre + unversionedWarningPost
	}
	return getMungeLines(insert)
}

// inserts/updates a warning for unversioned docs
func updateUnversionedWarning(file string, mlines mungeLines) (mungeLines, error) {
	file, err := makeRepoRelative(file, file)
	if err != nil {
		return mlines, err
	}
	if hasLine(mlines, "<!-- TAG IS_VERSIONED -->") {
		// No warnings on release branches
		return mlines, nil
	}

	var linkToReleaseDoc bool
	checkDocExistence := inJenkins || (len(filesInLatestRelease) != 0)
	if checkDocExistence {
		linkToReleaseDoc = strings.Contains(filesInLatestRelease, file)
	} else {
		linkToReleaseDoc = hasLine(mlines, "<!-- TAG RELEASE_LINK, added by the munger automatically -->")
	}

	if !hasMacroBlock(mlines, unversionedWarningTag) {
		mlines = prependMacroBlock(unversionedWarningTag, mlines)
	}

	mlines, err = updateMacroBlock(mlines, unversionedWarningTag, makeUnversionedWarning(file, linkToReleaseDoc))
	if err != nil {
		return mlines, err
	}
	return mlines, nil
}
