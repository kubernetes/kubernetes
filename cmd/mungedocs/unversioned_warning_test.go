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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUnversionedWarning(t *testing.T) {
	beginMark := beginMungeTag(unversionedWarningTag)
	endMark := endMungeTag(unversionedWarningTag)

	warningString := makeUnversionedWarning("filename.md", false).String()
	warningBlock := beginMark + "\n" + warningString + endMark + "\n"
	var cases = []struct {
		in       string
		expected string
	}{
		{"", warningBlock},
		{
			"Foo\nBar\n",
			warningBlock + "\nFoo\nBar\n",
		},
		{
			"Foo\n<!-- TAG IS_VERSIONED -->\nBar",
			"Foo\n<!-- TAG IS_VERSIONED -->\nBar",
		},
		{
			beginMark + "\n" + endMark + "\n",
			warningBlock,
		},
		{
			beginMark + "\n" + "something\n" + endMark + "\n",
			warningBlock,
		},
		{
			"Foo\n" + beginMark + "\n" + endMark + "\nBar\n",
			"Foo\n" + warningBlock + "Bar\n",
		},
		{
			"Foo\n" + warningBlock + "Bar\n",
			"Foo\n" + warningBlock + "Bar\n",
		},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		actual, err := updateUnversionedWarning("filename.md", in)
		assert.NoError(t, err)
		if !expected.Equal(actual) {
			t.Errorf("case[%d]: expected %v got %v", i, expected.String(), actual.String())
		}
	}
}

func TestMakeUnversionedWarning(t *testing.T) {
	const fileName = "filename.md"
	var cases = []struct {
		linkToReleaseDoc bool
		expected         string
	}{
		{
			true,
			`
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
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/filename.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

`,
		},
		{
			false,
			`
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
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

`,
		},
	}
	for i, c := range cases {
		if e, a := c.expected, makeUnversionedWarning(fileName, c.linkToReleaseDoc).String(); e != a {
			t.Errorf("case[%d]: \nexpected %s\ngot %s", i, e, a)
		}
	}
}
