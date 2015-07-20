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

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUnversionedWarning(t *testing.T) {
	beginMark := beginMungeTag(unversionedWarningTag)
	endMark := endMungeTag(unversionedWarningTag)

	warningBlock := beginMark + "\n" + makeUnversionedWarning("filename.md") + "\n" + endMark + "\n"
	var cases = []struct {
		in  string
		out string
	}{
		{"", warningBlock},
		{
			"Foo\nBar\n",
			warningBlock + "Foo\nBar\n",
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
		actual, err := updateUnversionedWarning("filename.md", []byte(c.in))
		assert.NoError(t, err)
		if string(actual) != c.out {
			t.Errorf("case[%d]: expected %q got %q", i, c.out, string(actual))
		}
	}
}
