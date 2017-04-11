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

func TestAnalytics(t *testing.T) {
	b := beginMungeTag("GENERATED_ANALYTICS")
	e := endMungeTag("GENERATED_ANALYTICS")
	var cases = []struct {
		in       string
		expected string
	}{
		{
			"aoeu",
			"aoeu" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n"},
		{
			"aoeu" + "\n" + "\n" + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()",
			"aoeu" + "\n" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n"},
		{
			"aoeu" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n",
			"aoeu" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n"},
		{
			"aoeu" + "\n" + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n",
			"aoeu" + "\n" + "\n" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n"},
		{
			"prefix" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e +
				"\n" + "suffix",
			"prefix" + "\n" + "suffix" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n"},
		{
			"aoeu" + "\n" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n",
			"aoeu" + "\n" + "\n" + "\n" +
				b + "\n" +
				"[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/path/to/file-name.md?pixel)]()" + "\n" +
				e + "\n"},
	}
	for i, c := range cases {
		in := getMungeLines(c.in)
		expected := getMungeLines(c.expected)
		out, err := updateAnalytics("path/to/file-name.md", in)
		assert.NoError(t, err)
		if !expected.Equal(out) {
			t.Errorf("Case %d Expected \n\n%v\n\n but got \n\n%v\n\n", i, expected.String(), out.String())
		}
	}
}
