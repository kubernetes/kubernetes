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
	"bytes"
	"fmt"
	"os"
	"regexp"
)

var (
	beginMungeExp = regexp.QuoteMeta(beginMungeTag("GENERATED_ANALYTICS"))
	endMungeExp   = regexp.QuoteMeta(endMungeTag("GENERATED_ANALYTICS"))
	analyticsExp  = regexp.QuoteMeta("[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/") +
		"[^?]*" +
		regexp.QuoteMeta("?pixel)]()")

	// Matches the analytics blurb, with or without the munge headers.
	analyticsRE = regexp.MustCompile(`[\n]*` + analyticsExp + `[\n]?` +
		`|` + `[\n]*` + beginMungeExp + `[^<]*` + endMungeExp)
)

// This adds the analytics link to every .md file.
func checkAnalytics(fileName string, fileBytes []byte) (output []byte, err error) {
	fileName = makeRepoRelative(fileName)
	desired := fmt.Sprintf(`


`+beginMungeTag("GENERATED_ANALYTICS")+`
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/%s?pixel)]()
`+endMungeTag("GENERATED_ANALYTICS")+`
`, fileName)
	if !analyticsRE.MatchString(desired) {
		fmt.Printf("%q does not match %q", analyticsRE.String(), desired)
		os.Exit(1)
	}
	//output = replaceNonPreformattedRegexp(fileBytes, analyticsRE, func(in []byte) []byte {
	output = analyticsRE.ReplaceAllFunc(fileBytes, func(in []byte) []byte {
		return []byte{}
	})
	output = bytes.TrimRight(output, "\n")
	output = append(output, []byte(desired)...)
	return output, nil
}
