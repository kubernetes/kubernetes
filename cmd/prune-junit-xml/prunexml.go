/*
Copyright 2022 The Kubernetes Authors.

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
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"

	"k8s.io/kubernetes/cmd/prune-junit-xml/logparse"
	"k8s.io/kubernetes/third_party/forked/gotestsum/junitxml"
)

func main() {
	maxTextSize := flag.Int("max-text-size", 1, "maximum size of attribute or text (in MB)")
	pruneTests := flag.Bool("prune-tests", true,
		"prune's xml files to display only top level tests and failed sub-tests")
	flag.Parse()
	for _, path := range flag.Args() {
		fmt.Printf("processing junit xml file : %s\n", path)
		xmlReader, err := os.Open(path)
		if err != nil {
			panic(err)
		}
		defer xmlReader.Close()
		suites, err := fetchXML(xmlReader) // convert MB into bytes (roughly!)
		if err != nil {
			panic(err)
		}

		pruneXML(suites, *maxTextSize*1e6) // convert MB into bytes (roughly!)
		if *pruneTests {
			pruneTESTS(suites)
		}

		xmlWriter, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			panic(err)
		}
		defer xmlWriter.Close()
		err = streamXML(xmlWriter, suites)
		if err != nil {
			panic(err)
		}
		fmt.Println("done.")
	}
}

func pruneXML(suites *junitxml.JUnitTestSuites, maxBytes int) {
	for _, suite := range suites.Suites {
		for i := range suite.TestCases {
			// Modify directly in the TestCases slice, if necessary.
			testcase := &suite.TestCases[i]
			if testcase.SkipMessage != nil {
				pruneStringIfNeeded(&testcase.SkipMessage.Message, maxBytes, "clipping skip message in test case : %s\n", testcase.Name)
			}
			if testcase.Failure != nil {
				// In Go unit tests, the entire test output
				// becomes the failure message because `go
				// test` doesn't track why a test fails. This
				// can make the failure message pretty large.
				//
				// We cannot identify the real failure here
				// either because Kubernetes has no convention
				// for how to format test failures. What we can
				// do is recognize log output added by klog.
				//
				// Therefore here we move the full text to
				// to the test output and only keep those
				// lines in the failure which are not from
				// klog.
				if testcase.SystemOut == "" {
					var buf strings.Builder
					// Iterate over all the log entries and decide what to keep as failure message.
					for entry := range logparse.All(strings.NewReader(testcase.Failure.Contents)) {
						if _, ok := entry.(*logparse.KlogEntry); ok {
							continue
						}
						_, _ = buf.WriteString(entry.LogData())
					}
					if buf.Len() < len(testcase.Failure.Contents) {
						// Update both strings because they became different.
						testcase.SystemOut = testcase.Failure.Contents
						pruneStringIfNeeded(&testcase.SystemOut, maxBytes, "clipping log output in test case: %s\n", testcase.Name)
						testcase.Failure.Contents = buf.String()
					}
				}
				pruneStringIfNeeded(&testcase.Failure.Contents, maxBytes, "clipping failure message in test case : %s\n", testcase.Name)
			}
		}
	}
}

func pruneStringIfNeeded(str *string, maxBytes int, msg string, args ...any) {
	if len(*str) <= maxBytes {
		return
	}
	fmt.Printf(msg, args...)
	head := (*str)[:maxBytes/2]
	tail := (*str)[len(*str)-maxBytes/2:]
	*str = head + "[...clipped...]" + tail
}

// This function condenses the junit xml to have package name as top level identifier
// and nesting under that.
func pruneTESTS(suites *junitxml.JUnitTestSuites) {
	var updatedTestsuites []junitxml.JUnitTestSuite

	for _, suite := range suites.Suites {
		var updatedTestcases []junitxml.JUnitTestCase
		var updatedTestcase junitxml.JUnitTestCase
		var updatedTestcaseFailure junitxml.JUnitFailure
		failflag := false
		name := suite.Name
		regex := regexp.MustCompile(`^(.*?)/([^/]+)/?$`)
		match := regex.FindStringSubmatch(name)
		updatedTestcase.Classname = match[1]
		updatedTestcase.Name = match[2]
		updatedTestcase.Time = suite.Time
		updatedSystemOut := ""
		updatedSystemErr := ""
		for _, testcase := range suite.TestCases {
			// The top level testcase element in a JUnit xml file does not have the / character.
			if testcase.Failure != nil {
				failflag = true
				updatedTestcaseFailure.Message = joinTexts(updatedTestcaseFailure.Message, testcase.Failure.Message)
				updatedTestcaseFailure.Contents = joinTexts(updatedTestcaseFailure.Contents, testcase.Failure.Contents)
				updatedTestcaseFailure.Type = joinTexts(updatedTestcaseFailure.Type, testcase.Failure.Type)
				updatedSystemOut = joinTexts(updatedSystemOut, testcase.SystemOut)
				updatedSystemErr = joinTexts(updatedSystemErr, testcase.SystemErr)
			}
		}
		if failflag {
			updatedTestcase.Failure = &updatedTestcaseFailure
			updatedTestcase.SystemOut = updatedSystemOut
			updatedTestcase.SystemErr = updatedSystemErr
		}
		suite.TestCases = append(updatedTestcases, updatedTestcase)
		updatedTestsuites = append(updatedTestsuites, suite)
	}
	suites.Suites = updatedTestsuites
}

// joinTexts returns "<a><empty line><b>" if both are non-empty,
// otherwise just the non-empty string, if there is one.
//
// If <b> is contained completely in <a>, <a> gets returned because repeating
// exactly the same string again doesn't add any information. Typically
// this occurs when joining the failure message because that is the fixed
// string "Failed" for all tests, regardless of what the test logged.
// The test log output is typically different because it cointains "=== RUN
// <test name>" and thus doesn't get dropped.
func joinTexts(a, b string) string {
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	if strings.Contains(a, b) {
		return a
	}
	sep := "\n"
	if !strings.HasSuffix(a, "\n") {
		sep = "\n\n"
	}
	return a + sep + b
}

func fetchXML(xmlReader io.Reader) (*junitxml.JUnitTestSuites, error) {
	decoder := xml.NewDecoder(xmlReader)
	var suites junitxml.JUnitTestSuites
	err := decoder.Decode(&suites)
	if err != nil {
		return nil, err
	}
	return &suites, nil
}

func streamXML(writer io.Writer, in *junitxml.JUnitTestSuites) error {
	_, err := writer.Write([]byte("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"))
	if err != nil {
		return err
	}
	encoder := xml.NewEncoder(writer)
	encoder.Indent("", "\t")
	err = encoder.Encode(in)
	if err != nil {
		return err
	}
	return encoder.Flush()
}
