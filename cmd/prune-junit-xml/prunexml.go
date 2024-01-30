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
		for _, testcase := range suite.TestCases {
			if testcase.SkipMessage != nil {
				if len(testcase.SkipMessage.Message) > maxBytes {
					fmt.Printf("clipping skip message in test case : %s\n", testcase.Name)
					head := testcase.SkipMessage.Message[:maxBytes/2]
					tail := testcase.SkipMessage.Message[len(testcase.SkipMessage.Message)-maxBytes/2:]
					testcase.SkipMessage.Message = head + "[...clipped...]" + tail
				}
			}
			if testcase.Failure != nil {
				if len(testcase.Failure.Contents) > maxBytes {
					fmt.Printf("clipping failure message in test case : %s\n", testcase.Name)
					head := testcase.Failure.Contents[:maxBytes/2]
					tail := testcase.Failure.Contents[len(testcase.Failure.Contents)-maxBytes/2:]
					testcase.Failure.Contents = head + "[...clipped...]" + tail
				}
			}
		}
	}
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
		for _, testcase := range suite.TestCases {
			// The top level testcase element in a JUnit xml file does not have the / character.
			if testcase.Failure != nil {
				failflag = true
				updatedTestcaseFailure.Message = updatedTestcaseFailure.Message + testcase.Failure.Message + ";"
				updatedTestcaseFailure.Contents = updatedTestcaseFailure.Contents + testcase.Failure.Contents + ";"
				updatedTestcaseFailure.Type = updatedTestcaseFailure.Type + testcase.Failure.Type
			}
		}
		if failflag {
			updatedTestcase.Failure = &updatedTestcaseFailure
		}
		suite.TestCases = append(updatedTestcases, updatedTestcase)
		updatedTestsuites = append(updatedTestsuites, suite)
	}
	suites.Suites = updatedTestsuites
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
