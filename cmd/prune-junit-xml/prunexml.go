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
	"io"
	"os"
)

// JUnitTestSuites is a collection of JUnit test suites.
type JUnitTestSuites struct {
	XMLName xml.Name         `xml:"testsuites"`
	Suites  []JUnitTestSuite `xml:"testsuite"`
}

// JUnitTestSuite is a single JUnit test suite which may contain many
// testcases.
type JUnitTestSuite struct {
	XMLName    xml.Name        `xml:"testsuite"`
	Tests      int             `xml:"tests,attr"`
	Failures   int             `xml:"failures,attr"`
	Time       string          `xml:"time,attr"`
	Name       string          `xml:"name,attr"`
	Properties []JUnitProperty `xml:"properties>property,omitempty"`
	TestCases  []JUnitTestCase `xml:"testcase"`
	Timestamp  string          `xml:"timestamp,attr"`
}

// JUnitTestCase is a single test case with its result.
type JUnitTestCase struct {
	XMLName     xml.Name          `xml:"testcase"`
	Classname   string            `xml:"classname,attr"`
	Name        string            `xml:"name,attr"`
	Time        string            `xml:"time,attr"`
	SkipMessage *JUnitSkipMessage `xml:"skipped,omitempty"`
	Failure     *JUnitFailure     `xml:"failure,omitempty"`
}

// JUnitSkipMessage contains the reason why a testcase was skipped.
type JUnitSkipMessage struct {
	Message string `xml:"message,attr"`
}

// JUnitProperty represents a key/value pair used to define properties.
type JUnitProperty struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

// JUnitFailure contains data related to a failed test.
type JUnitFailure struct {
	Message  string `xml:"message,attr"`
	Type     string `xml:"type,attr"`
	Contents string `xml:",chardata"`
}

func main() {
	maxTextSize := flag.Int("max-text-size", 1, "maximum size of attribute or text (in MB)")
	flag.Parse()

	if flag.NArg() > 0 {
		for _, path := range flag.Args() {
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

			xmlWriter, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
			if err != nil {
				panic(err)
			}
			defer xmlWriter.Close()
			err = streamXML(xmlWriter, suites)
			if err != nil {
				panic(err)
			}
		}
	}
}

func pruneXML(suites *JUnitTestSuites, maxBytes int) {
	for _, suite := range suites.Suites {
		for _, testcase := range suite.TestCases {
			if testcase.SkipMessage != nil {
				if len(testcase.SkipMessage.Message) > maxBytes {
					testcase.SkipMessage.Message = "[... clipped...]" +
						testcase.SkipMessage.Message[len(testcase.SkipMessage.Message)-maxBytes:]
				}
			}
			if testcase.Failure != nil {
				if len(testcase.Failure.Contents) > maxBytes {
					testcase.Failure.Contents = "[... clipped...]" +
						testcase.Failure.Contents[len(testcase.Failure.Contents)-maxBytes:]
				}
			}
		}
	}
}

func fetchXML(xmlReader io.Reader) (*JUnitTestSuites, error) {
	decoder := xml.NewDecoder(xmlReader)
	var suites JUnitTestSuites
	err := decoder.Decode(&suites)
	if err != nil {
		return nil, err
	}
	return &suites, nil
}

func streamXML(writer io.Writer, in *JUnitTestSuites) error {
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
