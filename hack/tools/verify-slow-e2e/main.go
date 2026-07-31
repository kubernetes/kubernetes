package main

import (
	"encoding/xml"
	"flag"
	"fmt"
	"os"
	"strings"
)

// TestSuites represents the root of a JUnit XML file
type TestSuites struct {
	Suites []TestSuite `xml:"testsuite"`
}

type TestSuite struct {
	TestCases []TestCase `xml:"testcase"`
}

type TestCase struct {
	Name    string   `xml:"name,attr"`
	Time    float64  `xml:"time,attr"`
	Skipped *Skipped `xml:"skipped"`
	Failure *Failure `xml:"failure"`
}

type Skipped struct{}
type Failure struct{}

const slowThresholdSeconds = 300.0

func main() {
	flag.Parse()
	files := flag.Args()
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "Usage: verify-slow-e2e <junit.xml> [junit2.xml...]")
		os.Exit(1)
	}

	exitCode := 0

	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading file %s: %v\n", file, err)
			exitCode = 1
			continue
		}

		var suites TestSuites
		if err := xml.Unmarshal(data, &suites); err != nil {
			// Some junit files don't have <testsuites> at the root, they start with <testsuite>
			var suite TestSuite
			if err2 := xml.Unmarshal(data, &suite); err2 != nil {
				fmt.Fprintf(os.Stderr, "Error parsing XML in %s: %v\n", file, err)
				exitCode = 1
				continue
			}
			suites.Suites = []TestSuite{suite}
		}

		for _, suite := range suites.Suites {
			for _, tc := range suite.TestCases {
				if tc.Skipped != nil {
					continue
				}

				hasSlow := strings.Contains(tc.Name, "[Slow]")
				isSlow := tc.Time > slowThresholdSeconds

				if isSlow && !hasSlow {
					fmt.Printf("MISSING [Slow] tag (took %.1fs): %s\n", tc.Time, tc.Name)
					exitCode = 1
				} else if !isSlow && hasSlow && tc.Time > 0 {
					// Add a buffer to avoid flapping tests. If it completes in less than 240 seconds
					// consistently but is tagged [Slow], it might be unnecessary.
					if tc.Time < 180.0 {
						fmt.Printf("UNNECESSARY [Slow] tag (took %.1fs): %s\n", tc.Time, tc.Name)
						exitCode = 1
					}
				}
			}
		}
	}

	os.Exit(exitCode)
}
