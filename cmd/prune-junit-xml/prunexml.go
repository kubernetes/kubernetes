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
	"bufio"
	"bytes"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strings"

	"k8s.io/kubernetes/cmd/prune-junit-xml/logparse"
	"k8s.io/kubernetes/third_party/forked/gotestsum/junitxml"
	"sigs.k8s.io/yaml"
)

func main() {
	maxTextSize := flag.Int("max-text-size", 1, "maximum size of attribute or text (in MB)")
	pruneTests := flag.Bool("prune-tests", true,
		"prune's xml files to display only top level tests and failed sub-tests")
	addOwners := flag.Bool("add-owners", true,
		"when pruning tests, also look for OWNERs files of the packages and prefix the names with [sig-...] if found")
	flag.Parse()

	pkgs := newPackageOwners(*addOwners)

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
			pruneTESTS(suites, pkgs)
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
func pruneTESTS(suites *junitxml.JUnitTestSuites, pkgs *packageOwners) {
	var updatedTestsuites []junitxml.JUnitTestSuite

	for _, suite := range suites.Suites {
		var updatedTestcases []junitxml.JUnitTestCase
		var updatedTestcase junitxml.JUnitTestCase
		var updatedTestcaseFailure junitxml.JUnitFailure
		failflag := false
		name := suite.Name

		// Inject the owning SIG prefix, if possible.
		// This has to be done while we still have what
		// is likely to be the full package name.
		name = pkgs.addOwner(name)

		regex := regexp.MustCompile(`^(.*?)/([^/]+)/?$`)
		match := regex.FindStringSubmatch(name)
		baseName := match[1]
		leafName := match[2]

		// testgrid uses suite.Name.
		// Spyglass/Prow use testcase.Classname.
		// Therefore we need to update both.
		suite.Name = baseName
		updatedTestcase.Classname = baseName
		updatedTestcase.Name = leafName
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

type packageOwners struct {
	// pkgs maps from import path to directory.
	pkgs map[string]string
}

func newPackageOwners(enabled bool) *packageOwners {
	if !enabled {
		return nil
	}

	return &packageOwners{
		pkgs: make(map[string]string),
	}
}

// addOwner takes a package name (= import path, like k8s.io/client-go),
// tries to look up the source code of the package, and then
// walks up until it finds an OWNERS file with some sig label.
// The first sig label found this way is used.
//
// If successful, that SIG gets added to the name.
// If not, the name remains unchanged.
func (p *packageOwners) addOwner(name string) string {
	if p == nil {
		return name
	}

	dir := p.pkgs[name]
	if dir == "" {
		// Look up via `go list`. To invoke it less often,
		// we also ask for sub-packages and cache the results.
		// We don't care about errors.
		//
		// This is roughly what golang.org/x/tools/go/packages does,
		// which we can't use here because it would add a new
		// dependency to k/k.
		cmd := exec.Command("go", "list", "-f", "{{.ImportPath}}:{{.Dir}}", name+"/...")
		out, _ := cmd.Output()
		scanner := bufio.NewScanner(bytes.NewReader(out))
		for scanner.Scan() {
			line := scanner.Text()
			parts := strings.SplitN(line, ":", 2)
			if len(parts) != 2 {
				continue
			}
			p.pkgs[parts[0]] = parts[1]
		}
		dir = p.pkgs[name]
	}

	// Walk up starting from an absolute path until we cannot go up further.
	for ; dir != "" && dir != "." && dir != "/"; dir = path.Dir(dir) {
		data, err := os.ReadFile(path.Join(dir, "OWNERS"))
		if err != nil {
			continue
		}
		var owners owners
		if err := yaml.Unmarshal(data, &owners); err != nil {
			continue
		}
		for _, label := range owners.Labels {
			if strings.HasPrefix(label, "sig/") {
				// Bingo!
				return fmt.Sprintf("[sig-%s] %s", label[4:], name)
			}
		}
	}

	return name
}

// owners contains only fields from https://go.k8s.io/owners that we care about.
type owners struct {
	Labels []string `json:"labels"`
}
