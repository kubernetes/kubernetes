/*
Copyright 2019 The Kubernetes Authors.

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
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"text/template"

	"gopkg.in/yaml.v2"

	"github.com/onsi/ginkgo/v2/types"
)

// ConformanceData describes the structure of the conformance.yaml file
type ConformanceData struct {
	// A URL to the line of code in the kube src repo for the test. Omitted from the YAML to avoid exposing line number.
	URL string `yaml:"-"`
	// Extracted from the "Testname:" comment before the test
	TestName string
	// CodeName is taken from the actual ginkgo descriptions, e.g. `[sig-apps] Foo should bar [Conformance]`
	CodeName string
	// Extracted from the "Description:" comment before the test
	Description string
	// Version when this test is added or modified ex: v1.12, v1.13
	Release string
	// File is the filename where the test is defined. We intentionally don't save the line here to avoid meaningless changes.
	File string
}

var (
	baseURL = flag.String("url", "https://github.com/kubernetes/kubernetes/tree/master/", "location of the current source")
	k8sPath = flag.String("source", "", "location of the current source on the current machine")
	confDoc = flag.Bool("docs", false, "write a conformance document")
	version = flag.String("version", "v1.9", "version of this conformance document")

	// If a test name contains any of these tags, it is ineligble for promotion to conformance
	regexIneligibleTags = regexp.MustCompile(`\[(Alpha|Feature:[^\]]+|Flaky)\]`)

	// Conformance comments should be within this number of lines to the call itself.
	// Allowing for more than one in case a spare comment or two is below it.
	conformanceCommentsLineWindow = 5

	seenLines map[string]struct{}
)

type frame struct {
	// File and Line are the file name and line number of the
	// location in this frame. For non-leaf frames, this will be
	// the location of a call. These may be the empty string and
	// zero, respectively, if not known.
	File string
	Line int
}

func main() {
	flag.Parse()

	if len(flag.Args()) < 1 {
		log.Fatalln("Requires the name of the test details file as first and only argument.")
	}
	testDetailsFile := flag.Args()[0]
	f, err := os.Open(testDetailsFile)
	if err != nil {
		log.Fatalf("Failed to open file %v: %v", testDetailsFile, err)
	}
	defer f.Close()

	seenLines = map[string]struct{}{}
	dec := json.NewDecoder(f)
	testInfos := []*ConformanceData{}
	for {
		var spec *types.SpecReport
		if err := dec.Decode(&spec); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		if isConformance(spec) {
			testInfo := getTestInfo(spec)
			if testInfo != nil {
				testInfos = append(testInfos, testInfo)
				if err := validateTestName(testInfo.CodeName); err != nil {
					log.Fatal(err)
				}
			}
		}
	}

	sort.Slice(testInfos, func(i, j int) bool { return testInfos[i].CodeName < testInfos[j].CodeName })
	saveAllTestInfo(testInfos)
}

func isConformance(spec *types.SpecReport) bool {
	return strings.Contains(getTestName(spec), "[Conformance]")
}

func getTestInfo(spec *types.SpecReport) *ConformanceData {
	var c *ConformanceData
	var err error
	// The key to this working is that we don't need to parse every file or walk
	// every types.CodeLocation. The LeafNodeLocation is going to be file:line which
	// attached to the comment that we want.
	leafNodeLocation := spec.LeafNodeLocation
	frame := frame{
		File: leafNodeLocation.FileName,
		Line: leafNodeLocation.LineNumber,
	}
	c, err = getConformanceData(frame)
	if err != nil {
		log.Printf("Error looking for conformance data: %v", err)
	}
	if c == nil {
		log.Printf("Did not find test info for spec: %#v\n", getTestName(spec))
		return nil
	}
	c.CodeName = getTestName(spec)
	return c
}

func getTestName(spec *types.SpecReport) string {
	return strings.Join(spec.ContainerHierarchyTexts[0:], " ") + " " + spec.LeafNodeText
}

func saveAllTestInfo(dataSet []*ConformanceData) {
	if *confDoc {
		// Note: this assumes that you're running from the root of the kube src repo
		templ, err := template.ParseFiles("./test/conformance/cf_header.md")
		if err != nil {
			fmt.Printf("Error reading the Header file information: %s\n\n", err)
		}
		data := struct {
			Version string
		}{
			Version: *version,
		}
		templ.Execute(os.Stdout, data)

		for _, data := range dataSet {
			fmt.Printf("## [%s](%s)\n\n", data.TestName, data.URL)
			fmt.Printf("- Added to conformance in release %s\n", data.Release)
			fmt.Printf("- Defined in code as: %s\n\n", data.CodeName)
			fmt.Printf("%s\n\n", data.Description)
		}
		return
	}

	// Serialize the list as a whole. Generally meant to end up as conformance.txt which tracks the set of tests.
	b, err := yaml.Marshal(dataSet)
	if err != nil {
		log.Printf("Error marshalling data into YAML: %v", err)
	}
	fmt.Println(string(b))
}

func getConformanceData(targetFrame frame) (*ConformanceData, error) {
	// filenames are in one of two special GOPATHs depending on if they were
	// built dockerized or with the host go
	// we want to trim this prefix to produce portable relative paths
	k8sSRC := *k8sPath + "/_output/local/go/src/k8s.io/kubernetes/"
	trimmedFile := strings.TrimPrefix(targetFrame.File, k8sSRC)
	trimmedFile = strings.TrimPrefix(trimmedFile, "/go/src/k8s.io/kubernetes/_output/dockerized/go/src/k8s.io/kubernetes/")
	targetFrame.File = trimmedFile

	freader, err := os.Open(targetFrame.File)
	if err != nil {
		return nil, err
	}
	defer freader.Close()

	cd, err := scanFileForFrame(targetFrame.File, freader, targetFrame)
	if err != nil {
		return nil, err
	}
	if cd != nil {
		return cd, nil
	}

	return nil, nil
}

// scanFileForFrame will scan the target and look for a conformance comment attached to the function
// described by the target frame. If the comment can't be found then nil, nil is returned.
func scanFileForFrame(filename string, src interface{}, targetFrame frame) (*ConformanceData, error) {
	fset := token.NewFileSet() // positions are relative to fset
	f, err := parser.ParseFile(fset, filename, src, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	cmap := ast.NewCommentMap(fset, f, f.Comments)
	for _, cs := range cmap {
		for _, c := range cs {
			if cd := tryCommentGroupAndFrame(fset, c, targetFrame); cd != nil {
				return cd, nil
			}
		}
	}
	return nil, nil
}

func validateTestName(s string) error {
	matches := regexIneligibleTags.FindAllString(s, -1)
	if matches != nil {
		return fmt.Errorf("'%s' cannot have invalid tags %v", s, strings.Join(matches, ","))
	}
	return nil
}

func tryCommentGroupAndFrame(fset *token.FileSet, cg *ast.CommentGroup, f frame) *ConformanceData {
	if !shouldProcessCommentGroup(fset, cg, f) {
		return nil
	}

	// Each file/line will either be some helper function (not a conformance comment) or apply to just a single test. Don't revisit.
	if seenLines != nil {
		seenLines[fmt.Sprintf("%v:%v", f.File, f.Line)] = struct{}{}
	}
	cd := commentToConformanceData(cg.Text())
	if cd == nil {
		return nil
	}

	cd.URL = fmt.Sprintf("%s%s#L%d", *baseURL, f.File, f.Line)
	cd.File = f.File
	return cd
}

func shouldProcessCommentGroup(fset *token.FileSet, cg *ast.CommentGroup, f frame) bool {
	lineDiff := f.Line - fset.Position(cg.End()).Line
	return lineDiff > 0 && lineDiff <= conformanceCommentsLineWindow
}

func commentToConformanceData(comment string) *ConformanceData {
	lines := strings.Split(comment, "\n")
	descLines := []string{}
	cd := &ConformanceData{}
	var curLine string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if sline := regexp.MustCompile("^Testname\\s*:\\s*").Split(line, -1); len(sline) == 2 {
			curLine = "Testname"
			cd.TestName = sline[1]
			continue
		}
		if sline := regexp.MustCompile("^Release\\s*:\\s*").Split(line, -1); len(sline) == 2 {
			curLine = "Release"
			cd.Release = sline[1]
			continue
		}
		if sline := regexp.MustCompile("^Description\\s*:\\s*").Split(line, -1); len(sline) == 2 {
			curLine = "Description"
			descLines = append(descLines, sline[1])
			continue
		}

		// Line has no header
		if curLine == "Description" {
			descLines = append(descLines, line)
		}
	}
	if cd.TestName == "" {
		return nil
	}

	cd.Description = strings.Join(descLines, " ")
	return cd
}
