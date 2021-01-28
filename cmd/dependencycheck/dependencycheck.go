/*
Copyright 2020 The Kubernetes Authors.

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

// Checks for restricted dependencies in go packages. Does not check transitive
// dependencies implicitly, so they must be supplied in dependencies file if
// they are to be evaluated.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"regexp"
)

var (
	exclude  = flag.String("exclude", "", "skip packages regex pattern (e.g. '^k8s.io/kubernetes/')")
	restrict = flag.String("restrict", "", "restricted dependencies regex pattern (e.g. '^k8s.io/(apimachinery|client-go)/')")
)

type goPackage struct {
	Name         string
	ImportPath   string
	Imports      []string
	TestImports  []string
	XTestImports []string
}

func main() {
	flag.Parse()

	args := flag.Args()

	if len(args) != 1 {
		log.Fatalf("usage: dependencycheck <json-dep-file> (e.g. 'go list -mod=vendor -test -deps -json ./vendor/...')")
	}
	if *restrict == "" {
		log.Fatalf("Must specify restricted regex pattern")
	}
	depsPattern, err := regexp.Compile(*restrict)
	if err != nil {
		log.Fatalf("Error compiling restricted dependencies regex: %v", err)
	}
	var excludePattern *regexp.Regexp
	if *exclude != "" {
		excludePattern, err = regexp.Compile(*exclude)
		if err != nil {
			log.Fatalf("Error compiling excluded package regex: %v", err)
		}
	}
	b, err := ioutil.ReadFile(args[0])
	if err != nil {
		log.Fatalf("Error reading dependencies file: %v", err)
	}

	packages := []goPackage{}
	decoder := json.NewDecoder(bytes.NewBuffer(b))
	for {
		pkg := goPackage{}
		if err := decoder.Decode(&pkg); err != nil {
			if err == io.EOF {
				break
			}
			log.Fatalf("Error unmarshaling dependencies file: %v", err)
		}
		packages = append(packages, pkg)
	}

	violations := map[string][]string{}
	for _, p := range packages {
		if excludePattern != nil && excludePattern.MatchString(p.ImportPath) {
			continue
		}
		importViolations := []string{}
		allImports := []string{}
		allImports = append(allImports, p.Imports...)
		allImports = append(allImports, p.TestImports...)
		allImports = append(allImports, p.XTestImports...)
		for _, i := range allImports {
			if depsPattern.MatchString(i) {
				importViolations = append(importViolations, i)
			}
		}
		if len(importViolations) > 0 {
			violations[p.ImportPath] = importViolations
		}
	}

	if len(violations) > 0 {
		for k, v := range violations {
			fmt.Printf("Found dependency violations in package %s:\n", k)
			for _, a := range v {
				fmt.Println("--> " + a)
			}
		}
		log.Fatal("Found restricted dependency violations in packages")
	}
}
