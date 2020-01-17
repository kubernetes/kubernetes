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

// verify that dependencies are up-to-date across different files
package main

import (
	"flag"
	"log"
	"strings"

	"bufio"
	"io/ioutil"
	"os"
	"regexp"

	"gopkg.in/yaml.v2"
)

type dependencies struct {
	Dependencies []*dependency `yaml:"dependencies"`
}

type dependency struct {
	Name     string     `yaml:"name"`
	Version  string     `yaml:"version"`
	RefPaths []*refPath `yaml:"refPaths"`
}

type refPath struct {
	Path  string `yaml:"path"`
	Match string `yaml:"match"`
}

func main() {

	flag.Parse()

	args := flag.Args()

	if len(args) == 0 {
		log.Fatalf("usage: verifydependency <file>")
	}
	externalDepsFilePath := args[0]
	externalDepsFile, err := ioutil.ReadFile(externalDepsFilePath)
	if err != nil {
		panic(err)
	}

	mismatchErrorMessage := "ERROR: %v indicates that %v should be at version %v, but the following files didn't match:\n\n" +
		"%v\n\nif you are changing the version of %v, make sure all of the following files are updated with the newest version including %v\n" +
		"then run ./hack/verify-external-dependencies-version.sh\n\n"
	externalDeps := &dependencies{}
	var pathsToUpdate []string
	err = yaml.Unmarshal(externalDepsFile, externalDeps)
	if err != nil {
		panic(err)
	}

	for _, dep := range externalDeps.Dependencies {
		for _, refPath := range dep.RefPaths {
			file, err := os.Open(refPath.Path)
			if err != nil {
				log.Fatalf("error opening file %v : %v", refPath.Path, err)
			}
			matcher := regexp.MustCompile(refPath.Match)
			depFileScanner := bufio.NewScanner(file)
			var found bool
			for depFileScanner.Scan() {
				line := depFileScanner.Text()
				if matcher.MatchString(line) && strings.Contains(line, dep.Version) {
					found = true
					break
				}
			}
			if !found {
				pathsToUpdate = append(pathsToUpdate, refPath.Path)
			}
		}
		if len(pathsToUpdate) > 0 {
			log.Fatalf(mismatchErrorMessage, externalDepsFilePath, dep.Name, dep.Version, strings.Join(pathsToUpdate, "\n"), dep.Name, externalDepsFilePath)
		}
	}

}
