/*
Copyright 2021 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"reflect"
	"strings"
)

type Unwanted struct {
	// things we want to stop referencing
	Spec UnwantedSpec `json:"spec"`
	// status of our unwanted dependencies
	Status UnwantedStatus `json:"status"`
}

type UnwantedSpec struct {
	// TODO implement checks for RootModules
	// module names/patterns of root modules whose dependencies should be considered direct references
	RootModules []string `json:"rootModules"`

	// module names we don't want to depend on, mapped to an optional message about why
	UnwantedModules map[string]string `json:"unwantedModules"`
}

type UnwantedStatus struct {
	// TODO implement checks for Vendored
	// unwanted modules we still vendor, based on vendor/modules.txt content.
	// eliminating things from this list is good.
	Vendored []string `json:"vendored"`

	// TODO implement checks for add DirectReferences
	// unwanted modules we still directly reference from spec.roots, based on `go mod graph` content.
	// eliminating things from this list is good.
	DirectReferences []string `json:"directReferences"`

	// unwanted modules indirectly referenced from modules other than spec.roots, based on `go mod graph` content.
	// eliminating things from this list is good, but usually requires working with upstreams to do so.
	IndirectReferences []string `json:"indirectReferences"`
}

// Check all unwanted dependencies and update its status.
func (config *Unwanted) checkUpdateStatus(modeGraph map[string][]string) {
	fmt.Println("Check all unwanted dependencies and update its status.")
	for unwanted := range config.Spec.UnwantedModules {
		if _, found := modeGraph[unwanted]; found {
			if !stringInSlice(unwanted, config.Status.IndirectReferences) {
				config.Status.IndirectReferences = append(config.Status.IndirectReferences, unwanted)
			}
		}
	}
}

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	output, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}

func readFile(path string) (string, error) {
	content, err := ioutil.ReadFile(path)
	// Convert []byte to string and print to screen
	return string(content), err
}

func stringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

func convertToMap(modStr string) map[string][]string {
	modMap := make(map[string][]string)
	for _, line := range strings.Split(modStr, "\n") {
		if len(line) == 0 {
			continue
		}
		deps := strings.Split(line, " ")
		if len(deps) == 2 {
			first := strings.Split(deps[0], "@")[0]
			second := strings.Split(deps[1], "@")[0]
			original, ok := modMap[first]
			if !ok {
				modMap[first] = []string{second}
			} else if stringInSlice(second, original) {
				continue
			} else {
				modMap[first] = append(original, second)
			}
		} else {
			// skip invalid line
			log.Printf("!!!invalid line in mod.graph: %s", line)
			continue
		}
	}
	return modMap
}

// option1: dependencyverifier dependencies.json
// it will run `go mod graph` and check it.
// option2: dependencyverifier dependencies.json mod.graph
// it will check the specified mod graph result file.
func main() {
	var modeGraphStr string
	var err error
	if len(os.Args) == 2 {
		// run `go mod graph`
		modeGraphStr, err = runCommand("go", "mod", "graph")
		if err != nil {
			log.Fatalf("Error running 'go mod graph': %s", err)
		}
	} else if len(os.Args) == 3 {
		modGraphFile := string(os.Args[2])
		modeGraphStr, err = readFile(modGraphFile)
		// read file, such as `mod.graph`
		if err != nil {
			log.Fatalf("Error reading mod file %s: %s", modGraphFile, err)
		}
	} else {
		log.Fatalf("Usage: %s dependencies.json {mod.graph}", os.Args[0])
	}

	dependenciesJSONPath := string(os.Args[1])
	dependencies, err := readFile(dependenciesJSONPath)
	if err != nil {
		log.Fatalf("Error reading dependencies file %s: %s", dependencies, err)
	}

	// load Unwanted from json
	configFromFile := &Unwanted{}
	decoder := json.NewDecoder(bytes.NewBuffer([]byte(dependencies)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(configFromFile); err != nil {
		log.Fatalf("Error reading dependencies file %s: %s", dependenciesJSONPath, err)
	}

	// Check and update status of struct Unwanted
	modeGraph := convertToMap(modeGraphStr)
	config := &Unwanted{}
	config.Spec.UnwantedModules = configFromFile.Spec.UnwantedModules
	config.checkUpdateStatus(modeGraph)

	// DeepEqual check current status with it in json file
	// TODO sort array
	if !reflect.DeepEqual(configFromFile.Status, config.Status) {
		fmt.Println("Status in ./hack/unwanted-dependencies.json:\n", configFromFile.Status)
		fmt.Println("Status detected:\n", config.Status)
		log.Println("!!! Please update ./hack/unwanted-dependencies.json")
		os.Exit(1)
	}
}
