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
	"sort"
	"strings"

	"github.com/google/go-cmp/cmp"
)

type Unwanted struct {
	// things we want to stop referencing
	Spec UnwantedSpec `json:"spec"`
	// status of our unwanted dependencies
	Status UnwantedStatus `json:"status"`
}

type UnwantedSpec struct {
	// module names we don't want to depend on, mapped to an optional message about why
	UnwantedModules map[string]string `json:"unwantedModules"`
}

type UnwantedStatus struct {
	// references to modules in the spec.unwantedModules list, based on `go mod graph` content.
	// eliminating things from this list is good, and sometimes requires working with upstreams to do so.
	UnwantedReferences map[string][]string `json:"unwantedReferences"`
	// list of modules in the spec.unwantedModules list which are vendored
	UnwantedVendored []string `json:"unwantedVendored"`
}

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	return runCommandInDir("", cmd)
}

func runCommandInDir(dir string, cmd []string) (string, error) {
	c := exec.Command(cmd[0], cmd[1:]...)
	c.Dir = dir
	output, err := c.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}

func readFile(path string) (string, error) {
	content, err := os.ReadFile(path)
	// Convert []byte to string and print to screen
	return string(content), err
}

func moduleInSlice(a module, list []module, matchVersion bool) bool {
	for _, b := range list {
		if b == a {
			return true
		}
		if !matchVersion && b.name == a.name {
			return true
		}
	}
	return false
}

// converts `go mod graph` output modStr into a map of from->[]to references and the main module
func convertToMap(modStr string) ([]module, map[module][]module) {
	var (
		mainModulesList = []module{}
		mainModules     = map[module]bool{}
	)
	modMap := make(map[module][]module)
	for _, line := range strings.Split(modStr, "\n") {
		if len(line) == 0 {
			continue
		}
		deps := strings.Split(line, " ")
		if len(deps) == 2 {
			first := parseModule(deps[0])
			second := parseModule(deps[1])
			if first.version == "" || first.version == "v0.0.0" {
				if !mainModules[first] {
					mainModules[first] = true
					mainModulesList = append(mainModulesList, first)
				}
			}
			modMap[first] = append(modMap[first], second)
		} else {
			// skip invalid line
			log.Printf("!!!invalid line in mod.graph: %s", line)
			continue
		}
	}
	return mainModulesList, modMap
}

// difference returns a-b and b-a as sorted lists
func difference(a, b []string) ([]string, []string) {
	aMinusB := map[string]bool{}
	bMinusA := map[string]bool{}
	for _, dependency := range a {
		aMinusB[dependency] = true
	}
	for _, dependency := range b {
		if _, found := aMinusB[dependency]; found {
			delete(aMinusB, dependency)
		} else {
			bMinusA[dependency] = true
		}
	}
	aMinusBList := []string{}
	bMinusAList := []string{}
	for dependency := range aMinusB {
		aMinusBList = append(aMinusBList, dependency)
	}
	for dependency := range bMinusA {
		bMinusAList = append(bMinusAList, dependency)
	}
	sort.Strings(aMinusBList)
	sort.Strings(bMinusAList)
	return aMinusBList, bMinusAList
}

type module struct {
	name    string
	version string
}

func (m module) String() string {
	if len(m.version) == 0 {
		return m.name
	}
	return m.name + "@" + m.version
}

func parseModule(s string) module {
	if !strings.Contains(s, "@") {
		return module{name: s}
	}
	parts := strings.SplitN(s, "@", 2)
	return module{name: parts[0], version: parts[1]}
}

// option1: dependencyverifier dependencies.json
// it will run `go mod graph` and check it.
func main() {
	var modeGraphStr string
	var err error
	if len(os.Args) == 2 {
		// run `go mod graph`
		modeGraphStr, err = runCommand("go", "mod", "graph")
		if err != nil {
			log.Fatalf("Error running 'go mod graph': %s", err)
		}
	} else {
		log.Fatalf("Usage: %s dependencies.json", os.Args[0])
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

	// convert from `go mod graph` to main module and map of from->[]to references
	mainModules, moduleGraph := convertToMap(modeGraphStr)

	directDependencies := map[string]map[string]bool{}
	for _, mainModule := range mainModules {
		dir := ""
		if mainModule.name != "k8s.io/kubernetes" {
			dir = "staging/src/" + mainModule.name
		}
		listOutput, err := runCommandInDir(dir, []string{"go", "list", "-m", "-f", "{{if not .Indirect}}{{if not .Main}}{{.Path}}{{end}}{{end}}", "all"})
		if err != nil {
			log.Fatalf("Error running 'go list' for %s: %s", mainModule.name, err)
		}
		directDependencies[mainModule.name] = map[string]bool{}
		for _, directDependency := range strings.Split(listOutput, "\n") {
			directDependencies[mainModule.name][directDependency] = true
		}
	}

	// gather the effective versions by looking at the versions required by the main modules
	effectiveVersions := map[string]module{}
	for _, mainModule := range mainModules {
		for _, override := range moduleGraph[mainModule] {
			if _, ok := effectiveVersions[override.name]; !ok {
				effectiveVersions[override.name] = override
			}
		}
	}

	unwantedToReferencers := map[string][]module{}
	for _, mainModule := range mainModules {
		// visit to find unwanted modules still referenced from the main module
		visit(func(m module, via []module) {
			if _, unwanted := configFromFile.Spec.UnwantedModules[m.name]; unwanted {
				// this is unwanted, store what is referencing it
				referencer := via[len(via)-1]
				if !moduleInSlice(referencer, unwantedToReferencers[m.name], false) {
					// // uncomment to get a detailed tree of the path that referenced the unwanted dependency
					//
					// i := 0
					// for _, v := range via {
					// 	if v.version != "" && v.version != "v0.0.0" {
					// 		fmt.Println(strings.Repeat("  ", i), v)
					// 		i++
					// 	}
					// }
					// if i > 0 {
					// 	fmt.Println(strings.Repeat("  ", i+1), m)
					// 	fmt.Println()
					// }
					unwantedToReferencers[m.name] = append(unwantedToReferencers[m.name], referencer)
				}
			}
		}, mainModule, moduleGraph, effectiveVersions)
	}

	config := &Unwanted{}
	config.Spec.UnwantedModules = configFromFile.Spec.UnwantedModules
	for unwanted := range unwantedToReferencers {
		if config.Status.UnwantedReferences == nil {
			config.Status.UnwantedReferences = map[string][]string{}
		}
		sort.Slice(unwantedToReferencers[unwanted], func(i, j int) bool {
			ri := unwantedToReferencers[unwanted][i]
			rj := unwantedToReferencers[unwanted][j]
			if ri.name != rj.name {
				return ri.name < rj.name
			}
			return ri.version < rj.version
		})
		for _, referencer := range unwantedToReferencers[unwanted] {
			// make sure any reference at all shows up as a non-nil status
			if config.Status.UnwantedReferences == nil {
				config.Status.UnwantedReferences[unwanted] = []string{}
			}
			// record specific names of versioned referents
			if referencer.version != "" && referencer.version != "v0.0.0" {
				config.Status.UnwantedReferences[unwanted] = append(config.Status.UnwantedReferences[unwanted], referencer.name)
			} else if directDependencies[referencer.name][unwanted] {
				config.Status.UnwantedReferences[unwanted] = append(config.Status.UnwantedReferences[unwanted], referencer.name)
			}
		}
	}

	vendorModulesTxt, err := ioutil.ReadFile("vendor/modules.txt")
	if err != nil {
		log.Fatal(err)
	}
	vendoredModules := map[string]bool{}
	for _, l := range strings.Split(string(vendorModulesTxt), "\n") {
		parts := strings.Split(l, " ")
		if len(parts) == 3 && parts[0] == "#" && strings.HasPrefix(parts[2], "v") {
			vendoredModules[parts[1]] = true
		}
	}
	config.Status.UnwantedVendored = []string{}
	for unwanted := range configFromFile.Spec.UnwantedModules {
		if vendoredModules[unwanted] {
			config.Status.UnwantedVendored = append(config.Status.UnwantedVendored, unwanted)
		}
	}
	sort.Strings(config.Status.UnwantedVendored)

	needUpdate := false

	// Compare unwanted list from unwanted-dependencies.json with current status from `go mod graph`
	expected, err := json.MarshalIndent(configFromFile.Status, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	actual, err := json.MarshalIndent(config.Status, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if !bytes.Equal(expected, actual) {
		log.Printf("Expected status of\n%s", string(expected))
		log.Printf("Got status of\n%s", string(actual))
		needUpdate = true
		log.Print("Status diff:\n", cmp.Diff(expected, actual))
	}
	for expectedRef, expectedFrom := range configFromFile.Status.UnwantedReferences {
		actualFrom, ok := config.Status.UnwantedReferences[expectedRef]
		if !ok {
			// disappeared entirely
			log.Printf("Good news! Unwanted dependency %q is no longer referenced. Remove status.unwantedReferences[%q] in %s to ensure it doesn't get reintroduced.", expectedRef, expectedRef, dependenciesJSONPath)
			needUpdate = true
			continue
		}
		removedReferences, unwantedReferences := difference(expectedFrom, actualFrom)
		if len(removedReferences) > 0 {
			log.Printf("Good news! Unwanted module %q dropped the following dependants:", expectedRef)
			for _, reference := range removedReferences {
				log.Printf("   %s", reference)
			}
			log.Printf("!!! Remove those from status.unwantedReferences[%q] in %s to ensure they don't get reintroduced.", expectedRef, dependenciesJSONPath)
			needUpdate = true
		}
		if len(unwantedReferences) > 0 {
			log.Printf("Unwanted module %q marked in %s is referenced by new dependants:", expectedRef, dependenciesJSONPath)
			for _, reference := range unwantedReferences {
				log.Printf("   %s", reference)
			}
			log.Printf("!!! Avoid updating referencing modules to versions that reintroduce use of unwanted dependencies\n")
			needUpdate = true
		}
	}
	for actualRef, actualFrom := range config.Status.UnwantedReferences {
		if _, expected := configFromFile.Status.UnwantedReferences[actualRef]; expected {
			// expected, already ensured referencers were equal in the first loop
			continue
		}
		log.Printf("Unwanted module %q marked in %s is referenced", actualRef, dependenciesJSONPath)
		for _, reference := range actualFrom {
			log.Printf("   %s", reference)
		}
		log.Printf("!!! Avoid updating referencing modules to versions that reintroduce use of unwanted dependencies\n")
		needUpdate = true
	}

	removedVendored, addedVendored := difference(configFromFile.Status.UnwantedVendored, config.Status.UnwantedVendored)
	if len(removedVendored) > 0 {
		log.Printf("Good news! Unwanted modules are no longer vendered: %q", removedVendored)
		log.Printf("!!! Remove those from status.unwantedVendored in %s to ensure they don't get reintroduced.", dependenciesJSONPath)
		needUpdate = true
	}
	if len(addedVendored) > 0 {
		log.Printf("Unwanted modules are newly vendored: %q", addedVendored)
		log.Printf("!!! Avoid updates that increase vendoring of unwanted dependencies\n")
		needUpdate = true
	}

	if needUpdate {
		os.Exit(1)
	}
}

func visit(visitor func(m module, via []module), main module, references map[module][]module, effectiveVersions map[string]module) {
	doVisit(visitor, main, nil, map[module]bool{}, references, effectiveVersions)
}

func doVisit(visitor func(m module, via []module), from module, via []module, visited map[module]bool, references map[module][]module, effectiveVersions map[string]module) {
	visitor(from, via)
	via = append(via, from)
	if visited[from] {
		return
	}
	for _, to := range references[from] {
		// switch to the effective version of this dependency
		if override, ok := effectiveVersions[to.name]; ok {
			to = override
		}
		// recurse unless we've already visited this module in this traversal
		if !moduleInSlice(to, via, false) {
			doVisit(visitor, to, via, visited, references, effectiveVersions)
		}
	}
	visited[from] = true
}
