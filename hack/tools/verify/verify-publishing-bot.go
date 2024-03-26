/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v2"
)

func getGomodDependencies(rootdir string, components []string) map[string][]string {
	allDependencies := make(map[string][]string)
	for _, component := range components {
		filePath := filepath.Join(rootdir, component, "go.mod")
		file, err := os.Open(filePath)
		if err != nil {
			log.Fatalf("Failed to open go.mod file for component %s: %v", component, err)
		}
		defer file.Close()

		fmt.Printf("%s dependencies\n", component)
		allDependencies[component] = []string{}

		content, err := ioutil.ReadAll(file)
		if err != nil {
			log.Fatalf("Failed to read go.mod file for component %s: %v", component, err)
		}

		lines := strings.Split(string(content), "\n")
		uniqueLines := make(map[string]bool)
		for _, line := range lines {
			uniqueLines[line] = true
		}

		for _, line := range lines {
			for _, dep := range components {
				if dep == component {
					continue
				}
				if !strings.Contains(line, "k8s.io/"+dep+" =>") {
					continue
				}
				fmt.Printf("\t%s\n", dep)
				if !contains(allDependencies[component], dep) {
					allDependencies[component] = append(allDependencies[component], dep)
				}
			}
		}
	}
	return allDependencies
}

func getRulesDependencies(rulesFile string) map[string]interface{} {
	data := make(map[string]interface{})
	file, err := os.Open(rulesFile)
	if err != nil {
		log.Fatalf("Failed to open rules file: %v", err)
	}
	defer file.Close()

	content, err := ioutil.ReadAll(file)
	if err != nil {
		log.Fatalf("Failed to read rules file: %v", err)
	}

	err = yaml.Unmarshal(content, &data)
	if err != nil {
		log.Fatalf("Failed to parse rules file: %v", err)
	}

	return data
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func main() {
	rootdir := filepath.Join(filepath.Dir(os.Args[0]), "/../../../")
	rootdir, err := filepath.Abs(rootdir)
	if err != nil {
		log.Fatalf("Failed to get absolute path: %v", err)
	}

	components := []string{}
	files, err := ioutil.ReadDir(filepath.Join(rootdir, "staging/src/k8s.io/"))
	if err != nil {
		log.Fatalf("Failed to read components directory: %v", err)
	}
	for _, file := range files {
		if file.IsDir() {
			components = append(components, file.Name())
		}
	}

	rulesFile := filepath.Join(rootdir, "staging/publishing/rules.yaml")
	rulesDependencies := getRulesDependencies(rulesFile)
	gomodDependencies := getGomodDependencies(filepath.Join(rootdir, "staging/src/k8s.io/"), components)

	processedRepos := []string{}
	for _, rule := range rulesDependencies["rules"].([]interface{}) {
		ruleData := rule.(map[interface{}]interface{})
		destination := ruleData["destination"].(string)
		branches := ruleData["branches"].([]interface{})
		branch := branches[0].(map[interface{}]interface{})

		if _, exists := gomodDependencies[destination]; !exists {
			// Make sure we don't include a rule to publish it from master
			for _, b := range branches {
				branchData := b.(map[interface{}]interface{})
				branchName := branchData["name"].(string)
				if branchName == "master" {
					log.Fatalf("Cannot find master branch for destination %s", destination)
				}
			}
			// And skip validation of publishing rules for it
			continue
		}

		for _, b := range branches {
			branchData := b.(map[interface{}]interface{})
			sourceDir := branchData["source"].(map[interface{}]interface{})["dir"].(string)

			fmt.Println()
			fmt.Println()
			fmt.Println()
			fmt.Println()
			fmt.Println(sourceDir)
			fmt.Println()
			fmt.Println()
			fmt.Println()
			fmt.Println()

			if sourceDir != "" {
				log.Fatalf("use of deprecated `dir` field in rules for `%s`", destination)
			}
			if len(branchData["source"].(map[interface{}]interface{})["dirs"].(string)) > 1 {
				log.Fatalf("cannot have more than one directory (`%s`) per source branch `%s` of `%s`",
					(branchData["source"].(map[interface{}]interface{})["dirs"].(string)),
					(branchData["source"].(map[interface{}]interface{})["branch"]),
					destination,
				)
			}

			if !strings.HasSuffix(sourceDir, destination) {
				log.Fatalf("Copy/paste error `%s` refers to `%s`", destination, sourceDir)
			}
		}

		branchName := branch["name"].(string)
		if branchName != "master" {
			log.Fatalf("Cannot find master branch for destination %s", destination)
		}

		sourceBranch := branch["source"].(map[interface{}]interface{})["branch"].(string)
		if sourceBranch != "master" {
			log.Fatalf("Cannot find master source branch for destination %s", destination)
		}

		if _, exists := branch["go"]; exists {
			log.Fatalf("Go version must not be specified for master branch for destination %s", destination)
		}

		fmt.Printf("processing : %s\n", destination)
		if _, exists := gomodDependencies[destination]; !exists {
			log.Fatalf("Missing go.mod for %s", destination)
		}
		processedRepos = append(processedRepos, destination)
		processedDeps := []string{}
		for _, dep := range gomodDependencies[destination] {
			found := false
			if _, exists := branch["dependencies"]; exists {
				dependencies := branch["dependencies"].([]interface{})
				for _, depData := range dependencies {
					depMap := depData.(map[interface{}]interface{})
					depRepo := depMap["repository"].(string)
					depBranch := depMap["branch"].(string)
					if depBranch != "master" {
						log.Fatalf("Looking for master branch and found: %s for destination %s", depBranch, destination)
					}
					if depRepo == dep {
						found = true
						break
					}
				}
			} else {
				log.Fatalf("Please add %s as dependencies under destination %s in %s", gomodDependencies[destination], destination, rulesFile)
			}

			if !found {
				log.Fatalf("Please add %s as a dependency under destination %s in %s", dep, destination, rulesFile)
			} else {
				fmt.Printf("  found dependency %s\n", dep)
			}
			processedDeps = append(processedDeps, dep)
		}

		extraDeps := []string{}
		for _, dep := range processedDeps {
			if !contains(gomodDependencies[destination], dep) {
				extraDeps = append(extraDeps, dep)
			}
		}

		if len(extraDeps) > 0 {
			log.Fatalf("Extra dependencies in rules for %s: %s", destination, strings.Join(extraDeps, ","))
		}
	}

	items := []string{}
	for k := range gomodDependencies {
		if !contains(processedRepos, k) {
			items = append(items, k)
		}
	}
	if len(items) > 0 {
		log.Fatalf("Missing rules for %s", strings.Join(items, ","))
	}

	fmt.Println("Done.")
}
