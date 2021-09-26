// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// summarize is a tool for summarizing the results of gnostic_analyze runs.
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"sort"

	"github.com/googleapis/gnostic/plugins/gnostic-analyze/statistics"
)

// Results are collected in this global slice.
var stats []statistics.DocumentStatistics

// walker is called for each summary file found.
func walker(p string, info os.FileInfo, err error) error {
	basename := path.Base(p)
	if basename != "summary.json" {
		return nil
	}
	data, err := ioutil.ReadFile(p)
	if err != nil {
		return err
	}
	var s statistics.DocumentStatistics
	err = json.Unmarshal(data, &s)
	if err != nil {
		return err
	}
	stats = append(stats, s)
	return nil
}

func printFrequencies(m map[string]int) {
	for _, pair := range rankByCount(m) {
		fmt.Printf("%6d %s\n", pair.Value, pair.Key)
	}
}

func rankByCount(frequencies map[string]int) pairList {
	pl := make(pairList, len(frequencies))
	i := 0
	for k, v := range frequencies {
		pl[i] = pair{k, v}
		i++
	}
	sort.Sort(sort.Reverse(pl))
	return pl
}

type pair struct {
	Key   string
	Value int
}

type pairList []pair

func (p pairList) Len() int           { return len(p) }
func (p pairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func (p pairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func main() {
	// Collect all statistics in the current directory and its subdirectories.
	stats = make([]statistics.DocumentStatistics, 0)
	filepath.Walk(".", walker)

	// Compute some interesting properties.
	apisWithAnonymousOperations := 0
	apisWithAnonymousObjects := 0
	apisWithAnonymousAnything := 0
	opFrequencies := make(map[string]int, 0)
	parameterTypeFrequencies := make(map[string]int, 0)
	resultTypeFrequencies := make(map[string]int, 0)
	definitionFieldTypeFrequencies := make(map[string]int, 0)
	definitionArrayTypeFrequencies := make(map[string]int, 0)
	definitionPrimitiveTypeFrequencies := make(map[string]int, 0)

	for _, api := range stats {
		if api.Operations["anonymous"] != 0 {
			apisWithAnonymousOperations++
		}
		if len(api.AnonymousObjects) > 0 {
			apisWithAnonymousObjects++
		}
		if len(api.AnonymousOperations) > 0 {
			apisWithAnonymousAnything++
			if len(api.AnonymousObjects) > 0 {
				fmt.Printf("%s has anonymous operations and objects\n", api.Name)
			} else {
				fmt.Printf("%s has anonymous operations\n", api.Name)
			}
		} else {
			if len(api.AnonymousObjects) > 0 {
				apisWithAnonymousAnything++
				fmt.Printf("%s has anonymous objects\n", api.Name)
			} else {
				fmt.Printf("%s has no anonymous operations or objects\n", api.Name)
			}
		}
		for k, v := range api.Operations {
			opFrequencies[k] += v
		}
		for k, v := range api.ParameterTypes {
			parameterTypeFrequencies[k] += v
		}
		for k, v := range api.ResultTypes {
			resultTypeFrequencies[k] += v
		}
		for k, v := range api.DefinitionFieldTypes {
			definitionFieldTypeFrequencies[k] += v
		}
		for k, v := range api.DefinitionArrayTypes {
			definitionArrayTypeFrequencies[k] += v
		}
		for k, v := range api.DefinitionPrimitiveTypes {
			definitionPrimitiveTypeFrequencies[k] += v
		}
	}

	// Report the results.
	fmt.Printf("\n")
	fmt.Printf("Collected information on %d APIs.\n\n", len(stats))
	fmt.Printf("APIs with anonymous operations: %d\n", apisWithAnonymousOperations)
	fmt.Printf("APIs with anonymous objects: %d\n", apisWithAnonymousObjects)
	fmt.Printf("APIs with anonymous anything: %d\n", apisWithAnonymousAnything)
	fmt.Printf("\nOperation frequencies:\n")
	printFrequencies(opFrequencies)
	fmt.Printf("\nParameter type frequencies:\n")
	printFrequencies(parameterTypeFrequencies)
	fmt.Printf("\nResult type frequencies:\n")
	printFrequencies(resultTypeFrequencies)
	fmt.Printf("\nDefinition object field type frequencies:\n")
	printFrequencies(definitionFieldTypeFrequencies)
	fmt.Printf("\nDefinition array type frequencies:\n")
	printFrequencies(definitionArrayTypeFrequencies)
	fmt.Printf("\nDefinition primitive type frequencies:\n")
	printFrequencies(definitionPrimitiveTypeFrequencies)
}
