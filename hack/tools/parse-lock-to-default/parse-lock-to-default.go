/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"os"
	"regexp"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)
	featureGateRegex := regexp.MustCompile(`^\s*(\w+):\s*{`) // regex to capture feature gates with form- " SomeFeature: {"
	lockToDefaultTrueRegex := regexp.MustCompile(`LockToDefault:\s*true`)

	featureMap := make(map[string]struct{})

	var currentFeature string

	for scanner.Scan() {
		line := scanner.Text()

		if matches := featureGateRegex.FindStringSubmatch(line); len(matches) > 1 {
			currentFeature = matches[1]
		}

		if lockToDefaultTrueRegex.MatchString(line) && currentFeature != "" {
			featureMap[currentFeature] = struct{}{}
			currentFeature = ""
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "Error reading stdin:", err)
	}

	fmt.Println("previousFeatureGatesWithLockToDefaultTrue: map[string]struct{}{")
	for feature := range featureMap {
		fmt.Printf("\t%q: {},\n", feature)
	}
	fmt.Println("}")
}
