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
	"sort"
)

func main() {
	scanner := bufio.NewScanner(os.Stdin)

	// regex to match the feature name and its enabled status
	re := regexp.MustCompile(`kubernetes_feature_enabled{name="([^"]+)",stage="[^"]*"} (\d+)`)

	featureGates := make(map[string]int)

	for scanner.Scan() {
		line := scanner.Text()

		matches := re.FindStringSubmatch(line)
		if len(matches) == 3 {
			featureName := matches[1]
			enabled := matches[2]

			// Convert enabled value to int
			if enabled == "1" {
				featureGates[featureName] = 1
			} else {
				featureGates[featureName] = 0
			}
		}
	}

	// Extract keys (feature names) and sort them
	keys := make([]string, 0, len(featureGates))
	for key := range featureGates {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	// Print the output in the desired map format, with sorted keys
	fmt.Println(`previousFeatureGates: map[string]int{`)
	for _, key := range keys {
		fmt.Printf("\t\"%s\": %d,\n", key, featureGates[key])
	}
	fmt.Println(`}`)
}
