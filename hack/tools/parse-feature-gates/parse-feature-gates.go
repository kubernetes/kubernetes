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
	"strconv"
)

type featureGateInfo struct {
	stage         string
	lockToDefault bool
	enabled       int
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <metrics_file> <source_file>\n", os.Args[0])
		os.Exit(1)
	}

	// Initialize feature gates map
	featureGates := make(map[string]featureGateInfo)

	// Parse metrics file
	if err := parseMetrics(os.Args[1], featureGates); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing metrics file: %v\n", err)
		os.Exit(1)
	}

	// Parse source file
	if err := parseSources(os.Args[2], featureGates); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing source file: %v\n", err)
		os.Exit(1)
	}

	// Print test case for test/integration/apiserver/compatibility_versions_test.go - TestFeatureGateCompatibilityEmulationVersion
	printFeatureGates(featureGates)
}

func parseMetrics(filename string, featureGates map[string]featureGateInfo) error {
	content, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read metrics file: %v", err)
	}

	metricsRe := regexp.MustCompile(`kubernetes_feature_enabled{name="([^"]+)",stage="([^"]*)"} (\d+)`)
	matches := metricsRe.FindAllStringSubmatch(string(content), -1)

	for _, match := range matches {
		featureName := match[1]
		stage := match[2]
		enabledStr := match[3]

		enabled, err := strconv.Atoi(enabledStr)
		if err != nil {
			return fmt.Errorf("unable to convert value %s to integer for feature gate %s", enabled, featureName)
		}

		featureGates[featureName] = featureGateInfo{
			stage:         stage,
			lockToDefault: false,
			enabled:       enabled,
		}
	}
	return nil
}

func parseSources(filename string, featureGates map[string]featureGateInfo) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open source file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	featureGateRegex := regexp.MustCompile(`^\s*(\w+):\s*{`)
	lockToDefaultTrueRegex := regexp.MustCompile(`LockToDefault:\s*true`)
	var currentFeature string

	for scanner.Scan() {
		line := scanner.Text()
		if matches := featureGateRegex.FindStringSubmatch(line); len(matches) > 1 {
			currentFeature = matches[1]
		}
		if lockToDefaultTrueRegex.MatchString(line) && currentFeature != "" {
			// Update existing feature gate info or create new one
			if info, exists := featureGates[currentFeature]; exists {
				info.lockToDefault = true
				featureGates[currentFeature] = info
			} else {
				featureGates[currentFeature] = featureGateInfo{
					stage:         "",
					lockToDefault: true,
					enabled:       0,
				}
			}
			currentFeature = ""
		}
	}
	return scanner.Err()
}

func printFeatureGates(featureGates map[string]featureGateInfo) {
	// Extract and sort keys
	keys := make([]string, 0, len(featureGates))
	for key := range featureGates {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	// Print combined output
	fmt.Println("expectedFeatureGatesForVersion: map[string]featureGateInfo{")
	for _, key := range keys {
		info := featureGates[key]
		fmt.Printf("\t%q: {\"%s\", %v, %d},\n",
			key,
			info.stage,
			info.lockToDefault,
			info.enabled)
	}
	fmt.Println("}")
}
