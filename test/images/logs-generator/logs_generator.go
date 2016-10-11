/*
Copyright 2016 The Kubernetes Authors.

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
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

const (
	linesTotalEnvVariableName = "LOGS_GENERATOR_LINES_TOTAL"
	durationEnvVariableName   = "LOGS_GENERATOR_DURATION"
)

var (
	httpMethods = []string{
		"GET",
		"POST",
		"PUT",
		"DELETE",
	}
	namespaces = []string{
		"kube-system",
		"default",
		"my-custom-namespace",
	}
	resources = []string{
		"pods",
		"services",
		"endpoints",
		"configmaps",
	}
)

func main() {
	linesTotal, durationSeconds, err := parseGenerateParameters()
	if err != nil {
		log.Fatalf("Failed to extract parameters for generator from environment: %v", err)
		os.Exit(1)
	}

	generateLogs(linesTotal, durationSeconds)
}

// Extracts parameters for logs generation from environment variables
func parseGenerateParameters() (linesTotal int, durationSeconds int, err error) {
	linesTotalStr, ok := os.LookupEnv(linesTotalEnvVariableName)
	if !ok {
		err = fmt.Errorf("Missing environment variable %s", linesTotalEnvVariableName)
		return
	}
	if linesTotal, err = strconv.Atoi(linesTotalStr); err != nil {
		err = fmt.Errorf("Error parsing %s: %v", linesTotalEnvVariableName, err)
		return
	}

	durationStr, ok := os.LookupEnv(durationEnvVariableName)
	if !ok {
		err = fmt.Errorf("Missing environment variable %s", durationEnvVariableName)
		return
	}
	duration, err := time.ParseDuration(durationStr)
	if err != nil {
		err = fmt.Errorf("Error parsing %s: %v", durationEnvVariableName, err)
		return
	}

	durationSeconds = int(duration.Seconds())
	if durationSeconds <= 0 {
		err = fmt.Errorf("Invalid duration: %v, must be a positive number of seconds", duration)
		return
	}

	return
}

// Outputs linesTotal lines of logs to stdout uniformly throughout durationSeconds seconds
func generateLogs(linesTotal int, durationSeconds int) {
	delay := time.Duration(float64(durationSeconds) / float64(linesTotal) * float64(time.Second))
	randomSource := rand.NewSource(time.Now().UnixNano())

	ch := make(chan (string))

	// Print in the separate goroutine
	go func() {
		for line := range ch {
			fmt.Println(line)
		}
	}()

	// Printing is made this way to avoid time spent on generating line
	// and actual printing ending up influencing the delay between two prints
	tick := time.Tick(delay)
	for id := 0; id < linesTotal; id++ {
		ch <- generateLogLine(randomSource, id)
		<-tick
	}

	close(ch)
}

// Generates apiserver-like line with average length of 100 symbols
func generateLogLine(randomSource rand.Source, id int) string {
	method := httpMethods[int(randomSource.Int63())%len(httpMethods)]
	namespace := namespaces[int(randomSource.Int63())%len(namespaces)]
	resource := resources[int(randomSource.Int63())%len(resources)]

	//Resource name length is from [20; 30] to make total line length 100 symbols on average
	resourceNameLength := int(20 + randomSource.Int63()%11)
	resourceName := generateRandomName(resourceNameLength, randomSource)

	url := fmt.Sprintf("/api/v1/namespaces/%s/%s/%s", namespace, resource, resourceName)
	status := 200 + randomSource.Int63()%300

	return fmt.Sprintf("%s %d %s %s %d", time.Now().Format(time.RFC3339), id, method, url, status)
}

// Generates string of length nameLength, containing random lowercase latin letters
func generateRandomName(nameLength int, randomSource rand.Source) string {
	runes := []rune{}
	for i := 0; i < nameLength; i++ {
		nextRune := rune('a' + randomSource.Int63()%26)
		runes = append(runes, nextRune)
	}

	return string(runes)
}
