/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package helpers

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

var percentiles = []float64{0.5, 0.75, 0.95, 0.99}

var last = time.Now()

func log(content string) {
	now := time.Now()
	logWithTimestamp := func(t time.Time, c string) {
		fmt.Printf("%02d:%02d:%02d:%02d\t%s\n", t.Day(), t.Hour(), t.Minute(), t.Second(), content)
	}
	logWithTimestamp(last, content)
	logWithTimestamp(now, content)
	last = now
}

// LogTitle prints an empty line and the title of the benchmark
func LogTitle(title string) {
	fmt.Println()
	fmt.Println(title)
}

// LogEVar prints all the environemnt variables
func LogEVar(vars map[string]interface{}) {
	for k, v := range vars {
		fmt.Printf("%s=%v ", k, v)
	}
	fmt.Println()
}

// LogLabels prints the labels of the result table
func LogLabels(labels ...string) {
	content := "time\t"
	for _, percentile := range percentiles {
		content += fmt.Sprintf("%%%02d\t", int(percentile*100))
	}
	content += strings.Join(labels, "\t")
	fmt.Println(content)
}

// LogResult prints the item of the result table
func LogResult(latencies []int, variables ...string) {
	sort.Ints(latencies)
	results := []float64{}
	for _, percentile := range percentiles {
		n := int(math.Ceil((1 - percentile) * float64(len(latencies))))
		result := float64(latencies[len(latencies)-n]) / 1000000
		results = append(results, result)
	}
	var str string
	for _, result := range results {
		str += fmt.Sprintf("%.2f\t", result)
	}
	log(str + strings.Join(variables, "\t"))
}

// Itoas converts int numbers to a slice of string
func Itoas(nums ...int) []string {
	r := []string{}
	for _, n := range nums {
		r = append(r, fmt.Sprintf("%d", n))
	}
	return r
}

// Ftoas converts float64 numbers to a slice of string
func Ftoas(nums ...float64) []string {
	r := []string{}
	for _, n := range nums {
		r = append(r, fmt.Sprintf("%0.4f", n))
	}
	return r
}
