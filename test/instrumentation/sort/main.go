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
	"os"
	"sort"

	flag "github.com/spf13/pflag"
	yaml "go.yaml.in/yaml/v2"

	"k8s.io/kubernetes/test/instrumentation/internal/metric"
)

func main() {
	var sortFile string
	flag.StringVar(&sortFile, "sort-file", "", "file of metrics to sort")
	flag.Parse()
	dat, err := os.ReadFile(sortFile)
	if err == nil {
		var parsedMetrics []metric.Metric
		err = yaml.Unmarshal(dat, &parsedMetrics)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err)
			os.Exit(1)
		}
		sort.Sort(metric.ByFQName(parsedMetrics))
		data, err := yaml.Marshal(parsedMetrics)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err)
			os.Exit(1)
		}

		fmt.Print(string(data))
	}
}
