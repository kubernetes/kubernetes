/*
Copyright 2026 The Kubernetes Authors.

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
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"k8s.io/kubernetes/test/utils/fuzzer"
)

func main() {
	count := flag.Int("count", 1000, "Number of pods to generate")
	templatePath := flag.String("template", "test/utils/fuzzer/templates/representative-pod.yaml", "Path to the pod template YAML")
	outDir := flag.String("out-dir", "", "Directory to write YAMLs (default: temp dir)")
	concurrency := flag.Int("concurrency", 50, "Number of concurrent workers")

	flag.Parse()

	template, err := fuzzer.LoadTemplateFromFile(*templatePath)
	if err != nil {
		log.Fatalf("Failed to load template: %v", err)
	}

	creator := fuzzer.NewExemplaryPodCreator(nil, time.Now().UnixNano())
	
	fmt.Printf("Generating %d pods from template %s...\n", *count, *templatePath)
	start := time.Now()
	
	dir, err := creator.WriteExemplaryPodsToDir(context.Background(), template, *count, *concurrency, *outDir)
	if err != nil {
		log.Fatalf("Failed to generate pods: %v", err)
	}

	duration := time.Since(start)
	fmt.Printf("Successfully created %d pod manifests in: %s\n", *count, dir)
	fmt.Printf("Time taken: %v\n", duration)

	// Validation snippet
	files, _ := os.ReadDir(dir)
	if len(files) != *count {
		log.Fatalf("Validation failed: expected %d files, found %d", *count, len(files))
	}
	fmt.Println("Validation: Correct number of files created.")
}
