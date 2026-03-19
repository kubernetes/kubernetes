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

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/test/utils/fuzzer"
)

func main() {
	count := flag.Int("count", 1000, "Number of pods to generate")
	offset := flag.Int("offset", 0, "Starting index for pod naming (prevents overwrites on multi-run)")
	templatePath := flag.String("template", "test/utils/fuzzer/templates/representative-pod.yaml", "Path to the pod template YAML")
	outDir := flag.String("out-dir", "", "Directory to write YAMLs (if specified, no cluster injection)")
	concurrency := flag.Int("concurrency", 50, "Number of concurrent workers")
	kubeconfig := flag.String("kubeconfig", "", "Path to the kubeconfig file for direct injection")

	flag.Parse()

	template, err := fuzzer.LoadTemplateFromFile(*templatePath)
	if err != nil {
		log.Fatalf("Failed to load template: %v", err)
	}

	var clientset *kubernetes.Clientset
	if *outDir == "" {
		config, err := clientcmd.BuildConfigFromFlags("", *kubeconfig)
		if err != nil {
			// Try default path if not specified
			if *kubeconfig == "" {
				home, _ := os.UserHomeDir()
				config, err = clientcmd.BuildConfigFromFlags("", home+"/.kube/config")
			}
			if err != nil {
				log.Fatalf("Failed to build kubeconfig: %v", err)
			}
		}
		// Increase QPS and Burst
		config.QPS = 500
		config.Burst = 1000
		clientset, err = kubernetes.NewForConfig(config)
		if err != nil {
			log.Fatalf("Failed to create clientset with high QPS: %v", err)
		}
	}

	creator := fuzzer.NewExemplaryPodCreator(clientset, time.Now().UnixNano())

	progress := func(current, total int) {
		fmt.Printf("\rProgress: %d/%d pods (%.1f%%)", current, total, float64(current)/float64(total)*100)
		if current == total {
			fmt.Println()
		}
	}

	start := time.Now()
	if *outDir != "" {
		fmt.Printf("Writing %d pod manifests to %s (offset %d)...\n", *count, *outDir, *offset)
		dir, err := creator.WriteExemplaryPodsToDir(context.Background(), template, *count, *offset, *concurrency, *outDir, progress)
		if err != nil {
			log.Fatalf("\nFailed to write pods: %v", err)
		}
		fmt.Printf("Successfully created %d pod manifests in: %s\n", *count, dir)
	} else {
		fmt.Printf("Injecting %d pods into cluster (offset %d)...\n", *count, *offset)
		err := creator.CreateExemplaryPods(context.Background(), template, *count, *offset, *concurrency, progress)
		if err != nil {
			log.Fatalf("\nFailed to inject pods: %v", err)
		}
		fmt.Printf("Successfully injected %d pods.\n", *count)
	}

	duration := time.Since(start)
	fmt.Printf("Time taken: %v\n", duration)
}
