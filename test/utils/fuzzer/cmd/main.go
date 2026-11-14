/*
Copyright The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/test/utils/fuzzer"
	"sigs.k8s.io/yaml"
)

func main() {
	count := flag.Int("count", 1000, "Number of pods to generate")
	offset := flag.Int("offset", 0, "Starting index for pod naming")
	basePodPath := flag.String("base-pod", "test/utils/fuzzer/templates/complex-daemonset.yaml", "Path to the real pod YAML to sanitize and clone")
	namespace := flag.String("namespace", "fuzz-test", "Target namespace for fuzzed pods")
	namePrefix := flag.String("name-prefix", "fuzzed-pod", "Prefix for generated pod names")
	outDir := flag.String("out-dir", "", "Directory to write YAMLs (if specified, no cluster injection)")
	concurrency := flag.Int("concurrency", 50, "Number of concurrent workers")
	kubeconfig := flag.String("kubeconfig", "", "Path to the kubeconfig file for direct injection")

	flag.Parse()

	// Load Base Pod
	basePodData, err := os.ReadFile(*basePodPath)
	if err != nil {
		log.Fatalf("Failed to read base pod file: %v", err)
	}
	var basePod v1.Pod
	if err := yaml.Unmarshal(basePodData, &basePod); err != nil {
		log.Fatalf("Failed to unmarshal base pod: %v", err)
	}

	var clientset *kubernetes.Clientset
	if *outDir == "" {
		loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
		if *kubeconfig != "" {
			loadingRules.ExplicitPath = *kubeconfig
		}
		config, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, &clientcmd.ConfigOverrides{}).ClientConfig()
		if err != nil {
			log.Fatalf("Failed to load kubeconfig: %v", err)
		}

		// Increase QPS and Burst for high-performance injection
		config.QPS = 500
		config.Burst = 1000
		clientset, err = kubernetes.NewForConfig(config)
		if err != nil {
			log.Fatalf("Failed to create clientset: %v", err)
		}
	}

	creator := fuzzer.NewExemplaryPodCreator(clientset, time.Now().UnixNano(), *namePrefix, *namespace)

	progress := func(current, total int) {
		fmt.Printf("\rProgress: %d/%d pods (%.1f%%)", current, total, float64(current)/float64(total)*100)
		if current == total {
			fmt.Println()
		}
	}

	start := time.Now()
	if *outDir != "" {
		fmt.Printf("Writing %d fuzzed pod manifests to %s (base: %s)...\n", *count, *outDir, *basePodPath)
		dir, err := creator.WriteExemplaryPodsToDir(context.Background(), &basePod, *count, *offset, *concurrency, *outDir, progress)
		if err != nil {
			log.Fatalf("\nFailed to write pods: %v", err)
		}
		fmt.Printf("Successfully created %d pod manifests in: %s\n", *count, dir)
	} else {
		fmt.Printf("Injecting %d fuzzed pods into cluster (base: %s)...\n", *count, *basePodPath)
		err := creator.CreateExemplaryPods(context.Background(), &basePod, *count, *offset, *concurrency, progress)
		if err != nil {
			log.Fatalf("\nFailed to inject pods: %v", err)
		}
		fmt.Printf("Successfully injected %d pods.\n", *count)
	}

	duration := time.Since(start)
	fmt.Printf("Time taken: %v\n", duration)
}
