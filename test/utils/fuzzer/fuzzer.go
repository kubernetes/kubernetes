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

package fuzzer

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/sync/errgroup"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/utils"
	"sigs.k8s.io/yaml"
)

// ExemplaryPodFuzzer generates fuzzed Pod objects derived from a base pod.
type ExemplaryPodFuzzer struct {
	rng *rand.Rand

	// Settings for the generated pods
	Namespace  string
	NamePrefix string

	// Cache for identical data to test interning across multiple pods.
	// Maps BasePodName -> FuzzedPrototype
	mu               sync.Mutex
	cachedPrototypes map[string]*v1.Pod
}

// NewExemplaryPodFuzzer creates a new fuzzer with a seeded RNG and global settings.
func NewExemplaryPodFuzzer(seed int64, namePrefix, namespace string) *ExemplaryPodFuzzer {
	if namespace == "" {
		namespace = "default"
	}
	return &ExemplaryPodFuzzer{
		rng:              rand.New(rand.NewSource(seed)),
		NamePrefix:       namePrefix,
		Namespace:        namespace,
		cachedPrototypes: make(map[string]*v1.Pod),
	}
}

// FuzzPod transforms a base pod into a concrete fuzzed v1.Pod object.
func (f *ExemplaryPodFuzzer) FuzzPod(base *v1.Pod, id int) *v1.Pod {
	f.mu.Lock()
	proto, ok := f.cachedPrototypes[base.Name]
	if !ok {
		proto = f.generatePrototype(base)
		f.cachedPrototypes[base.Name] = proto
	}
	f.mu.Unlock()

	pod := proto.DeepCopy()
	pod.Name = fmt.Sprintf("%s-%d", f.NamePrefix, id)
	pod.UID = types.UID(fmt.Sprintf("fuzzed-uid-%08d-%s", id, strings.ToLower(f.randomString(8))))

	return pod
}

func (f *ExemplaryPodFuzzer) generatePrototype(base *v1.Pod) *v1.Pod {
	pod := base.DeepCopy()

	// 1. Sanitize Metadata
	pod.Namespace = f.Namespace
	pod.ResourceVersion = ""
	pod.CreationTimestamp = metav1.Time{}
	pod.GenerateName = ""

	// Fuzz OwnerRefs
	for i := range pod.OwnerReferences {
		pod.OwnerReferences[i].Name = "fuzzed-owner-" + strings.ToLower(f.randomString(8))
		pod.OwnerReferences[i].UID = types.UID("fuzzed-uid-" + strings.ToLower(f.randomString(8)))
	}

	// Fuzz Annotations & Labels
	for k := range pod.Annotations {
		pod.Annotations[k] = "fuzzed-val-" + f.randomString(16)
	}
	for k := range pod.Labels {
		pod.Labels[k] = "fuzzed-label-" + f.randomString(8)
	}

	// 2. Sanitize Spec
	pod.Spec.NodeName = "fuzzed-node-" + strings.ToLower(f.randomString(8))
	pod.Spec.SchedulerName = "non-existent-fuzz-scheduler"
	if pod.Spec.NodeSelector == nil {
		pod.Spec.NodeSelector = make(map[string]string)
	}
	pod.Spec.NodeSelector["disktype"] = "non-existent-ssd"

	// Fuzz Env Vars (Keys and Values)
	for i := range pod.Spec.Containers {
		for j := range pod.Spec.Containers[i].Env {
			pod.Spec.Containers[i].Env[j].Name = "FUZZED_ENV_" + f.randomString(8)
			pod.Spec.Containers[i].Env[j].Value = f.randomString(64)
			pod.Spec.Containers[i].Env[j].ValueFrom = nil // Critical: cannot have both Value and ValueFrom
		}
	}

	for i := range pod.Spec.InitContainers {
		for j := range pod.Spec.InitContainers[i].Env {
			pod.Spec.InitContainers[i].Env[j].Name = "FUZZED_ENV_" + f.randomString(8)
			pod.Spec.InitContainers[i].Env[j].Value = f.randomString(64)
			pod.Spec.InitContainers[i].Env[j].ValueFrom = nil
		}
	}

	// 3. Fuzz ManagedFields
	for i := range pod.ManagedFields {
		if pod.ManagedFields[i].FieldsV1 != nil {
			pod.ManagedFields[i].FieldsV1.Raw = f.fuzzFieldsV1JSON(pod.ManagedFields[i].FieldsV1.Raw)
		}
	}

	// Clear Status
	pod.Status = v1.PodStatus{}

	return pod
}

func (f *ExemplaryPodFuzzer) fuzzFieldsV1JSON(raw []byte) []byte {
	var data map[string]interface{}
	if err := json.Unmarshal(raw, &data); err != nil {
		return raw
	}
	f.fuzzMapRecursive(data)
	res, _ := json.Marshal(data)
	return res
}

func (f *ExemplaryPodFuzzer) fuzzMapRecursive(m map[string]interface{}) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}

	for _, oldKey := range keys {
		val := m[oldKey]
		delete(m, oldKey)

		var newKey string
		if strings.HasPrefix(oldKey, "k:") {
			newKey = fmt.Sprintf("k:{\"id\":%d,\"name\":\"fuzzed-node-%s\"}", f.rng.Intn(100), f.randomString(4))
		} else if strings.HasPrefix(oldKey, "f:") {
			newKey = "f:fuzzed_field_" + f.randomString(4)
		} else if oldKey == "." {
			newKey = "."
		} else {
			newKey = oldKey
		}

		if subMap, ok := val.(map[string]interface{}); ok {
			f.fuzzMapRecursive(subMap)
			m[newKey] = subMap
		} else {
			m[newKey] = val
		}
	}
}

func (f *ExemplaryPodFuzzer) randomString(length int) string {
	if length <= 0 {
		return ""
	}
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[f.rng.Intn(len(charset))]
	}
	return string(b)
}

// ProgressCallback is called periodically during pod generation.
type ProgressCallback func(current, total int)

// ExemplaryPodCreator handles the creation of exemplary pods in a cluster.
type ExemplaryPodCreator struct {
	client clientset.Interface
	fuzzer *ExemplaryPodFuzzer
}

// NewExemplaryPodCreator creates a new creator with settings.
func NewExemplaryPodCreator(client clientset.Interface, seed int64, namePrefix, namespace string) *ExemplaryPodCreator {
	return &ExemplaryPodCreator{
		client: client,
		fuzzer: NewExemplaryPodFuzzer(seed, namePrefix, namespace),
	}
}

// CreateExemplaryPods creates a batch of pods concurrently based on a base pod.
func (c *ExemplaryPodCreator) CreateExemplaryPods(ctx context.Context, base *v1.Pod, count int, offset int, concurrency int, progress ProgressCallback) error {
	return c.processExemplaryPods(ctx, base, count, offset, concurrency, progress, func(pod *v1.Pod) error {
		return utils.CreatePodWithRetries(c.client, pod.Namespace, pod)
	})
}

// WriteExemplaryPodsToDir writes a batch of pod manifests to a directory.
func (c *ExemplaryPodCreator) WriteExemplaryPodsToDir(ctx context.Context, base *v1.Pod, count int, offset int, concurrency int, dirPath string, progress ProgressCallback) (string, error) {
	if dirPath == "" {
		var err error
		dirPath, err = os.MkdirTemp("", "exemplary-pods-")
		if err != nil {
			return "", fmt.Errorf("failed to create temp dir: %w", err)
		}
	}

	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return "", fmt.Errorf("failed to create directory %s: %w", dirPath, err)
	}

	err := c.processExemplaryPods(ctx, base, count, offset, concurrency, progress, func(pod *v1.Pod) error {
		data, err := yaml.Marshal(pod)
		if err != nil {
			return fmt.Errorf("failed to marshal pod %s: %w", pod.Name, err)
		}
		filename := filepath.Join(dirPath, fmt.Sprintf("%s.yaml", pod.Name))
		if err := os.WriteFile(filename, data, 0644); err != nil {
			return fmt.Errorf("failed to write pod file %s: %w", filename, err)
		}
		return nil
	})

	return dirPath, err
}

func (c *ExemplaryPodCreator) processExemplaryPods(ctx context.Context, base *v1.Pod, count int, offset int, concurrency int, progress ProgressCallback, processFunc func(*v1.Pod) error) error {
	g, ctx := errgroup.WithContext(ctx)
	podsChan := make(chan int, count)

	// Producer
	go func() {
		defer close(podsChan)
		for i := range count {
			select {
			case podsChan <- i + offset:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Consumers
	var processed int64
	for range concurrency {
		g.Go(func() error {
			for id := range podsChan {
				pod := c.fuzzer.FuzzPod(base, id)
				if err := processFunc(pod); err != nil {
					return err
				}

				if progress != nil {
					val := atomic.AddInt64(&processed, 1)
					if val%100 == 0 || val == int64(count) {
						progress(int(val), count)
					}
				}
			}
			return nil
		})
	}

	return g.Wait()
}
