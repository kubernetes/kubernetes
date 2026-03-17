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

package fuzzer

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"sigs.k8s.io/yaml"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/utils"
)

// ExemplaryManagedFieldTemplate defines how to generate a ManagedFieldsEntry.
type ExemplaryManagedFieldTemplate struct {
	Manager    string `json:"manager"`
	Operation  string `json:"operation"`
	APIVersion string `json:"apiVersion"`
	// FieldsSchema is a JSON string representing the FieldsV1 structure.
	FieldsSchema string `json:"fieldsSchema"`
	// Length specifies a target length for the FieldsV1.Raw data to simulate bloat.
	Length int `json:"length"`
}

// ExemplaryAnnotationTemplate defines how to generate an annotation.
type ExemplaryAnnotationTemplate struct {
	Key    string `json:"key"`
	Length int    `json:"length"`
}

// ExemplaryPodTemplate defines the "shape" of pods to be created for fuzzing.
type ExemplaryPodTemplate struct {
	Name          string                          `json:"name"`
	BaseSpec      *v1.PodSpec                     `json:"baseSpec"`
	ManagedFields []ExemplaryManagedFieldTemplate `json:"managedFields"`
	Annotations   []ExemplaryAnnotationTemplate   `json:"annotations"`
	Namespace     string                          `json:"namespace"`
}

// ExemplaryPodFuzzer generates fuzzed Pod objects based on a template.
type ExemplaryPodFuzzer struct {
	rng *rand.Rand
	
	// Cache for identical strings to test interning
	mu             sync.Mutex
	cachedMF       map[string][][]byte
	cachedAnnos    map[string]map[string]string
}

// NewExemplaryPodFuzzer creates a new fuzzer with a seeded RNG.
func NewExemplaryPodFuzzer(seed int64) *ExemplaryPodFuzzer {
	return &ExemplaryPodFuzzer{
		rng:         rand.New(rand.NewSource(seed)),
		cachedMF:    make(map[string][][]byte),
		cachedAnnos: make(map[string]map[string]string),
	}
}

// FuzzPod transforms an ExemplaryPodTemplate into a concrete v1.Pod object.
func (f *ExemplaryPodFuzzer) FuzzPod(template *ExemplaryPodTemplate, id int) *v1.Pod {
	podName := fmt.Sprintf("%s-%d", template.Name, id)
	namespace := template.Namespace
	if namespace == "" {
		namespace = "default"
	}

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
	}

	// 1. Shared PodSpec (Deduplication Test)
	if template.BaseSpec != nil {
		pod.Spec = *template.BaseSpec
	}

	// 2. Metadata Bloat: ManagedFields (Deduplicated for Interning Test)
	f.mu.Lock()
	rawFields, ok := f.cachedMF[template.Name]
	if !ok {
		rawFields = f.precomputeManagedFields(template)
		f.cachedMF[template.Name] = rawFields
	}
	f.mu.Unlock()

	if len(rawFields) > 0 {
		pod.ManagedFields = make([]metav1.ManagedFieldsEntry, len(template.ManagedFields))
		now := metav1.NewTime(time.Now())
		for i, mfTemplate := range template.ManagedFields {
			apiVersion := mfTemplate.APIVersion
			if apiVersion == "" {
				apiVersion = "v1"
			}
			pod.ManagedFields[i] = metav1.ManagedFieldsEntry{
				Manager:    mfTemplate.Manager,
				Operation:  metav1.ManagedFieldsOperationType(mfTemplate.Operation),
				APIVersion: apiVersion,
				Time:       &now,
				FieldsV1:   &metav1.FieldsV1{Raw: rawFields[i]},
			}
		}
	}

	// 3. Metadata Bloat: Annotations (Deduplicated for Interning Test)
	f.mu.Lock()
	annos, ok := f.cachedAnnos[template.Name]
	if !ok {
		annos = f.precomputeAnnotations(template)
		f.cachedAnnos[template.Name] = annos
	}
	f.mu.Unlock()

	if len(annos) > 0 {
		pod.Annotations = make(map[string]string)
		for k, v := range annos {
			pod.Annotations[k] = v
		}
	}

	return pod
}

func (f *ExemplaryPodFuzzer) precomputeManagedFields(template *ExemplaryPodTemplate) [][]byte {
	res := make([][]byte, len(template.ManagedFields))
	for i, mfTemplate := range template.ManagedFields {
		raw := []byte(mfTemplate.FieldsSchema)
		if mfTemplate.Length > len(raw) {
			bloat := f.randomString(mfTemplate.Length - len(raw) - 13)
			raw = []byte(fmt.Sprintf(`{"fuzzBloat":"%s",%s`, bloat, mfTemplate.FieldsSchema[1:]))
		}
		res[i] = raw
	}
	return res
}

func (f *ExemplaryPodFuzzer) precomputeAnnotations(template *ExemplaryPodTemplate) map[string]string {
	res := make(map[string]string)
	for _, annTemplate := range template.Annotations {
		res[annTemplate.Key] = f.randomString(annTemplate.Length)
	}
	return res
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

// ExemplaryPodCreator handles the creation of exemplary pods in a cluster.
type ExemplaryPodCreator struct {
	client clientset.Interface
	fuzzer *ExemplaryPodFuzzer
}

// NewExemplaryPodCreator creates a new creator.
func NewExemplaryPodCreator(client clientset.Interface, seed int64) *ExemplaryPodCreator {
	return &ExemplaryPodCreator{
		client: client,
		fuzzer: NewExemplaryPodFuzzer(seed),
	}
}

// LoadTemplateFromFile loads an ExemplaryPodTemplate from a YAML/JSON file.
func LoadTemplateFromFile(path string) (*ExemplaryPodTemplate, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read template file: %v", err)
	}
	var template ExemplaryPodTemplate
	if err := yaml.Unmarshal(data, &template); err != nil {
		return nil, fmt.Errorf("failed to unmarshal template: %v", err)
	}
	return &template, nil
}

// CreateExemplaryPods creates a batch of pods concurrently based on a template.
func (c *ExemplaryPodCreator) CreateExemplaryPods(ctx context.Context, template *ExemplaryPodTemplate, count int, concurrency int) error {
	return c.processExemplaryPods(ctx, template, count, concurrency, func(pod *v1.Pod) error {
		return utils.CreatePodWithRetries(c.client, pod.Namespace, pod)
	})
}

// WriteExemplaryPodsToDir writes a batch of pod manifests to a directory.
// If dirPath is empty, a temporary directory is created and returned.
func (c *ExemplaryPodCreator) WriteExemplaryPodsToDir(ctx context.Context, template *ExemplaryPodTemplate, count int, concurrency int, dirPath string) (string, error) {
	if dirPath == "" {
		var err error
		dirPath, err = os.MkdirTemp("", "exemplary-pods-")
		if err != nil {
			return "", fmt.Errorf("failed to create temp dir: %v", err)
		}
	}

	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return "", fmt.Errorf("failed to create directory %s: %v", dirPath, err)
	}

	err := c.processExemplaryPods(ctx, template, count, concurrency, func(pod *v1.Pod) error {
		data, err := yaml.Marshal(pod)
		if err != nil {
			return fmt.Errorf("failed to marshal pod %s: %v", pod.Name, err)
		}
		filename := filepath.Join(dirPath, fmt.Sprintf("%s.yaml", pod.Name))
		if err := os.WriteFile(filename, data, 0644); err != nil {
			return fmt.Errorf("failed to write pod file %s: %v", filename, err)
		}
		return nil
	})

	return dirPath, err
}

func (c *ExemplaryPodCreator) processExemplaryPods(ctx context.Context, template *ExemplaryPodTemplate, count int, concurrency int, processFunc func(*v1.Pod) error) error {
	g, ctx := errgroup.WithContext(ctx)
	podsChan := make(chan int, count)

	// Producer
	go func() {
		defer close(podsChan)
		for i := 0; i < count; i++ {
			select {
			case podsChan <- i:
			case <-ctx.Done():
				return
			}
		}
	}()

	// Consumers
	for i := 0; i < concurrency; i++ {
		g.Go(func() error {
			for id := range podsChan {
				pod := c.fuzzer.FuzzPod(template, id)
				if err := processFunc(pod); err != nil {
					return err
				}
			}
			return nil
		})
	}

	return g.Wait()
}
