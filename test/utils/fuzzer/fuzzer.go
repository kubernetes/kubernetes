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
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
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
	// FieldPathCount specifies the total number of "f:" fields to generate.
	FieldPathCount int `json:"fieldPathCount"`
	// FieldPathDepth specifies the maximum nesting depth for the "f:" fields.
	FieldPathDepth int `json:"fieldPathDepth"`
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

	// Generative complexity settings
	EnvVarCount       int `json:"envVarCount"`       // Number of unique environment variables to generate
	ManagedFieldCount int `json:"managedFieldCount"` // Total number of ManagedFields entries to generate
}

// ExemplaryPodFuzzer generates fuzzed Pod objects based on a template.
type ExemplaryPodFuzzer struct {
	rng *rand.Rand

	// Cache for identical data to test interning across multiple pods
	mu          sync.Mutex
	cachedMF    map[string][]metav1.ManagedFieldsEntry
	cachedAnnos map[string]map[string]string
	cachedEnv   map[string][]v1.EnvVar
}

// NewExemplaryPodFuzzer creates a new fuzzer with a seeded RNG.
func NewExemplaryPodFuzzer(seed int64) *ExemplaryPodFuzzer {
	return &ExemplaryPodFuzzer{
		rng:         rand.New(rand.NewSource(seed)),
		cachedMF:    make(map[string][]metav1.ManagedFieldsEntry),
		cachedAnnos: make(map[string]map[string]string),
		cachedEnv:   make(map[string][]v1.EnvVar),
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

	// 1. Base Spec with Generative Environment Variables
	if template.BaseSpec != nil {
		pod.Spec = *template.BaseSpec.DeepCopy()
	}

	if template.EnvVarCount > 0 && len(pod.Spec.Containers) > 0 {
		f.mu.Lock()
		env, ok := f.cachedEnv[template.Name]
		if !ok {
			env = f.precomputeEnvVars(template.EnvVarCount)
			f.cachedEnv[template.Name] = env
		}
		f.mu.Unlock()
		// Inject into the primary container
		pod.Spec.Containers[0].Env = append(pod.Spec.Containers[0].Env, env...)
	}

	// 2. Metadata Bloat: ManagedFields (Identical across pods for interning)
	f.mu.Lock()
	mfEntries, ok := f.cachedMF[template.Name]
	if !ok {
		mfEntries = f.precomputeManagedFields(template)
		f.cachedMF[template.Name] = mfEntries
	}
	f.mu.Unlock()
	pod.ManagedFields = mfEntries

	// 3. Metadata Bloat: Annotations (Identical across pods for interning)
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

func (f *ExemplaryPodFuzzer) precomputeManagedFields(template *ExemplaryPodTemplate) []metav1.ManagedFieldsEntry {
	if len(template.ManagedFields) == 0 {
		return nil
	}

	totalEntries := template.ManagedFieldCount
	if totalEntries < len(template.ManagedFields) {
		totalEntries = len(template.ManagedFields)
	}

	res := make([]metav1.ManagedFieldsEntry, totalEntries)
	now := metav1.NewTime(time.Now())

	for i := 0; i < totalEntries; i++ {
		mfTemplate := template.ManagedFields[i%len(template.ManagedFields)]
		
		// Build structurally complex FieldsV1
		raw := f.buildComplexFieldsV1(mfTemplate)

		apiVersion := mfTemplate.APIVersion
		if apiVersion == "" {
			apiVersion = "v1"
		}

		// Replicate "few managers, many entries" pattern by cycling suffixes
		res[i] = metav1.ManagedFieldsEntry{
			Manager:    fmt.Sprintf("%s-%d", mfTemplate.Manager, i/len(template.ManagedFields)),
			Operation:  metav1.ManagedFieldsOperationType(mfTemplate.Operation),
			APIVersion: apiVersion,
			Time:       &now,
			FieldsV1:   &metav1.FieldsV1{Raw: raw},
		}
	}
	return res
}

func (f *ExemplaryPodFuzzer) buildComplexFieldsV1(tm ExemplaryManagedFieldTemplate) []byte {
	// Start with the provided schema if any
	schema := tm.FieldsSchema
	if schema == "" || schema == "{}" {
		schema = "{}"
	}

	// If no generative fields requested, return as is (with optional bloat)
	if tm.FieldPathCount <= 0 {
		raw := []byte(schema)
		if tm.Length > len(raw) {
			bloat := f.randomString(tm.Length - len(raw) - 15)
			if schema == "{}" {
				return []byte(fmt.Sprintf(`{"fuzzBloat":"%s"}`, bloat))
			}
			return []byte(fmt.Sprintf(`{"fuzzBloat":"%s",%s`, bloat, schema[1:]))
		}
		return raw
	}

	// Build a complex NESTED structure
	depth := tm.FieldPathDepth
	if depth <= 0 {
		depth = 1
	}
	
	// Create a map-based tree structure
	tree := f.generateNestedMap(tm.FieldPathCount, depth)
	
	genJson, err := json.Marshal(tree)
	if err != nil {
		// Fallback to simple flat JSON if nesting fails
		return []byte(`{"f:error":"generation_failed"}`)
	}
	
	// If a target length is also specified, add a large bloat field
	if tm.Length > len(genJson) {
		bloat := f.randomString(tm.Length - len(genJson) - 17)
		// Inject bloat at the start
		res := fmt.Sprintf(`{"f:fuzzBloat":"%s",%s`, bloat, string(genJson)[1:])
		return []byte(res)
	}
	
	return genJson
}

func (f *ExemplaryPodFuzzer) generateNestedMap(totalFields, maxDepth int) map[string]interface{} {
	root := make(map[string]interface{})
	fieldsCreated := 0
	
	for fieldsCreated < totalFields {
		curr := root
		// Pick a random depth for this specific field branch
		branchDepth := f.rng.Intn(maxDepth) + 1
		
		for d := 0; d < branchDepth; d++ {
			key := fmt.Sprintf("f:node_%d_%d", d, f.rng.Intn(10)) // Use few branch nodes to force overlap
			if d == branchDepth-1 {
				// Leaf node
				key = fmt.Sprintf("f:field_%04d", fieldsCreated)
				curr[key] = map[string]interface{}{}
				fieldsCreated++
			} else {
				// Intermediate node
				if _, ok := curr[key]; !ok {
					curr[key] = make(map[string]interface{})
				}
				curr = curr[key].(map[string]interface{})
			}
		}
	}
	return root
}

func (f *ExemplaryPodFuzzer) precomputeAnnotations(template *ExemplaryPodTemplate) map[string]string {
	res := make(map[string]string)
	for _, annTemplate := range template.Annotations {
		res[annTemplate.Key] = f.randomString(annTemplate.Length)
	}
	return res
}

func (f *ExemplaryPodFuzzer) precomputeEnvVars(count int) []v1.EnvVar {
	res := make([]v1.EnvVar, count)
	for i := 0; i < count; i++ {
		res[i] = v1.EnvVar{
			Name:  fmt.Sprintf("FUZZ_GEN_VARIABLE_%03d", i),
			Value: f.randomString(64),
		}
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

// ProgressCallback is called periodically during pod generation.
type ProgressCallback func(current, total int)

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
func (c *ExemplaryPodCreator) CreateExemplaryPods(ctx context.Context, template *ExemplaryPodTemplate, count int, concurrency int, progress ProgressCallback) error {
	return c.processExemplaryPods(ctx, template, count, concurrency, progress, func(pod *v1.Pod) error {
		return utils.CreatePodWithRetries(c.client, pod.Namespace, pod)
	})
}

// WriteExemplaryPodsToDir writes a batch of pod manifests to a directory.
// If dirPath is empty, a temporary directory is created and returned.
func (c *ExemplaryPodCreator) WriteExemplaryPodsToDir(ctx context.Context, template *ExemplaryPodTemplate, count int, concurrency int, dirPath string, progress ProgressCallback) (string, error) {
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

	err := c.processExemplaryPods(ctx, template, count, concurrency, progress, func(pod *v1.Pod) error {
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

func (c *ExemplaryPodCreator) processExemplaryPods(ctx context.Context, template *ExemplaryPodTemplate, count int, concurrency int, progress ProgressCallback, processFunc func(*v1.Pod) error) error {
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
	var processed int64
	for i := 0; i < concurrency; i++ {
		g.Go(func() error {
			for id := range podsChan {
				pod := c.fuzzer.FuzzPod(template, id)
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
