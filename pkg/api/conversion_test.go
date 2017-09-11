/*
Copyright 2015 The Kubernetes Authors.

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

package api_test

import (
	"io/ioutil"
	"math/rand"
	"testing"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
)

func BenchmarkPodConversion(b *testing.B) {
	apiObjectFuzzer := fuzzer.FuzzerFor(kapitesting.FuzzerFuncs, rand.NewSource(benchmarkSeed), api.Codecs)
	items := make([]api.Pod, 4)
	for i := range items {
		apiObjectFuzzer.Fuzz(&items[i])
		items[i].Spec.InitContainers = nil
		items[i].Status.InitContainerStatuses = nil
	}

	// add a fixed item
	items = append(items, benchmarkPod)
	width := len(items)

	scheme := api.Scheme
	for i := 0; i < b.N; i++ {
		pod := &items[i%width]
		versionedObj, err := scheme.UnsafeConvertToVersion(pod, api.Registry.GroupOrDie(api.GroupName).GroupVersion)
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		if _, err = scheme.UnsafeConvertToVersion(versionedObj, testapi.Default.InternalGroupVersion()); err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
	}
}

func BenchmarkNodeConversion(b *testing.B) {
	data, err := ioutil.ReadFile("node_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var node api.Node
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &node); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	scheme := api.Scheme
	var result *api.Node
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.UnsafeConvertToVersion(&node, api.Registry.GroupOrDie(api.GroupName).GroupVersion)
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.UnsafeConvertToVersion(versionedObj, testapi.Default.InternalGroupVersion())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		result = obj.(*api.Node)
	}
	b.StopTimer()
	if !apiequality.Semantic.DeepDerivative(node, *result) {
		b.Fatalf("Incorrect conversion: %s", diff.ObjectDiff(node, *result))
	}
}

func BenchmarkReplicationControllerConversion(b *testing.B) {
	data, err := ioutil.ReadFile("replication_controller_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var replicationController api.ReplicationController
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &replicationController); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	scheme := api.Scheme
	var result *api.ReplicationController
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.UnsafeConvertToVersion(&replicationController, api.Registry.GroupOrDie(api.GroupName).GroupVersion)
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.UnsafeConvertToVersion(versionedObj, testapi.Default.InternalGroupVersion())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		result = obj.(*api.ReplicationController)
	}
	b.StopTimer()
	if !apiequality.Semantic.DeepDerivative(replicationController, *result) {
		b.Fatalf("Incorrect conversion: expected %v, got %v", replicationController, *result)
	}
}
