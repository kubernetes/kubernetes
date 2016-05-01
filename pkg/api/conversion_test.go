/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
)

func benchmarkConversionRoundTrip(b *testing.B, items []runtime.Object) {
	width := len(items)
	scheme := api.Scheme
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.ConvertToVersion(items[i%width], testapi.Default.GroupVersion().String())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		if _, err = scheme.ConvertToVersion(versionedObj, testapi.Default.InternalGroupVersion().String()); err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
	}
	b.StopTimer()
}

func BenchmarkPodConversion(b *testing.B) {
	apiObjectFuzzer := apitesting.FuzzerFor(nil, api.SchemeGroupVersion, rand.NewSource(benchmarkSeed))
	items := make([]runtime.Object, 4)
	for i := range items {
		items[i] = &api.Pod{}
		apiObjectFuzzer.Fuzz(items[i])
	}
	// add a fixed item
	data, err := ioutil.ReadFile("pod_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var pod api.Pod
	if err := runtime.DecodeInto(testapi.Default.Codec(), data, &pod); err != nil {
		b.Fatalf("Unexpected error decoding pod: %v", err)
	}
	items = append(items, &pod)

	benchmarkConversionRoundTrip(b, items)
}

func fuzzPodList(width, variants int) []runtime.Object {
	apiObjectFuzzer := apitesting.FuzzerFor(nil, api.SchemeGroupVersion, rand.NewSource(benchmarkSeed))
	items := make([]runtime.Object, variants)
	for i := range items {
		list := &api.PodList{}
		apiObjectFuzzer.Fuzz(list)
		list.Items = make([]api.Pod, width)
		for j := range list.Items {
			apiObjectFuzzer.Fuzz(&list.Items[j])
		}
		items[i] = list
	}
	return items
}

func benchmarkPodListConversion(b *testing.B, width int) {
	items := fuzzPodList(width, 4)
	benchmarkConversionRoundTrip(b, items)
}

func BenchmarkPodListConversion100(b *testing.B) { benchmarkPodListConversion(b, 100) }

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
		versionedObj, err := scheme.ConvertToVersion(&node, testapi.Default.GroupVersion().String())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.ConvertToVersion(versionedObj, testapi.Default.InternalGroupVersion().String())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		result = obj.(*api.Node)
	}
	b.StopTimer()
	if !api.Semantic.DeepDerivative(node, *result) {
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
		versionedObj, err := scheme.ConvertToVersion(&replicationController, testapi.Default.GroupVersion().String())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.ConvertToVersion(versionedObj, testapi.Default.InternalGroupVersion().String())
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		result = obj.(*api.ReplicationController)
	}
	b.StopTimer()
	if !api.Semantic.DeepDerivative(replicationController, *result) {
		b.Fatalf("Incorrect conversion: expected %v, got %v", replicationController, *result)
	}
}
