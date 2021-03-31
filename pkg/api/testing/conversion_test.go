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

package testing

import (
	"io/ioutil"
	"math/rand"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func BenchmarkPodConversion(b *testing.B) {
	apiObjectFuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(benchmarkSeed), legacyscheme.Codecs)
	items := make([]api.Pod, 4)
	for i := range items {
		apiObjectFuzzer.Fuzz(&items[i])
		items[i].Spec.InitContainers = nil
		items[i].Status.InitContainerStatuses = nil
	}

	// add a fixed item
	items = append(items, benchmarkPod)
	width := len(items)

	scheme := legacyscheme.Scheme
	for i := 0; i < b.N; i++ {
		pod := &items[i%width]
		versionedObj, err := scheme.UnsafeConvertToVersion(pod, schema.GroupVersion{Group: "", Version: "v1"})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		if _, err = scheme.UnsafeConvertToVersion(versionedObj, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal}); err != nil {
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
	if err := runtime.DecodeInto(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), data, &node); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	scheme := legacyscheme.Scheme
	var result *api.Node
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.UnsafeConvertToVersion(&node, schema.GroupVersion{Group: "", Version: "v1"})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.UnsafeConvertToVersion(versionedObj, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal})
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
	if err := runtime.DecodeInto(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), data, &replicationController); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	scheme := legacyscheme.Scheme
	var result *api.ReplicationController
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.UnsafeConvertToVersion(&replicationController, schema.GroupVersion{Group: "", Version: "v1"})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.UnsafeConvertToVersion(versionedObj, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal})
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
