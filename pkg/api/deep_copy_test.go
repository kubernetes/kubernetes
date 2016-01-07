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
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

func BenchmarkPodCopy(b *testing.B) {
	data, err := ioutil.ReadFile("pod_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var pod api.Pod
	if err := api.Scheme.DecodeInto(data, &pod); err != nil {
		b.Fatalf("Unexpected error decoding pod: %v", err)
	}

	var result *api.Pod
	for i := 0; i < b.N; i++ {
		obj, err := api.Scheme.DeepCopy(&pod)
		if err != nil {
			b.Fatalf("Unexpected error copying pod: %v", err)
		}
		result = obj.(*api.Pod)
	}
	if !api.Semantic.DeepEqual(pod, *result) {
		b.Fatalf("Incorrect copy: expected %v, got %v", pod, *result)
	}
}

func BenchmarkNodeCopy(b *testing.B) {
	data, err := ioutil.ReadFile("node_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var node api.Node
	if err := api.Scheme.DecodeInto(data, &node); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	var result *api.Node
	for i := 0; i < b.N; i++ {
		obj, err := api.Scheme.DeepCopy(&node)
		if err != nil {
			b.Fatalf("Unexpected error copying node: %v", err)
		}
		result = obj.(*api.Node)
	}
	if !api.Semantic.DeepEqual(node, *result) {
		b.Fatalf("Incorrect copy: expected %v, got %v", node, *result)
	}
}

func BenchmarkReplicationControllerCopy(b *testing.B) {
	data, err := ioutil.ReadFile("replication_controller_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var replicationController api.ReplicationController
	if err := api.Scheme.DecodeInto(data, &replicationController); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	var result *api.ReplicationController
	for i := 0; i < b.N; i++ {
		obj, err := api.Scheme.DeepCopy(&replicationController)
		if err != nil {
			b.Fatalf("Unexpected error copying replication controller: %v", err)
		}
		result = obj.(*api.ReplicationController)
	}
	if !api.Semantic.DeepEqual(replicationController, *result) {
		b.Fatalf("Incorrect copy: expected %v, got %v", replicationController, *result)
	}
}
