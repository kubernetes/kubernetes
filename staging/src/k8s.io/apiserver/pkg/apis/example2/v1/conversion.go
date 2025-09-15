/*
Copyright 2017 The Kubernetes Authors.

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

package v1

import (
	conversion "k8s.io/apimachinery/pkg/conversion"
	example "k8s.io/apiserver/pkg/apis/example"
)

func Convert_example_ReplicaSetSpec_To_v1_ReplicaSetSpec(in *example.ReplicaSetSpec, out *ReplicaSetSpec, s conversion.Scope) error {
	out.Replicas = new(int32)
	*out.Replicas = int32(in.Replicas)
	return nil
}

func Convert_v1_ReplicaSetSpec_To_example_ReplicaSetSpec(in *ReplicaSetSpec, out *example.ReplicaSetSpec, s conversion.Scope) error {
	if in.Replicas != nil {
		out.Replicas = *in.Replicas
	}
	return nil
}
