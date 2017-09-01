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

package fuzzer

import (
	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/federation/apis/federation"
	federationv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
)

// Funcs returns the fuzzer functions for the extensions api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(o *federation.Cluster, c fuzz.Continue) {
			c.FuzzNoCustom(o) // fuzz self without calling this function again
			if o.Labels == nil {
				o.Labels = map[string]string{}
			}
			if _, ok := o.Labels[federationv1.FederationClusterNameLabel]; !ok {
				o.Labels[federationv1.FederationClusterNameLabel] = o.Name
			}
		},
	}
}
