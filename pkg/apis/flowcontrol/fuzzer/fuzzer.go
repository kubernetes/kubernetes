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
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

// Funcs returns the fuzzer functions for the flowcontrol api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *flowcontrol.ResourcePolicyRule, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			// fuzzing api groups
			if c.RandBool() {
				obj.APIGroups = []string{"*"}
			} else {
				obj.APIGroups = randStringArray(c)
			}
			// fuzzing api resources
			if c.RandBool() {
				obj.Resources = []string{"*"}
			} else {
				obj.Resources = randStringArray(c)
			}
			// fuzzing verbs
			if c.RandBool() {
				obj.Verbs = []string{"*"}
			} else {
				obj.Verbs = randStringArray(c)
			}
			// fuzzing namespaces
			if c.RandBool() {
				obj.ClusterScope = true
			} else {
				obj.ClusterScope = false
				if c.RandBool() {
					obj.Namespaces = []string{"*"}
				} else {
					obj.Namespaces = randStringArray(c)
				}
			}
			// fuzzing resource names
			if c.RandBool() {
				obj.ResourceNames = []string{"*"}
			} else {
				obj.ResourceNames = randStringArray(c)
			}
		},
	}
}

func randStringArray(c fuzz.Continue) []string {
	length := c.Intn(9) + 1 // [1,10)
	ret := make([]string, length)
	for i := 0; i < length; i++ {
		ret[i] = c.RandString()
	}
	return ret
}
