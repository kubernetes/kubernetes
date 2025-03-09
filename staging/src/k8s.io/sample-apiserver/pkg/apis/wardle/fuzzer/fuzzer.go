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
	"k8s.io/sample-apiserver/pkg/apis/wardle"
	"sigs.k8s.io/randfill"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
)

// Funcs returns the fuzzer functions for the apps api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(s *wardle.FlunderSpec, c randfill.Continue) {
			c.FillNoCustom(s) // fuzz self without calling this function again

			if len(s.FlunderReference) != 0 && len(s.FischerReference) != 0 {
				s.FischerReference = ""
			}
			if len(s.FlunderReference) != 0 {
				s.ReferenceType = wardle.FlunderReferenceType
			} else if len(s.FischerReference) != 0 {
				s.ReferenceType = wardle.FischerReferenceType
			} else {
				s.ReferenceType = ""
			}
		},
	}
}
