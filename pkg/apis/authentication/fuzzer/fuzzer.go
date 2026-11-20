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
	"sigs.k8s.io/randfill"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/authentication"
)

// Funcs returns the fuzzer functions for the authentication api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *authentication.TokenRequestSpec, c randfill.Continue) {
			c.FillNoCustom(obj) // fuzz self without calling this function again

			// ExpirationSeconds has a v1 defaulter that populates 3600 when nil,
			// so ensure it is non-nil to prevent TestRoundTripTypes diffs.
			if obj.ExpirationSeconds == nil {
				obj.ExpirationSeconds = new(int64(60 * 60))
			}
		},
	}
}
