/*
Copyright 2025 The Kubernetes Authors.

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

package scheme

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubectl/pkg/config"
	kubectlfuzzer "k8s.io/kubectl/pkg/config/fuzzer"
	"sigs.k8s.io/randfill"
)

func TestRoundTripTypes(t *testing.T) {
	// Because v1alpha1 does not have fields of these types, the fuzzing will
	// be incorrect, so we need to manually intervene here
	customFuzzerFuncs := func(codecs runtimeserializer.CodecFactory) []interface{} {
		return []interface{}{
			func(s *config.CredentialPluginPolicy, c randfill.Continue) {
				*s = ""
			},
			func(s *[]config.AllowlistEntry, c randfill.Continue) {
				*s = nil
			},
		}
	}

	funcs := fuzzer.MergeFuzzerFuncs(kubectlfuzzer.Funcs, customFuzzerFuncs)
	roundtrip.RoundTripTestForScheme(t, Scheme, funcs)
}
