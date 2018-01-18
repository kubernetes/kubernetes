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

package validation

import (
	"math/rand"
	"testing"

	"github.com/go-openapi/spec"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/json"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsfuzzer "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

// TestRoundTrip checks the conversion to go-openapi types.
// internal -> go-openapi -> JSON -> external -> internal
func TestRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)

	// add internal and external types to scheme
	if err := apiextensions.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := apiextensionsv1beta1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	seed := rand.Int63()
	fuzzerFuncs := fuzzer.MergeFuzzerFuncs(apiextensionsfuzzer.Funcs)
	f := fuzzer.FuzzerFor(fuzzerFuncs, rand.NewSource(seed), codecs)

	for i := 0; i < 20; i++ {
		// fuzz internal types
		internal := &apiextensions.JSONSchemaProps{}
		f.Fuzz(internal)

		// internal -> go-openapi
		openAPITypes := &spec.Schema{}
		if err := convertJSONSchemaProps(internal, openAPITypes); err != nil {
			t.Fatal(err)
		}

		// go-openapi -> JSON
		openAPIJSON, err := json.Marshal(openAPITypes)
		if err != nil {
			t.Fatal(err)
		}

		// JSON -> external
		external := &apiextensionsv1beta1.JSONSchemaProps{}
		if err := json.Unmarshal(openAPIJSON, external); err != nil {
			t.Fatal(err)
		}

		// external -> internal
		internalRoundTripped := &apiextensions.JSONSchemaProps{}
		if err := scheme.Convert(external, internalRoundTripped, nil); err != nil {
			t.Fatal(err)
		}

		if !apiequality.Semantic.DeepEqual(internal, internalRoundTripped) {
			t.Fatalf("expected\n\t%#v, got \n\t%#v", internal, internalRoundTripped)
		}
	}
}
