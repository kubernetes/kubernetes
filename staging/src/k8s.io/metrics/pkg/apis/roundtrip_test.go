/*
Copyright 2020 The Kubernetes Authors.

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
	"math/rand"
	"testing"

	custommetrics "k8s.io/metrics/pkg/apis/custom_metrics"
	custommetricsv1beta1 "k8s.io/metrics/pkg/apis/custom_metrics/v1beta1"
	custommetricsv1beta2 "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	externalmetrics "k8s.io/metrics/pkg/apis/external_metrics"
	externalmetricsv1beta1 "k8s.io/metrics/pkg/apis/external_metrics/v1beta1"
	metrics "k8s.io/metrics/pkg/apis/metrics"
	metricsv1alpha1 "k8s.io/metrics/pkg/apis/metrics/v1alpha1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	genericfuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

var groups = []runtime.SchemeBuilder{
	custommetrics.SchemeBuilder,
	custommetricsv1beta1.SchemeBuilder,
	custommetricsv1beta2.SchemeBuilder,
	externalmetrics.SchemeBuilder,
	externalmetricsv1beta1.SchemeBuilder,
	metrics.SchemeBuilder,
	metricsv1alpha1.SchemeBuilder,
	metricsv1beta1.SchemeBuilder,
}

func TestRoundTripTypes(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	for _, builder := range groups {
		require.NoError(t, builder.AddToScheme(scheme))
	}
	seed := rand.Int63()
	// I'm only using the generic fuzzer funcs, but at some point in time we might need to
	// switch to specialized. For now we're happy with the current serialization test.
	fuzzer := fuzzer.FuzzerFor(genericfuzzer.Funcs, rand.NewSource(seed), codecs)

	roundtrip.RoundTripExternalTypes(t, scheme, codecs, fuzzer, nil)
	roundtrip.RoundTripTypes(t, scheme, codecs, fuzzer, nil)
}
