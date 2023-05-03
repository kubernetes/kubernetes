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

package apiserver

import (
	"context"
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	k8sfuzz "k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	k8stest "k8s.io/kubernetes/pkg/api/testing"
)

func convertToUnstructured(b *testing.B, obj runtime.Object) runtime.Object {
	converter := managedfields.NewDeducedTypeConverter()
	typed, err := converter.ObjectToTyped(obj)
	require.NoError(b, err)
	res, err := converter.TypedToObject(typed)
	require.NoError(b, err)
	return res
}

func doBench(b *testing.B, useUnstructured bool, shortCircuit bool) {
	var (
		expectedLarge runtime.Object
		actualLarge   runtime.Object
		expectedSmall runtime.Object
		actualSmall   runtime.Object
	)

	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	seed := rand.Int63()
	fuzzer := k8sfuzz.FuzzerFor(k8stest.FuzzerFuncs, rand.NewSource(seed), codecs)
	fuzzer.NilChance(0)

	fuzzer.MaxDepth(1000).NilChance(0.2).NumElements(2, 15)
	pod := &v1.Pod{}
	fuzzer.Fuzz(pod)

	fuzzer.NilChance(0.2).NumElements(10, 100).MaxDepth(10)
	deployment := &v1.Endpoints{}
	fuzzer.Fuzz(deployment)

	bts, err := json.Marshal(deployment)
	require.NoError(b, err)
	b.Logf("Small (Deployment): %v bytes", len(bts))
	bts, err = json.Marshal(pod)
	require.NoError(b, err)
	b.Logf("Large (Pod): %v bytes", len(bts))

	expectedLarge = deployment
	expectedSmall = pod

	if useUnstructured {
		expectedSmall = convertToUnstructured(b, expectedSmall)
		expectedLarge = convertToUnstructured(b, expectedLarge)
	}

	actualLarge = expectedLarge.DeepCopyObject()
	actualSmall = expectedSmall.DeepCopyObject()

	if shortCircuit {
		// Modify managed fields of the compared objects to induce a short circuit
		now := metav1.Now()
		extraEntry := &metav1.ManagedFieldsEntry{
			Manager:    "sidecar_controller",
			Operation:  metav1.ManagedFieldsOperationApply,
			APIVersion: "apps/v1",
			Time:       &now,
			FieldsType: "FieldsV1",
			FieldsV1: &metav1.FieldsV1{
				Raw: []byte(`{"f:metadata":{"f:labels":{"f:sidecar_version":{}}},"f:spec":{"f:template":{"f:spec":{"f:containers":{"k:{\"name\":\"sidecar\"}":{".":{},"f:image":{},"f:name":{}}}}}}}`),
			},
		}

		largeMeta, err := meta.Accessor(actualLarge)
		require.NoError(b, err)
		largeMeta.SetManagedFields(append(largeMeta.GetManagedFields(), *extraEntry))

		smallMeta, err := meta.Accessor(actualSmall)
		require.NoError(b, err)
		smallMeta.SetManagedFields(append(smallMeta.GetManagedFields(), *extraEntry))
	}

	b.ResetTimer()

	b.Run("Large", func(b2 *testing.B) {
		for i := 0; i < b2.N; i++ {
			if _, err := fieldmanager.IgnoreManagedFieldsTimestampsTransformer(
				context.TODO(),
				actualLarge,
				expectedLarge,
			); err != nil {
				b2.Fatal(err)
			}
		}
	})

	b.Run("Small", func(b2 *testing.B) {
		for i := 0; i < b2.N; i++ {
			if _, err := fieldmanager.IgnoreManagedFieldsTimestampsTransformer(
				context.TODO(),
				actualSmall,
				expectedSmall,
			); err != nil {
				b2.Fatal(err)
			}
		}
	})
}

func BenchmarkIgnoreManagedFieldsTimestampTransformerStructuredShortCircuit(b *testing.B) {
	doBench(b, false, true)
}

func BenchmarkIgnoreManagedFieldsTimestampTransformerStructuredWorstCase(b *testing.B) {
	doBench(b, false, false)
}

func BenchmarkIgnoreManagedFieldsTimestampTransformerUnstructuredShortCircuit(b *testing.B) {
	doBench(b, true, true)
}

func BenchmarkIgnoreManagedFieldsTimestampTransformerUnstructuredWorstCase(b *testing.B) {
	doBench(b, true, false)
}
