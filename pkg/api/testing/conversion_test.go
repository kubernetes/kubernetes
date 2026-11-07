/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func BenchmarkPodConversion(b *testing.B) {
	apiObjectFuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(benchmarkSeed), legacyscheme.Codecs)
	items := make([]api.Pod, 4)
	for i := range items {
		apiObjectFuzzer.Fill(&items[i])
		items[i].Spec.InitContainers = nil
		items[i].Status.InitContainerStatuses = nil
	}

	// add a fixed item
	items = append(items, benchmarkPod)
	width := len(items)

	scheme := legacyscheme.Scheme
	for i := 0; i < b.N; i++ {
		pod := &items[i%width]
		versionedObj, err := scheme.UnsafeConvertToVersion(pod, schema.GroupVersion{Group: "", Version: "v1"})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		if _, err = scheme.UnsafeConvertToVersion(versionedObj, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal}); err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
	}
}

func BenchmarkNodeConversion(b *testing.B) {
	data, err := os.ReadFile("node_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var node api.Node
	if err := runtime.DecodeInto(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), data, &node); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	scheme := legacyscheme.Scheme
	var result *api.Node
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.UnsafeConvertToVersion(&node, schema.GroupVersion{Group: "", Version: "v1"})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.UnsafeConvertToVersion(versionedObj, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		result = obj.(*api.Node)
	}
	b.StopTimer()
	if !apiequality.Semantic.DeepDerivative(node, *result) {
		b.Fatalf("Incorrect conversion: %s", cmp.Diff(node, *result))
	}
}

func BenchmarkReplicationControllerConversion(b *testing.B) {
	data, err := os.ReadFile("replication_controller_example.json")
	if err != nil {
		b.Fatalf("Unexpected error while reading file: %v", err)
	}
	var replicationController api.ReplicationController
	if err := runtime.DecodeInto(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), data, &replicationController); err != nil {
		b.Fatalf("Unexpected error decoding node: %v", err)
	}

	scheme := legacyscheme.Scheme
	var result *api.ReplicationController
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		versionedObj, err := scheme.UnsafeConvertToVersion(&replicationController, schema.GroupVersion{Group: "", Version: "v1"})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		obj, err := scheme.UnsafeConvertToVersion(versionedObj, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal})
		if err != nil {
			b.Fatalf("Conversion error: %v", err)
		}
		result = obj.(*api.ReplicationController)
	}
	b.StopTimer()
	if !apiequality.Semantic.DeepDerivative(replicationController, *result) {
		b.Fatalf("Incorrect conversion: expected %v, got %v", replicationController, *result)
	}
}

// checkMemoryIdentical recursively checks if two reflect.Values are memory identical
// and returns a list of differences.
//
// NOTE: this does not support recursive types. If the types have recursive fields,
// this function will not terminate.
func checkMemoryIdentical(path string, a, b reflect.Value) []string {
	if a.Kind() != b.Kind() {
		return []string{fmt.Sprintf("%s: kind mismatch: %s vs %s", path, a.Kind(), b.Kind())}
	}
	var diffs []string
	switch a.Kind() {
	case reflect.Struct:
		if a.Type().Size() != b.Type().Size() {
			return []string{fmt.Sprintf("%s: struct size mismatch: %d vs %d", path, a.Type().Size(), b.Type().Size())}
		}
		if a.NumField() != b.NumField() {
			return []string{fmt.Sprintf("%s: field count mismatch: %d vs %d", path, a.NumField(), b.NumField())}
		}
		for i := 0; i < a.NumField(); i++ {
			aTypeField := a.Type().Field(i)
			bTypeField := b.Type().Field(i)
			if aTypeField.Name != bTypeField.Name {
				diffs = append(diffs, fmt.Sprintf("%s: field name mismatch: %s vs %s", path, aTypeField.Name, bTypeField.Name))
			}
			if aTypeField.Offset != bTypeField.Offset {
				diffs = append(diffs, fmt.Sprintf("%s.%s: field offset mismatch: %d vs %d", path, aTypeField.Name, aTypeField.Offset, bTypeField.Offset))
			}
			diffs = append(diffs, checkMemoryIdentical(path+"."+aTypeField.Name, a.Field(i), b.Field(i))...)
		}
	case reflect.Pointer, reflect.Map, reflect.Slice:
		if a.IsNil() != b.IsNil() {
			return []string{fmt.Sprintf("%s: nil mismatch: %v vs %v", path, a.IsNil(), b.IsNil())}
		}
		if !a.IsNil() && a.UnsafePointer() != b.UnsafePointer() {
			return []string{fmt.Sprintf("%s: nilable type (%s) was copied", path, a.Kind())}
		}
	case reflect.Interface:
		if a.IsNil() != b.IsNil() {
			return []string{fmt.Sprintf("%s: nil interface mismatch: %v vs %v", path, a.IsNil(), b.IsNil())}
		}
		if !a.IsNil() {
			diffs = append(diffs, checkMemoryIdentical(path, a.Elem(), b.Elem())...)
		}
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.String:
		// Assume scalars are copied by value
	default:
		diffs = append(diffs, fmt.Sprintf("%s: unexpected kind: %v", path, a.Kind()))
	}
	return diffs
}

func checkVersionMemoryIdentical(t *testing.T, scheme *runtime.Scheme, fuzzerFiller interface{ Fill(interface{}) }, internalGVK, extGVK schema.GroupVersionKind) []string {
	var diffs []string

	extIn, err := scheme.New(extGVK)
	if err != nil {
		t.Fatalf("failed to create %v: %v", extGVK, err)
	}
	fuzzerFiller.Fill(extIn)
	intOut, err := scheme.New(internalGVK)
	if err != nil {
		t.Fatalf("failed to create %v: %v", internalGVK, err)
	}
	if err := scheme.Convert(extIn, intOut, nil); err != nil {
		diffs = append(diffs, fmt.Sprintf("%s->internal convert error: %v", extGVK.Version, err))
	} else {
		diffs = append(diffs, checkMemoryIdentical(extGVK.Version+"->internal", reflect.ValueOf(extIn).Elem(), reflect.ValueOf(intOut).Elem())...)
	}

	intIn, err := scheme.New(internalGVK)
	if err != nil {
		t.Fatalf("failed to create %v: %v", internalGVK, err)
	}
	fuzzerFiller.Fill(intIn)
	extOut, err := scheme.New(extGVK)
	if err != nil {
		t.Fatalf("failed to create %v: %v", extGVK, err)
	}
	if err := scheme.Convert(intIn, extOut, nil); err != nil {
		diffs = append(diffs, fmt.Sprintf("internal->%s convert error: %v", extGVK.Version, err))
	} else {
		diffs = append(diffs, checkMemoryIdentical("internal->"+extGVK.Version, reflect.ValueOf(intIn).Elem(), reflect.ValueOf(extOut).Elem())...)
	}

	return diffs
}

// TestMemoryIdenticalConversion ensures that all internal API types have memory-identical
// conversion (including their corresponding List type, if any) with their highest-priority
// external version, unless explicitly exempted.
func TestMemoryIdenticalConversion(t *testing.T) {
	exempt := sets.New[string](
		// Remaining APIs to migrate to memory-identical internal types:
		"AdmissionReview.admission.k8s.io",
		"DaemonSet.apps",
		"Deployment.apps",
		"Event.events.k8s.io",
		"HorizontalPodAutoscaler.autoscaling",
		"Node",
		"PodCertificateRequest.certificates.k8s.io",
		"PriorityLevelConfiguration.flowcontrol.apiserver.k8s.io",
		"ReplicaSet.apps",
		"Secret",
		"StatefulSet.apps",

		// Legacy unserved groups/types that share internal hub types with newer groups:
		"DaemonSet.extensions",
		"Deployment.extensions",
		"Ingress.extensions",
		"NetworkPolicy.extensions",
		"ReplicaSet.extensions",
		"Scale.apps",
		"Scale.extensions",

		// These types are not considered memory-identical right now by conversion-gen, although they are,
		// as we need to define Convert_v1_ConditionsAwareDecision_To_authorization_ConditionsAwareDecision (and vice versa)
		// function to make ConditionsAwareDecision possible to vendor.
		// TODO: Remove this when https://github.com/kubernetes/kubernetes/issues/142345 is fixed
		"SubjectAccessReview.authorization.k8s.io",
		"SelfSubjectAccessReview.authorization.k8s.io",
		"LocalSubjectAccessReview.authorization.k8s.io",
		// Embeds admission.AdmissionRequest, whose internal runtime.Object fields are converted from runtime.RawExtension:
		"AuthorizationConditionsReview.authorization.k8s.io",

		// Generic meta-list type (metainternalversion.List vs metav1.List):
		"List",
	)

	scheme := legacyscheme.Scheme
	f := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(1), legacyscheme.Codecs).NilChance(0).NumElements(1, 1)

	allKnown := scheme.AllKnownTypes()
	var internalGVKs []schema.GroupVersionKind
	for gvk := range allKnown {
		if gvk.Version != runtime.APIVersionInternal {
			continue
		}
		if roundtrip.GlobalNonRoundTrippableTypes().Has(gvk.Kind) {
			continue
		}
		if baseKind, isList := strings.CutSuffix(gvk.Kind, "List"); isList && baseKind != "" {
			if _, hasBase := allKnown[gvk.GroupVersion().WithKind(baseKind)]; hasBase {
				// Tested alongside its singular resource Kind below.
				continue
			}
		}
		internalGVKs = append(internalGVKs, gvk)
	}
	sort.Slice(internalGVKs, func(i, j int) bool {
		return internalGVKs[i].GroupKind().String() < internalGVKs[j].GroupKind().String()
	})

	for _, internalGVK := range internalGVKs {
		var extGVK schema.GroupVersionKind
		for _, gv := range scheme.PrioritizedVersionsForGroup(internalGVK.Group) {
			candidate := gv.WithKind(internalGVK.Kind)
			if _, ok := allKnown[candidate]; ok {
				extGVK = candidate
				break
			}
		}
		if extGVK.Empty() {
			continue
		}

		gkName := internalGVK.GroupKind().String()
		t.Run(gkName, func(t *testing.T) {
			diffs := checkVersionMemoryIdentical(t, scheme, f, internalGVK, extGVK)

			internalListGVK := internalGVK.GroupVersion().WithKind(internalGVK.Kind + "List")
			extListGVK := extGVK.GroupVersion().WithKind(extGVK.Kind + "List")
			if _, hasIntList := allKnown[internalListGVK]; hasIntList {
				if _, hasExtList := allKnown[extListGVK]; hasExtList {
					listDiffs := checkVersionMemoryIdentical(t, scheme, f, internalListGVK, extListGVK)
					for _, d := range listDiffs {
						diffs = append(diffs, "List."+d)
					}
				}
			}

			conforms := len(diffs) == 0
			if exempt.Has(gkName) {
				if conforms {
					t.Errorf("%s: unexpectedly has memory-identical conversion with %s. Remove it from the exempt list.", gkName, extGVK.Version)
				}
			} else if !conforms {
				t.Errorf("%s: does not have memory-identical conversion with %s:\n  %s", gkName, extGVK.Version, strings.Join(diffs, "\n  "))
			}
		})
	}
}
