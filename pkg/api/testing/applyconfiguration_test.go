/*
Copyright 2021 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/client-go/applyconfigurations"
	v1mf "k8s.io/client-go/applyconfigurations/core/v1"
	"sigs.k8s.io/randfill"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// TestUnstructuredRoundTripApplyConfigurations converts each known object type through unstructured
// to the apply configuration for that object type, then converts it back to the object type and
// verifies it is unchanged.
func TestUnstructuredRoundTripApplyConfigurations(t *testing.T) {
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if nonRoundTrippableTypes.Has(gvk.Kind) {
			continue
		}
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		if builder := applyconfigurations.ForKind(gvk); builder == nil {
			continue
		}

		t.Run(gvk.String(), func(t *testing.T) {
			for i := 0; i < 3; i++ {
				item := fuzzObject(t, gvk)
				builder := applyconfigurations.ForKind(gvk)
				unstructuredRoundTripApplyConfiguration(t, item, builder)
				if t.Failed() {
					break
				}
			}
		})
	}
}

// TestJsonRoundTripApplyConfigurations converts each known object type through JSON to the apply
// configuration for that object type, then converts it back to the object type and verifies it
// is unchanged.
func TestJsonRoundTripApplyConfigurations(t *testing.T) {
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if nonRoundTrippableTypes.Has(gvk.Kind) {
			continue
		}
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		if builder := applyconfigurations.ForKind(gvk); builder == nil {
			continue
		}

		t.Run(gvk.String(), func(t *testing.T) {
			for i := 0; i < 3; i++ {
				item := fuzzObject(t, gvk)
				builder := applyconfigurations.ForKind(gvk)
				jsonRoundTripApplyConfiguration(t, item, builder)
				if t.Failed() {
					break
				}

			}
		})
	}
}

func unstructuredRoundTripApplyConfiguration(t *testing.T, item runtime.Object, applyConfig interface{}) {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(item)
	if err != nil {
		t.Errorf("ToUnstructured failed: %v", err)
		return
	}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(u, applyConfig)
	if err != nil {
		t.Errorf("FromUnstructured failed: %v", err)
		return
	}
	rtObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	u, err = runtime.DefaultUnstructuredConverter.ToUnstructured(applyConfig)
	if err != nil {
		t.Errorf("ToUnstructured failed: %v", err)
		return
	}
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(u, rtObj)
	if err != nil {
		t.Errorf("FromUnstructured failed: %v", err)
		return
	}
	if !apiequality.Semantic.DeepEqual(item, rtObj) {
		t.Errorf("Object changed, diff: %v", cmp.Diff(item, rtObj))
	}
}

func jsonRoundTripApplyConfiguration(t *testing.T, item runtime.Object, applyConfig interface{}) {

	objData, err := json.Marshal(item)
	if err != nil {
		t.Errorf("json.Marshal failed: %v", err)
		return
	}
	err = json.Unmarshal(objData, applyConfig)
	if err != nil {
		t.Errorf("applyConfig.UnmarshalJSON failed: %v", err)
		return
	}
	rtObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	applyData, err := json.Marshal(applyConfig)
	if err != nil {
		t.Errorf("applyConfig.MarshalJSON failed: %v", err)
		return
	}
	err = json.Unmarshal(applyData, rtObj)
	if err != nil {
		t.Errorf("json.Unmarshal failed: %v", err)
		return
	}
	if !apiequality.Semantic.DeepEqual(item, rtObj) {
		t.Errorf("Object changed, diff: %v", cmp.Diff(item, rtObj))
	}
}

func fuzzObject(t *testing.T, gvk schema.GroupVersionKind) runtime.Object {
	internalVersion := schema.GroupVersion{Group: gvk.Group, Version: runtime.APIVersionInternal}
	externalVersion := gvk.GroupVersion()
	kind := gvk.Kind

	// We do fuzzing on the internal version of the object, and only then
	// convert to the external version. This is because custom fuzzing
	// function are only supported for internal objects.
	internalObj, err := legacyscheme.Scheme.New(internalVersion.WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't create internal object %v: %v", kind, err)
	}
	seed := rand.Int63()
	fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(seed), legacyscheme.Codecs).
		Funcs(
			// Ensure that InitContainers and their statuses are not generated. This
			// is because in this test we are simply doing json operations, in which
			// those disappear.
			func(s *api.PodSpec, c randfill.Continue) {
				c.FillNoCustom(s)
				s.InitContainers = nil
			},
			func(s *api.PodStatus, c randfill.Continue) {
				c.FillNoCustom(s)
				s.InitContainerStatuses = nil
			},
			// Apply configuration types do not have managed fields, so we exclude
			// them in our fuzz test cases.
			func(s *v1.ObjectMeta, c randfill.Continue) {
				c.FillNoCustom(s)
				s.ManagedFields = nil
				s.SelfLink = ""
			},
		).Fill(internalObj)

	item, err := legacyscheme.Scheme.New(externalVersion.WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't create external object %v: %v", kind, err)
	}
	if err := legacyscheme.Scheme.Convert(internalObj, item, nil); err != nil {
		t.Fatalf("Conversion for %v failed: %v", kind, err)
	}
	return item
}

func BenchmarkApplyConfigurationsFromUnstructured(b *testing.B) {
	items := benchmarkItems(b)
	convertor := runtime.DefaultUnstructuredConverter
	unstr := make([]map[string]interface{}, len(items))
	for i := range items {
		item, err := convertor.ToUnstructured(&items[i])
		if err != nil || item == nil {
			b.Fatalf("unexpected error: %v", err)
		}
		unstr = append(unstr, item)
	}
	size := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builder := &v1mf.PodApplyConfiguration{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstr[i%size], builder); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}

func BenchmarkApplyConfigurationsToUnstructured(b *testing.B) {
	items := benchmarkItems(b)
	convertor := runtime.DefaultUnstructuredConverter
	builders := make([]*v1mf.PodApplyConfiguration, len(items))
	for i := range items {
		item, err := convertor.ToUnstructured(&items[i])
		if err != nil || item == nil {
			b.Fatalf("unexpected error: %v", err)
		}
		builder := &v1mf.PodApplyConfiguration{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(item, builder); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
		builders[i] = builder
	}
	b.ResetTimer()
	size := len(items)
	for i := 0; i < b.N; i++ {
		builder := builders[i%size]
		if _, err := runtime.DefaultUnstructuredConverter.ToUnstructured(builder); err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
	b.StopTimer()
}
