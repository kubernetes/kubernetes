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

package api_test

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
)

type orderedGroupVersionKinds []unversioned.GroupVersionKind

func (o orderedGroupVersionKinds) Len() int      { return len(o) }
func (o orderedGroupVersionKinds) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o orderedGroupVersionKinds) Less(i, j int) bool {
	return o[i].String() < o[j].String()
}

// TODO: add a reflexive test that verifies that all SetDefaults functions are registered

// TODO: once we remove defaulting from conversion, convert this test to ensuring
// that all objects that say they have defaulting are verified to mutate the originating
// object.
func TestDefaulting(t *testing.T) {
	f := fuzz.New().NilChance(.5).NumElements(1, 1)
	f.RandSource(rand.NewSource(1))
	f.Funcs(
		func(s *runtime.RawExtension, c fuzz.Continue) {},
		func(s *unversioned.LabelSelector, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.MatchExpressions = nil // need to fuzz this specially
		},
		func(s *apiv1.ListOptions, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.LabelSelector = "" // need to fuzz requirement strings specially
			s.FieldSelector = "" // need to fuzz requirement strings specially
		},
		// No longer necessary when we remove defaulting from conversion
		func(s *apiv1.Secret, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.StringData = nil // is mapped into Data, which cannot easily be tested
		},
		func(s *extensionsv1beta1.ListOptions, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.LabelSelector = "" // need to fuzz requirement strings specially
			s.FieldSelector = "" // need to fuzz requirement strings specially
		},
		func(s *extensionsv1beta1.ScaleStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.TargetSelector = "" // need to fuzz requirement strings specially
		},
	)

	scheme := api.Scheme
	var testTypes orderedGroupVersionKinds
	for gvk := range scheme.AllKnownTypes() {
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		testTypes = append(testTypes, gvk)
	}
	sort.Sort(testTypes)

	for _, gvk := range testTypes {
		for i := 0; i < *fuzzIters; i++ {
			src, err := scheme.New(gvk)
			if err != nil {
				t.Fatal(err)
			}
			f.Fuzz(src)

			src.GetObjectKind().SetGroupVersionKind(unversioned.GroupVersionKind{})

			original, _ := scheme.DeepCopy(src)

			// get internal
			copied, _ := scheme.DeepCopy(src)
			scheme.Default(copied.(runtime.Object))

			// get expected
			// TODO: this relies on the side effect behavior of defaulters applying to the external
			// object
			if _, err = scheme.UnsafeConvertToVersion(src, runtime.InternalGroupVersioner); err != nil {
				t.Errorf("[%v] unable to convert: %v", gvk, err)
				continue
			}
			src.GetObjectKind().SetGroupVersionKind(unversioned.GroupVersionKind{})

			existingChanged := !reflect.DeepEqual(original, src)
			newChanged := !reflect.DeepEqual(original, copied)
			if existingChanged != newChanged {
				t.Errorf("[%v] mismatched changes: old=%t new=%t", gvk, existingChanged, newChanged)
			}

			if !reflect.DeepEqual(src, copied) {
				t.Errorf("[%v] changed: %s", gvk, diff.ObjectReflectDiff(src, copied))
			}
		}
	}
}
