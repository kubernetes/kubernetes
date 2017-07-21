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
	"bytes"
	"math/rand"
	"reflect"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/testing/fuzzer"
	"k8s.io/apimachinery/pkg/api/testing/roundtrip"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
)

func TestDeepCopyApiObjects(t *testing.T) {
	for i := 0; i < *roundtrip.FuzzIters; i++ {
		for _, version := range []schema.GroupVersion{testapi.Default.InternalGroupVersion(), api.Registry.GroupOrDie(api.GroupName).GroupVersion} {
			f := fuzzer.FuzzerFor(kapitesting.FuzzerFuncs, rand.NewSource(rand.Int63()), api.Codecs)
			for kind := range api.Scheme.KnownTypes(version) {
				doDeepCopyTest(t, version.WithKind(kind), f)
			}
		}
	}
}

func doDeepCopyTest(t *testing.T, kind schema.GroupVersionKind, f *fuzz.Fuzzer) {
	item, err := api.Scheme.New(kind)
	if err != nil {
		t.Fatalf("Could not create a %v: %s", kind, err)
	}
	f.Fuzz(item)
	itemCopy, err := api.Scheme.DeepCopy(item)
	if err != nil {
		t.Errorf("Could not deep copy a %v: %s", kind, err)
		return
	}

	if !reflect.DeepEqual(item, itemCopy) {
		t.Errorf("\nexpected: %#v\n\ngot:      %#v\n\ndiff:      %v", item, itemCopy, diff.ObjectReflectDiff(item, itemCopy))
	}

	prefuzzData := &bytes.Buffer{}
	if err := api.Codecs.LegacyCodec(kind.GroupVersion()).Encode(item, prefuzzData); err != nil {
		t.Errorf("Could not encode a %v: %s", kind, err)
		return
	}

	// Refuzz the copy, which should have no effect on the original
	f.Fuzz(itemCopy)

	postfuzzData := &bytes.Buffer{}
	if err := api.Codecs.LegacyCodec(kind.GroupVersion()).Encode(item, postfuzzData); err != nil {
		t.Errorf("Could not encode a %v: %s", kind, err)
		return
	}

	if bytes.Compare(prefuzzData.Bytes(), postfuzzData.Bytes()) != 0 {
		t.Log(diff.StringDiff(prefuzzData.String(), postfuzzData.String()))
		t.Errorf("Fuzzing copy modified original of %#v", kind)
		return
	}
}

func TestDeepCopySingleType(t *testing.T) {
	for i := 0; i < *roundtrip.FuzzIters; i++ {
		for _, version := range []schema.GroupVersion{testapi.Default.InternalGroupVersion(), api.Registry.GroupOrDie(api.GroupName).GroupVersion} {
			f := fuzzer.FuzzerFor(kapitesting.FuzzerFuncs, rand.NewSource(rand.Int63()), api.Codecs)
			doDeepCopyTest(t, version.WithKind("Pod"), f)
		}
	}
}
