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

package meta

import (
	"math/rand"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/util/diff"

	fuzz "github.com/google/gofuzz"
)

func TestAsPartialObjectMetadata(t *testing.T) {
	f := fuzz.New().NilChance(.5).NumElements(0, 1).RandSource(rand.NewSource(1))

	for i := 0; i < 100; i++ {
		m := &metav1.ObjectMeta{}
		f.Fuzz(m)
		partial := AsPartialObjectMetadata(m)
		if !reflect.DeepEqual(&partial.ObjectMeta, m) {
			t.Fatalf("incomplete partial object metadata: %s", diff.ObjectReflectDiff(&partial.ObjectMeta, m))
		}
	}

	for i := 0; i < 100; i++ {
		m := &metav1beta1.PartialObjectMetadata{}
		f.Fuzz(&m.ObjectMeta)
		partial := AsPartialObjectMetadata(m)
		if !reflect.DeepEqual(&partial.ObjectMeta, &m.ObjectMeta) {
			t.Fatalf("incomplete partial object metadata: %s", diff.ObjectReflectDiff(&partial.ObjectMeta, &m.ObjectMeta))
		}
	}
}
