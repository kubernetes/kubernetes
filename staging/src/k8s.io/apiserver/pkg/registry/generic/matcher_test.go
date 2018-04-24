/*
Copyright 2018 The Kubernetes Authors.

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

package generic

import (
	"reflect"
	"testing"

	fuzz "github.com/google/gofuzz"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestResetObjectMetaForStatus(t *testing.T) {
	meta := &metav1.ObjectMeta{}
	existingMeta := &metav1.ObjectMeta{}

	// fuzz the existingMeta to set every field, no nils
	f := fuzz.New().NilChance(0).NumElements(1, 1)
	f.Fuzz(existingMeta)

	ResetObjectMetaForStatus(meta, existingMeta)

	// Not all fields are stomped during the reset. These fields should not have been set.
	// Set them all to their zero values. Before you add anything to this list, consider whether or not
	// you're enforcing immutability (those are fine) and whether /status should be able to update
	// these values (these are usually not fine).

	// generateName doesn't do anything after create
	existingMeta.SetGenerateName("")
	// resourceVersion is enforced in validation and used during the storage update
	existingMeta.SetResourceVersion("")
	// fields made immutable in validation
	existingMeta.SetUID(types.UID(""))
	existingMeta.SetName("")
	existingMeta.SetNamespace("")
	existingMeta.SetClusterName("")
	existingMeta.SetCreationTimestamp(metav1.Time{})
	existingMeta.SetDeletionTimestamp(nil)
	existingMeta.SetDeletionGracePeriodSeconds(nil)
	existingMeta.SetInitializers(nil)

	if !reflect.DeepEqual(meta, existingMeta) {
		t.Error(diff.ObjectDiff(meta, existingMeta))
	}
}
