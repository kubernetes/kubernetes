/*
Copyright 2019 The Kubernetes Authors.

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

package csi_test

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/volume/csi"
)

func TestDriversStore(t *testing.T) {
	store := &csi.DriversStore{}
	someDriver := csi.Driver{}

	expectAbsent(t, store, "does-not-exist")

	store.Set("some-driver", someDriver)
	expectPresent(t, store, "some-driver", someDriver)

	store.Delete("some-driver")
	expectAbsent(t, store, "some-driver")

	store.Set("some-driver", someDriver)

	store.Clear()
	expectAbsent(t, store, "some-driver")
}

func expectPresent(t *testing.T, store *csi.DriversStore, name string, expected csi.Driver) {
	t.Helper()

	retrieved, ok := store.Get(name)

	if !ok {
		t.Fatalf("expected driver '%s' to exist", name)
	}

	if !reflect.DeepEqual(retrieved, expected) {
		t.Fatalf("expected driver '%s' to be equal to %v", name, expected)
	}
}

func expectAbsent(t *testing.T, store *csi.DriversStore, name string) {
	t.Helper()

	if _, ok := store.Get(name); ok {
		t.Fatalf("expected driver '%s' not to exist in store", name)
	}
}
