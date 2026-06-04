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
	"fmt"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/integration/apiserver/apidefinitions"
)

// noConflicts is the set of resources for which
// a conflict cannot occur.
var noConflicts = map[string]struct{}{
	// both spec and status get wiped for CSRs,
	// nothing is expected to be managed for it, skip it
	"certificatesigningrequests": {},
	// storageVersions are skipped because their spec is empty
	// and thus they can never have a conflict.
	"storageversions": {},
	// servicecidrs are skipped because their spec is inmutable
	// and thus they can never have a conflict.
	"servicecidrs": {},
	// namespaces only have a spec.finalizers field which is also skipped,
	// thus it will never have a conflict.
	"namespaces": {},
}

// TestResetFields makes sure that fieldManager does not own fields reset by the storage strategy.
// It takes 2 objects obj1 and obj2 that differ by one field in the spec and one field in the status.
// It applies obj1 to the spec endpoint and obj2 to the status endpoint, the lack of conflicts
// confirms that the fieldmanager1 is wiped of the status and fieldmanager2 is wiped of the spec.
// We then attempt to apply obj2 to the spec endpoint which fails with an expected conflict.
func TestApplyResetFields(t *testing.T) {
	apidefinitions.TestAllDefinitions(t, "reset-fields-test", func(t *testing.T, api apidefinitions.Definition) {
		if !api.HasStatus() {
			t.Skip()
		}
		if !api.HasVerb("patch") || !api.HasVerb("get") || !api.HasVerb("update") {
			t.Skip("Resource does not support patch, get, and update")
		}

		// assemble first object
		status := api.StorageData.GetStatusStub()
		obj1 := apidefinitions.TestObj(t, api.StorageData.Stub, status, api.Mapping.GroupVersionKind)
		name := obj1.GetName()

		rsc := api.ResourceClient()

		// apply the spec of the first object
		_, err := rsc.Apply(context.TODO(), name, obj1, metav1.ApplyOptions{FieldManager: "fieldmanager1"})
		if err != nil {
			t.Fatalf("Failed to apply obj1: %v", err)
		}

		// create second object
		status2 := api.StorageData.GetMutatedStatusStub()
		differentSpec := api.StorageData.MutatedStub
		if differentSpec == "" {
			t.Skipf("No conflicting spec data defined for %v to test reset fields", api.Mapping.Resource)
		}

		obj2 := apidefinitions.TestObj(t, api.StorageData.Stub, status2, api.Mapping.GroupVersionKind)
		if err := json.Unmarshal([]byte(differentSpec), &obj2.Object); err != nil {
			t.Fatal(err)
		}
		obj2.SetName(name)

		if reflect.DeepEqual(obj1, obj2) {
			t.Fatalf("obj1 and obj2 should not be equal %v", obj2)
		}

		// apply the status of the second object
		_, err = rsc.ApplyStatus(context.TODO(), name, obj2, metav1.ApplyOptions{FieldManager: "fieldmanager2"})
		if err != nil {
			t.Fatalf("Failed to apply obj2: %v", err)
		}

		if _, ok := noConflicts[api.Mapping.Resource.Resource]; !ok {
			// reapply second object to the spec endpoint
			_, err = rsc.Apply(context.TODO(), name, obj2, metav1.ApplyOptions{FieldManager: "fieldmanager2"})
			if err := expectConflict(nil, err, rsc, name); err != nil {
				t.Fatalf("Did not get expected conflict on spec apply: %v", err)
			}
		}

		if err := rsc.Delete(context.TODO(), name, *metav1.NewDeleteOptions(0)); err != nil {
			t.Fatalf("deleting final object failed: %v", err)
		}
	})
}

func expectConflict(objRet *unstructured.Unstructured, err error, rsc dynamic.ResourceInterface, name string) error {
	if err != nil && strings.Contains(err.Error(), "conflict") {
		return nil
	}
	which := "returned"
	// something unexpected is going on here, let's not assume that objRet==nil if any only if err!=nil
	if objRet == nil {
		which = "subsequently fetched"
		var err2 error
		objRet, err2 = rsc.Get(context.TODO(), name, metav1.GetOptions{})
		if err2 != nil {
			return fmt.Errorf("instead got error %w, and failed to Get object: %v", err, err2)
		}
	}
	marshBytes, marshErr := json.Marshal(objRet)
	var gotten string
	if marshErr == nil {
		gotten = string(marshBytes)
	} else {
		gotten = fmt.Sprintf("<failed to json.Marshall(%#+v): %v>", objRet, marshErr)
	}
	return fmt.Errorf("instead got error %w; %s object is %s", err, which, gotten)
}
