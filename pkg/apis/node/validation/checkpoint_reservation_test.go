/*
Copyright The Kubernetes Authors.

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

package validation

import (
	"encoding/json"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
)

func TestCheckpointReservationDeletionValidation(t *testing.T) {
	for _, tc := range []struct {
		name, oldJSON, newJSON string
		status                 bool
	}{
		{"new reservation after deletion", `{"metadata":{"name":"cp","namespace":"ns","resourceVersion":"1","deletionTimestamp":"2026-11-13T00:00:00Z","finalizers":["node.k8s.io/restore-lock"]},"status":{"nodeName":"node-a","conditions":[{"type":"Ready","status":"True","reason":"Completed","lastTransitionTime":"2026-11-13T00:00:00Z"}]}}`, `{"metadata":{"name":"cp","namespace":"ns","resourceVersion":"1","deletionTimestamp":"2026-11-13T00:00:00Z","finalizers":["node.k8s.io/restore-lock"]},"status":{"nodeName":"node-a","conditions":[{"type":"Ready","status":"True","reason":"Completed","lastTransitionTime":"2026-11-13T00:00:00Z"}],"activeRestores":[{"name":"restore","uid":"restore-uid"}]}}`, true},
		{"remove protection with reserved reader", `{"metadata":{"name":"cp","namespace":"ns","resourceVersion":"1","finalizers":["node.k8s.io/restore-lock"]},"status":{"activeRestores":[{"name":"restore","uid":"restore-uid"}]}}`, `{"metadata":{"name":"cp","namespace":"ns","resourceVersion":"1"}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var oldPC, newPC nodeapi.PodCheckpoint
			if err := json.Unmarshal([]byte(tc.oldJSON), &oldPC); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tc.newJSON), &newPC); err != nil {
				t.Fatal(err)
			}
			var oldEnvelope, newEnvelope struct {
				Metadata metav1.ObjectMeta `json:"metadata"`
			}
			if err := json.Unmarshal([]byte(tc.oldJSON), &oldEnvelope); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tc.newJSON), &newEnvelope); err != nil {
				t.Fatal(err)
			}
			oldPC.ObjectMeta, newPC.ObjectMeta = oldEnvelope.Metadata, newEnvelope.Metadata
			if tc.status {
				if errs := ValidatePodCheckpointStatusUpdate(&newPC, &oldPC); len(errs) == 0 {
					t.Fatal("deletion must reject a new restore reservation")
				}
			} else {
				if errs := ValidatePodCheckpointUpdate(&newPC, &oldPC); len(errs) == 0 {
					t.Fatal("active restore must prevent protection removal")
				}
			}
		})
	}
}
