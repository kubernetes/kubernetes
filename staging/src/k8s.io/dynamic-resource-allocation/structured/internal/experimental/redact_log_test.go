/*
Copyright 2026 The Kubernetes Authors.

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

package experimental

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

func TestRedactSlicesForLog(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		if got := redactSlicesForLog(nil); got != nil {
			t.Errorf("redactSlicesForLog(nil) = %v, want nil", got)
		}
	})
	t.Run("redacts CEL without mutating input", func(t *testing.T) {
		in := &draapi.ResourceSlice{ObjectMeta: metav1.ObjectMeta{
			Name:          "s",
			Labels:        map[string]string{"l": "v"},
			Annotations:   map[string]string{"cel": "device.attributes['x']"},
			ManagedFields: []metav1.ManagedFieldsEntry{{Manager: "m"}},
		}}
		in.Spec.SharingAffinity = make([]resourceapi.SharingAffinityExtractor, 1)
		in.Spec.SharingAffinity[0].Expression = "device.attributes['x']"

		out := redactSlicesForLog([]*draapi.ResourceSlice{in})
		if len(out) != 1 {
			t.Fatalf("len = %d, want 1", len(out))
		}
		got := out[0]
		if got.Name != "s" || got.Labels["l"] != "v" {
			t.Errorf("safe metadata lost: %+v", got.ObjectMeta)
		}
		if v, ok := got.Annotations["cel"]; !ok || v != "<redacted>" {
			t.Errorf("annotation = %q (present %v), want key kept, value <redacted>", v, ok)
		}
		if got.ManagedFields != nil {
			t.Errorf("managedFields = %v, want nil", got.ManagedFields)
		}
		if e := got.Spec.SharingAffinity[0].Expression; e != "<redacted>" {
			t.Errorf("expression = %q, want <redacted>", e)
		}
		if in.Annotations["cel"] != "device.attributes['x']" || in.Spec.SharingAffinity[0].Expression != "device.attributes['x']" || in.ManagedFields == nil {
			t.Errorf("input slice was mutated: %+v", in)
		}
	})
}
