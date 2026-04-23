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

package json

import (
	"bytes"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
)

func sampleCarp() *testapigroupv1.Carp {
	now := metav1.Now()
	return &testapigroupv1.Carp{
		TypeMeta: metav1.TypeMeta{Kind: "Carp", APIVersion: "testapigroup/v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "carp-0",
			Namespace: "ns",
			ManagedFields: []metav1.ManagedFieldsEntry{
				{Manager: "kubectl", Operation: metav1.ManagedFieldsOperationApply, APIVersion: "v1", Time: &now, FieldsType: "FieldsV1", FieldsV1: &metav1.FieldsV1{Raw: []byte(`{"f:metadata":{"f:labels":{"f:app":{}}}}`)}},
			},
		},
	}
}

func TestExcludeManagedFields(t *testing.T) {
	full := NewSerializerWithOptions(DefaultMetaFactory, nil, nil, SerializerOptions{})
	stripped := NewSerializerWithOptions(DefaultMetaFactory, nil, nil, SerializerOptions{ExcludeManagedFields: true})

	// Distinct Identifier so runtime.CacheableObject caches the two forms separately.
	if full.Identifier() == stripped.Identifier() {
		t.Fatalf("expected distinct identifiers, both were %q", full.Identifier())
	}

	carp := sampleCarp()
	originalLen := len(carp.ObjectMeta.ManagedFields)

	var buf bytes.Buffer
	if err := stripped.Encode(carp, &buf); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if strings.Contains(buf.String(), "managedFields") {
		t.Fatalf("expected output to omit managedFields, got: %s", buf.String())
	}

	// Encode must not mutate the caller's object on the non-CacheableObject path.
	if got := len(carp.ObjectMeta.ManagedFields); got != originalLen {
		t.Fatalf("input was mutated: managedFields len went from %d to %d", originalLen, got)
	}
}

// TestExcludeManagedFieldsList verifies the stripped serializer also drops
// managedFields off each item in a list response. Without per-item stripping,
// LIST and WATCH bookmark fan-outs would still ship the field on the wire and
// the KEP's bytes-on-wire claim would be wrong for the most common request
// shape. Covers both the standard JSON path and the streaming-collections
// encoder.
func TestExcludeManagedFieldsList(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "standard"
		if streaming {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			stripped := NewSerializerWithOptions(DefaultMetaFactory, nil, nil, SerializerOptions{
				ExcludeManagedFields:         true,
				StreamingCollectionsEncoding: streaming,
			})

			list := &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{Kind: "CarpList", APIVersion: "testapigroup/v1"},
				Items:    []testapigroupv1.Carp{*sampleCarp(), *sampleCarp()},
			}
			originalItemMFLen := len(list.Items[0].ObjectMeta.ManagedFields)
			if originalItemMFLen == 0 {
				t.Fatalf("test fixture broken: item has no managedFields to strip")
			}

			var buf bytes.Buffer
			if err := stripped.Encode(list, &buf); err != nil {
				t.Fatalf("encode list: %v", err)
			}
			if strings.Contains(buf.String(), "managedFields") {
				t.Fatalf("list output still contains managedFields: %s", buf.String())
			}
			// Caller's items must not have been mutated.
			for i := range list.Items {
				if got := len(list.Items[i].ObjectMeta.ManagedFields); got != originalItemMFLen {
					t.Fatalf("item %d was mutated: managedFields len went from %d to %d", i, originalItemMFLen, got)
				}
			}
		})
	}
}
