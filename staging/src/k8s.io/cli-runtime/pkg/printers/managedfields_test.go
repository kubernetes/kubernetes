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

package printers

import (
	"bytes"
	"encoding/json"
	"encoding/json/jsontext"
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/yaml"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

type managedFieldsEncodingProbe string

// The legacy engine would encode the underlying string instead of this marker.
func (managedFieldsEncodingProbe) MarshalJSONTo(enc *jsontext.Encoder) error {
	return enc.WriteToken(jsontext.String("json/v2"))
}

type managedFieldsEncodingObject struct {
	metav1.TypeMeta   `json:""`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Probe             managedFieldsEncodingProbe `json:"probe"`
}

func (o *managedFieldsEncodingObject) DeepCopyObject() runtime.Object {
	copy := *o
	o.ObjectMeta.DeepCopyInto(&copy.ObjectMeta)
	return &copy
}

// TestOmitManagedFieldsPrinterUsesJSONV2 verifies real JSON and YAML delegates
// for single objects and RawExtension lists, including v2 dispatch, preserved
// identity fields, managed-field removal, and an unchanged input object.
func TestOmitManagedFieldsPrinterUsesJSONV2(t *testing.T) {
	for _, format := range []string{"json", "yaml"} {
		for _, list := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/list=%t", format, list), func(t *testing.T) {
				item := &managedFieldsEncodingObject{
					TypeMeta: metav1.TypeMeta{APIVersion: "test.example/v1", Kind: "EncodingProbe"},
					ObjectMeta: metav1.ObjectMeta{
						Name: "example",
						ManagedFields: []metav1.ManagedFieldsEntry{{
							Manager: "test-manager", Operation: metav1.ManagedFieldsOperationApply,
						}},
					},
					Probe: "legacy encoding",
				}
				var obj runtime.Object = item
				if list {
					obj = &metav1.List{
						TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "List"},
						Items:    []runtime.RawExtension{{Object: item}},
					}
				}
				original := obj.DeepCopyObject()
				var delegate ResourcePrinter = &JSONPrinter{}
				if format == "yaml" {
					delegate = &YAMLPrinter{}
				}
				// A real delegate verifies that omission still reaches the v2
				// engine, including through the standard library compatibility API.
				printer := &OmitManagedFieldsPrinter{Delegate: delegate}
				var out bytes.Buffer
				err := printer.PrintObj(obj, &out)
				require.Equal(t, original, obj, "printing must not mutate the input object")
				require.NoError(t, err)
				data := out.Bytes()
				if format == "yaml" {
					data, err = yaml.YAMLToJSON(data)
					require.NoError(t, err)
				}
				if list {
					var got struct {
						Items []json.RawMessage `json:"items"`
					}
					require.NoError(t, json.Unmarshal(data, &got))
					require.Len(t, got.Items, 1)
					data = got.Items[0]
				}
				var got struct {
					metav1.TypeMeta
					Metadata map[string]json.RawMessage `json:"metadata"`
					Probe    string                     `json:"probe"`
				}
				require.NoError(t, json.Unmarshal(data, &got))
				require.Equal(t, "json/v2", got.Probe)
				require.Equal(t, item.TypeMeta, got.TypeMeta)
				require.Equal(t, `"example"`, string(got.Metadata["name"]))
				require.NotContains(t, got.Metadata, "managedFields")
			})
		}
	}
}

type testResourcePrinter func(object runtime.Object, writer io.Writer) error

func (p testResourcePrinter) PrintObj(o runtime.Object, w io.Writer) error {
	return p(o, w)
}

func TestOmitManagedFieldsPrinter(t *testing.T) {
	testCases := []struct {
		name     string
		object   runtime.Object
		expected runtime.Object
	}{
		{
			name: "pod without managedFields",
			object: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1"},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1"},
			},
		},
		{
			name: "pod with managedFields",
			object: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					ManagedFields: []metav1.ManagedFieldsEntry{
						{Manager: "kubectl", Operation: metav1.ManagedFieldsOperationApply},
					},
				},
			},
			expected: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1"},
			},
		},
		{
			name: "pod list",
			object: &v1.PodList{
				Items: []v1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:          "pod1",
							ManagedFields: []metav1.ManagedFieldsEntry{},
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "pod2",
							ManagedFields: []metav1.ManagedFieldsEntry{
								{Manager: "kubectl", Operation: metav1.ManagedFieldsOperationApply},
							},
						},
					},
					{ObjectMeta: metav1.ObjectMeta{Name: "pod3"}},
				},
			},
			expected: &v1.PodList{
				Items: []v1.Pod{
					{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "pod3"}},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := require.New(t)
			delegate := func(o runtime.Object, w io.Writer) error {
				r.Equal(tc.expected, o)
				return nil
			}
			p := OmitManagedFieldsPrinter{Delegate: testResourcePrinter(delegate)}
			r.NoError(p.PrintObj(tc.object, nil))
		})
	}
}
