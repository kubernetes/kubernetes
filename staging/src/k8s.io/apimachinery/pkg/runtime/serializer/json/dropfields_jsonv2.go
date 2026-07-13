//go:build goexperiment.jsonv2 || go1.27

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

package json

import (
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"io"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// managedFieldsPath is the only field path currently supported by DropFields.
const managedFieldsPath = "metadata.managedFields"

// encodeDroppingFields encodes obj while omitting the configured DropFields, using
// json/v2 custom marshalers so the fields are dropped at marshal time without copying
// or mutating obj. Unsupported field paths are ignored.
func (s *Serializer) encodeDroppingFields(obj runtime.Object, w io.Writer) error {
	return jsonv2.MarshalWrite(w, obj, jsonv2.WithMarshalers(dropFieldMarshalers(s.options.DropFields)))
}

// dropFieldMarshalers returns json/v2 marshalers that emit null for each supported
// drop target. metadata.managedFields is the only supported target today; its Go type
// (metav1.ManagedFieldsEntry slice) only appears as ObjectMeta.ManagedFields.
func dropFieldMarshalers(fields []string) *jsonv2.Marshalers {
	var ms []*jsonv2.Marshalers
	for _, field := range fields {
		switch field {
		case managedFieldsPath:
			ms = append(ms, jsonv2.MarshalToFunc(func(enc *jsontext.Encoder, _ []metav1.ManagedFieldsEntry) error {
				return enc.WriteToken(jsontext.Null)
			}))
		}
	}
	return jsonv2.JoinMarshalers(ms...)
}
