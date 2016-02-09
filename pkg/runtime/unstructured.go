/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package runtime

import (
	"encoding/json"
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// UnstructuredJSONScheme is capable of converting JSON data into the Unstructured
// type, which can be used for generic access to objects without a predefined scheme.
// TODO: move into serializer/json.
var UnstructuredJSONScheme Codec = unstructuredJSONScheme{}

type unstructuredJSONScheme struct{}

func (s unstructuredJSONScheme) Decode(data []byte, _ *unversioned.GroupVersionKind, _ Object) (Object, *unversioned.GroupVersionKind, error) {
	unstruct := &Unstructured{}

	m := make(map[string]interface{})
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, nil, err
	}
	if v, ok := m["kind"]; ok {
		if s, ok := v.(string); ok {
			unstruct.Kind = s
		}
	}
	if v, ok := m["apiVersion"]; ok {
		if s, ok := v.(string); ok {
			unstruct.APIVersion = s
		}
	}

	if len(unstruct.APIVersion) == 0 {
		return nil, nil, NewMissingVersionErr(string(data))
	}
	gv, err := unversioned.ParseGroupVersion(unstruct.APIVersion)
	if err != nil {
		return nil, nil, err
	}
	gvk := gv.WithKind(unstruct.Kind)
	if len(unstruct.Kind) == 0 {
		return nil, &gvk, NewMissingKindErr(string(data))
	}
	unstruct.Object = m
	return unstruct, &gvk, nil
}

func (s unstructuredJSONScheme) EncodeToStream(obj Object, w io.Writer, overrides ...unversioned.GroupVersion) error {
	switch t := obj.(type) {
	case *Unstructured:
		return json.NewEncoder(w).Encode(t.Object)
	case *Unknown:
		_, err := w.Write(t.RawJSON)
		return err
	default:
		return json.NewEncoder(w).Encode(t)
	}
}
