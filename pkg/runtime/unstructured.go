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
	gojson "encoding/json"
	"io"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/json"
)

// UnstructuredJSONScheme is capable of converting JSON data into the Unstructured
// type, which can be used for generic access to objects without a predefined scheme.
// TODO: move into serializer/json.
var UnstructuredJSONScheme Codec = unstructuredJSONScheme{}

type unstructuredJSONScheme struct{}

func (s unstructuredJSONScheme) Decode(data []byte, _ *unversioned.GroupVersionKind, obj Object) (Object, *unversioned.GroupVersionKind, error) {
	var err error
	if obj != nil {
		err = s.decodeInto(data, obj)
	} else {
		obj, err = s.decode(data)
	}

	if err != nil {
		return nil, nil, err
	}

	gvk := obj.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return nil, gvk, NewMissingKindErr(string(data))
	}

	return obj, gvk, nil
}

func (unstructuredJSONScheme) EncodeToStream(obj Object, w io.Writer, overrides ...unversioned.GroupVersion) error {
	switch t := obj.(type) {
	case *Unstructured:
		return json.NewEncoder(w).Encode(t.Object)
	case *UnstructuredList:
		var items []map[string]interface{}
		for _, i := range t.Items {
			items = append(items, i.Object)
		}
		t.Object["items"] = items
		defer func() { delete(t.Object, "items") }()
		return json.NewEncoder(w).Encode(t.Object)
	case *Unknown:
		// TODO: Unstructured needs to deal with ContentType.
		_, err := w.Write(t.Raw)
		return err
	default:
		return json.NewEncoder(w).Encode(t)
	}
}

func (s unstructuredJSONScheme) decode(data []byte) (Object, error) {
	type detector struct {
		Items gojson.RawMessage
	}
	var det detector
	if err := json.Unmarshal(data, &det); err != nil {
		return nil, err
	}

	if det.Items != nil {
		list := &UnstructuredList{}
		err := s.decodeToList(data, list)
		return list, err
	}

	// No Items field, so it wasn't a list.
	unstruct := &Unstructured{}
	err := s.decodeToUnstructured(data, unstruct)
	return unstruct, err
}
func (s unstructuredJSONScheme) decodeInto(data []byte, obj Object) error {
	switch x := obj.(type) {
	case *Unstructured:
		return s.decodeToUnstructured(data, x)
	case *UnstructuredList:
		return s.decodeToList(data, x)
	default:
		return json.Unmarshal(data, x)
	}
}

func (unstructuredJSONScheme) decodeToUnstructured(data []byte, unstruct *Unstructured) error {
	m := make(map[string]interface{})
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}

	unstruct.Object = m

	return nil
}

func (s unstructuredJSONScheme) decodeToList(data []byte, list *UnstructuredList) error {
	type decodeList struct {
		Items []gojson.RawMessage
	}

	var dList decodeList
	if err := json.Unmarshal(data, &dList); err != nil {
		return err
	}

	if err := json.Unmarshal(data, &list.Object); err != nil {
		return err
	}

	delete(list.Object, "items")
	list.Items = nil
	for _, i := range dList.Items {
		unstruct := &Unstructured{}
		if err := s.decodeToUnstructured([]byte(i), unstruct); err != nil {
			return err
		}
		list.Items = append(list.Items, unstruct)
	}
	return nil
}
