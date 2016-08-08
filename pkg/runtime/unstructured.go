/*
Copyright 2015 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"strings"

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
		return nil, &gvk, NewMissingKindErr(string(data))
	}

	return obj, &gvk, nil
}

func (unstructuredJSONScheme) Encode(obj Object, w io.Writer) error {
	switch t := obj.(type) {
	case *Unstructured:
		return json.NewEncoder(w).Encode(t.Object)
	case *UnstructuredList:
		items := make([]map[string]interface{}, 0, len(t.Items))
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
	case *VersionedObjects:
		o, err := s.decode(data)
		if err == nil {
			x.Objects = []Object{o}
		}
		return err
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

	// For typed lists, e.g., a PodList, API server doesn't set each item's
	// APIVersion and Kind. We need to set it.
	listAPIVersion := list.GetAPIVersion()
	listKind := list.GetKind()
	itemKind := strings.TrimSuffix(listKind, "List")

	delete(list.Object, "items")
	list.Items = nil
	for _, i := range dList.Items {
		unstruct := &Unstructured{}
		if err := s.decodeToUnstructured([]byte(i), unstruct); err != nil {
			return err
		}
		// This is hacky. Set the item's Kind and APIVersion to those inferred
		// from the List.
		if len(unstruct.GetKind()) == 0 && len(unstruct.GetAPIVersion()) == 0 {
			unstruct.SetKind(itemKind)
			unstruct.SetAPIVersion(listAPIVersion)
		}
		list.Items = append(list.Items, unstruct)
	}
	return nil
}

// UnstructuredObjectConverter is an ObjectConverter for use with
// Unstructured objects. Since it has no schema or type information,
// it will only succeed for no-op conversions. This is provided as a
// sane implementation for APIs that require an object converter.
type UnstructuredObjectConverter struct{}

func (UnstructuredObjectConverter) Convert(in, out interface{}) error {
	unstructIn, ok := in.(*Unstructured)
	if !ok {
		return fmt.Errorf("input type %T in not valid for unstructured conversion", in)
	}

	unstructOut, ok := out.(*Unstructured)
	if !ok {
		return fmt.Errorf("output type %T in not valid for unstructured conversion", out)
	}

	// maybe deep copy the map? It is documented in the
	// ObjectConverter interface that this function is not
	// guaranteeed to not mutate the input. Or maybe set the input
	// object to nil.
	unstructOut.Object = unstructIn.Object
	return nil
}

func (UnstructuredObjectConverter) ConvertToVersion(in Object, outVersion unversioned.GroupVersion) (Object, error) {
	if gvk := in.GetObjectKind().GroupVersionKind(); gvk.GroupVersion() != outVersion {
		return nil, errors.New("unstructured converter cannot convert versions")
	}
	return in, nil
}

func (UnstructuredObjectConverter) ConvertFieldLabel(version, kind, label, value string) (string, string, error) {
	return "", "", errors.New("unstructured cannot convert field labels")
}
