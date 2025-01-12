/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

func streamingEncode(obj runtime.Object, w io.Writer) (bool, error) {
	typeMeta, listMeta, items, err := getListMeta(obj)
	if err == nil {
		return true, streamingEncodeList(w, typeMeta, listMeta, items)
	}
	list, ok := obj.(*unstructured.UnstructuredList)
	if ok {
		return true, streamingEncodeUnstructuredList(w, list)
	}
	return false, nil
}

func getListMeta(list runtime.Object) (metav1.TypeMeta, metav1.ListMeta, []runtime.Object, error) {
	v, err := conversion.EnforcePtr(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	listType := v.Type()
	if listType.NumField() != 3 {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListType to have 3 fields")
	}
	listMetaType, ok := listType.FieldByName("ListMeta")
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field")
	}
	if listMetaType.Tag.Get("json") != "metadata,omitempty" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field tag")
	}
	listMetaValue := v.FieldByName("ListMeta")
	if !listMetaValue.IsValid() {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field")
	}
	listMeta, ok := listMetaValue.Interface().(metav1.ListMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field to have ListMeta type")
	}
	typeMetaField, ok := listType.FieldByName("TypeMeta")
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field")
	}
	if typeMetaField.Tag.Get("json") != ",inline" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field tag")
	}
	typeMetaValue := v.FieldByName("TypeMeta")
	if !typeMetaValue.IsValid() {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field")
	}
	typeMeta, ok := typeMetaValue.Interface().(metav1.TypeMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field to have TypeMeta type")
	}

	itemsField, ok := listType.FieldByName("Items")
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected Items field")
	}
	if itemsField.Tag.Get("json") != "items" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field tag")
	}
	items, err := meta.ExtractList(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	return typeMeta, listMeta, items, nil
}

func streamingEncodeList(w io.Writer, typeMeta metav1.TypeMeta, listMeta metav1.ListMeta, items []runtime.Object) error {
	_, err := w.Write([]byte(`{`))
	if err != nil {
		return err
	}
	if typeMeta.Kind != "" {
		err := encodeValue(w, []byte(`"kind":`), typeMeta.Kind, []byte(","))
		if err != nil {
			return err
		}
	}
	if typeMeta.APIVersion != "" {
		err := encodeValue(w, []byte(`"apiVersion":`), typeMeta.APIVersion, []byte(","))
		if err != nil {
			return err
		}
	}
	err = encodeValue(w, []byte(`"metadata":`), listMeta, []byte(`,"items":`))
	if err != nil {
		return err
	}
	if items == nil {
		_, err = w.Write([]byte("null}\n"))
		return err
	}
	if len(items) == 0 {
		_, err = w.Write([]byte("[]}\n"))
		return err
	}
	suffix := []byte(",")
	prefix := []byte("[")
	for i, item := range items {
		if i == len(items)-1 {
			suffix = nil
		}
		err = encodeValue(w, prefix, item, suffix)
		if err != nil {
			return err
		}
		if i == 0 {
			prefix = nil
		}
	}
	_, err = w.Write([]byte("]}\n"))
	return err
}

func encodeValue(w io.Writer, prefix []byte, value any, suffix []byte) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(prefix) != 0 {
		_, err = w.Write(prefix)
		if err != nil {
			return err
		}
	}
	_, err = w.Write(data)
	if err != nil {
		return err
	}
	if len(suffix) != 0 {
		_, err = w.Write(suffix)
		if err != nil {
			return err
		}
	}
	return nil
}

func streamingEncodeUnstructuredList(w io.Writer, list *unstructured.UnstructuredList) error {
	_, err := w.Write([]byte(`{`))
	if err != nil {
		return err
	}
	keys := slices.Sorted(maps.Keys(list.Object))
	index := 0
	for ; index < len(keys) && keys[index] < "items"; index++ {
		k := keys[index]
		err := encodeKeyValuePair(w, k, list.Object[k])
		if err != nil {
			return err
		}
		_, err = w.Write([]byte(","))
		if err != nil {
			return err
		}
	}
	_, err = w.Write([]byte(`"items":`))
	if err != nil {
		return err
	}
	if len(list.Items) == 0 {
		_, err = w.Write([]byte("[]"))
		if err != nil {
			return err
		}
	}
	suffix := []byte(",")
	prefix := []byte("[")
	for i, item := range list.Items {
		if i == len(list.Items)-1 {
			suffix = []byte("]")
		}
		err = encodeValue(w, prefix, item.Object, suffix)
		if err != nil {
			return err
		}
		if i == 0 {
			prefix = nil
		}
	}
	if index < len(keys) && keys[index] == "items" {
		index++
	}
	for ; index < len(keys); index++ {
		_, err = w.Write([]byte(","))
		if err != nil {
			return err
		}
		k := keys[index]
		err := encodeKeyValuePair(w, k, list.Object[k])
		if err != nil {
			return err
		}
	}
	_, err = w.Write([]byte("}\n"))
	return err
}

func encodeKeyValuePair(w io.Writer, key string, value any) error {
	keyData, err := json.Marshal(key)
	if err != nil {
		return err
	}
	if _, err := w.Write(keyData); err != nil {
		return err
	}
	_, err = w.Write([]byte(":"))
	if err != nil {
		return err
	}
	valueData, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = w.Write(valueData)
	return err
}
