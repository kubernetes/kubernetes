/*
Copyright 2025 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"
	"sort"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

func streamEncodeCollections(obj runtime.Object, w io.Writer) (bool, error) {
	list, ok := obj.(*unstructured.UnstructuredList)
	if ok {
		return true, newStreamEncoder(w).encodeUnstructuredList(list)
	}
	if _, ok := obj.(json.Marshaler); ok {
		return false, nil
	}
	typeMeta, listMeta, items, err := getListMeta(obj)
	if err == nil {
		return true, newStreamEncoder(w).encodeList(typeMeta, listMeta, items)
	}
	return false, nil
}

// getListMeta implements list extraction logic for json stream serialization.
//
// Reason for a custom logic instead of reusing accessors from meta package:
// * Validate json tags to prevent incompatibility with json standard package.
// * ListMetaAccessor doesn't distinguish empty from nil value.
// * TypeAccessort reparsing "apiVersion" and serializing it with "{group}/{version}"
func getListMeta(list runtime.Object) (metav1.TypeMeta, metav1.ListMeta, []runtime.Object, error) {
	listValue, err := conversion.EnforcePtr(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	listType := listValue.Type()
	if listType.NumField() != 3 {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListType to have 3 fields")
	}
	// TypeMeta
	typeMeta, ok := listValue.Field(0).Interface().(metav1.TypeMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected TypeMeta field to have TypeMeta type")
	}
	if !listType.Field(0).Anonymous {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta json field tag to be embedded`)
	}
	if jsonTag, jsonTagExists := listType.Field(0).Tag.Lookup("json"); !jsonTagExists {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta json field tag`)
	} else if jsonTag != "" && jsonTag != ",inline" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected TypeMeta json field tag to be "" or ",inline"`)
	}
	// ListMeta
	listMeta, ok := listValue.Field(1).Interface().(metav1.ListMeta)
	if !ok {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf("expected ListMeta field to have ListMeta type")
	}
	if listType.Field(1).Tag.Get("json") != "metadata,omitempty" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected ListMeta json field tag to be "metadata,omitempty"`)
	}
	// Items
	items, err := meta.ExtractList(list)
	if err != nil {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, err
	}
	if listType.Field(2).Tag.Get("json") != "items" {
		return metav1.TypeMeta{}, metav1.ListMeta{}, nil, fmt.Errorf(`expected Items json field tag to be "items"`)
	}
	return typeMeta, listMeta, items, nil
}

// streamEncoder encodes JSON values to w, reusing an internal buffer across
// values to avoid the fresh output allocation json.Marshal makes per call.
type streamEncoder struct {
	w    io.Writer
	buf  bytes.Buffer
	json *json.Encoder
}

func newStreamEncoder(w io.Writer) *streamEncoder {
	e := &streamEncoder{w: w}
	e.json = json.NewEncoder(&e.buf)
	return e
}

func (e *streamEncoder) encodeList(typeMeta metav1.TypeMeta, listMeta metav1.ListMeta, items []runtime.Object) error {
	// Start
	if _, err := e.w.Write([]byte(`{`)); err != nil {
		return err
	}

	// TypeMeta
	if typeMeta.Kind != "" {
		if err := e.encodeKeyValuePair("kind", typeMeta.Kind, []byte(",")); err != nil {
			return err
		}
	}
	if typeMeta.APIVersion != "" {
		if err := e.encodeKeyValuePair("apiVersion", typeMeta.APIVersion, []byte(",")); err != nil {
			return err
		}
	}

	// ListMeta
	if err := e.encodeKeyValuePair("metadata", listMeta, []byte(",")); err != nil {
		return err
	}

	// Items
	if err := e.encodeItemsObjectSlice(items); err != nil {
		return err
	}

	// End
	_, err := e.w.Write([]byte("}\n"))
	return err
}

func (e *streamEncoder) encodeItemsObjectSlice(items []runtime.Object) (err error) {
	if items == nil {
		err := e.encodeKeyValuePair("items", nil, nil)
		return err
	}
	_, err = e.w.Write([]byte(`"items":[`))
	if err != nil {
		return err
	}
	suffix := []byte(",")
	for i, item := range items {
		if i == len(items)-1 {
			suffix = nil
		}
		err := e.encodeValue(item, suffix)
		if err != nil {
			return err
		}
	}
	_, err = e.w.Write([]byte("]"))
	if err != nil {
		return err
	}
	return err
}

func (e *streamEncoder) encodeUnstructuredList(list *unstructured.UnstructuredList) error {
	_, err := e.w.Write([]byte(`{`))
	if err != nil {
		return err
	}
	keys := slices.Collect(maps.Keys(list.Object))
	if _, exists := list.Object["items"]; !exists {
		keys = append(keys, "items")
	}
	sort.Strings(keys)

	suffix := []byte(",")
	for i, key := range keys {
		if i == len(keys)-1 {
			suffix = nil
		}
		if key == "items" {
			err = e.encodeItemsUnstructuredSlice(list.Items, suffix)
		} else {
			err = e.encodeKeyValuePair(key, list.Object[key], suffix)
		}
		if err != nil {
			return err
		}
	}
	_, err = e.w.Write([]byte("}\n"))
	return err
}

func (e *streamEncoder) encodeItemsUnstructuredSlice(items []unstructured.Unstructured, suffix []byte) (err error) {
	_, err = e.w.Write([]byte(`"items":[`))
	if err != nil {
		return err
	}
	comma := []byte(",")
	for i, item := range items {
		if i == len(items)-1 {
			comma = nil
		}
		err := e.encodeValue(item.Object, comma)
		if err != nil {
			return err
		}
	}
	_, err = e.w.Write([]byte("]"))
	if err != nil {
		return err
	}
	if len(suffix) > 0 {
		_, err = e.w.Write(suffix)
	}
	return err
}

func (e *streamEncoder) encodeKeyValuePair(key string, value any, suffix []byte) (err error) {
	err = e.encodeValue(key, []byte(":"))
	if err != nil {
		return err
	}
	err = e.encodeValue(value, suffix)
	if err != nil {
		return err
	}
	return err
}

func (e *streamEncoder) encodeValue(value any, suffix []byte) error {
	e.buf.Reset()
	if err := e.json.Encode(value); err != nil {
		return err
	}
	// Encode appends a newline after the value; replace it with the suffix to
	// keep the output identical to json.Marshal's.
	e.buf.Truncate(e.buf.Len() - 1)
	e.buf.Write(suffix)
	_, err := e.w.Write(e.buf.Bytes())
	return err
}
