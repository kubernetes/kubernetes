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

package unstructured

import (
	gojson "encoding/json"
	"fmt"
	"io"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/klog/v2"
)

// NestedFieldCopy returns a deep copy of the value of a nested field.
// Returns false if the value is missing.
// No error is returned for a nil field.
//
// Note: fields passed to this function are treated as keys within the passed
// object; no array/slice syntax is supported.
func NestedFieldCopy(obj map[string]any, fields ...string) (any, bool, error) {
	val, found, err := NestedFieldNoCopy(obj, fields...)
	if !found || err != nil {
		return nil, found, err
	}
	return runtime.DeepCopyJSONValue(val), true, nil
}

// NestedFieldNoCopy returns a reference to a nested field.
// Returns false if value is not found and an error if unable
// to traverse obj.
//
// Note: fields passed to this function are treated as keys within the passed
// object; no array/slice syntax is supported.
func NestedFieldNoCopy(obj map[string]any, fields ...string) (any, bool, error) {
	var val any = obj

	for i, field := range fields {
		if val == nil {
			return nil, false, nil
		}
		if m, ok := val.(map[string]any); ok {
			val, ok = m[field]
			if !ok {
				return nil, false, nil
			}
		} else {
			return nil, false, fmt.Errorf("%v accessor error: %v is of the type %T, expected map[string]any", jsonPath(fields[:i+1]), val, val)
		}
	}
	return val, true, nil
}

// NestedValueCopy returns a deep copy of the value of a nested field
// which has desired type. Returns false if the value is not found and
// an error if not matched instantiated type argument.
//
// Note: fields passed to this function are treated as keys within the passed
// object; no array/slice syntax is supported.
func NestedValueCopy[T runtime.JSONCopyable](obj map[string]any, fields ...string) (val T, found bool, err error) {
	nestedVal, found, err := NestedFieldCopy(obj, fields...)
	if !found || err != nil {
		return val, found, err
	}
	val, ok := nestedVal.(T)
	if !ok {
		return val, false, fmt.Errorf("%v accessor error: %v is of the type %T, expected %T", jsonPath(fields), nestedVal, nestedVal, val)
	}
	return val, true, nil
}

// NestedValueNoCopy returns the value of a nested field which
// has desired type. Returns false if the value is not found and
// an error if not matched instantiated type argument.
//
// Note: fields passed to this function are treated as keys within the passed
// object; no array/slice syntax is supported.
func NestedValueNoCopy[T any](obj map[string]any, fields ...string) (val T, found bool, err error) {
	nestedVal, found, err := NestedFieldNoCopy(obj, fields...)
	if !found || err != nil {
		return val, found, err
	}
	val, ok := nestedVal.(T)
	if !ok {
		return val, false, fmt.Errorf("%v accessor error: %v is of the type %T, expected %T", jsonPath(fields), nestedVal, nestedVal, val)
	}
	return val, true, nil
}

// NestedTypedMap returns a deep copy of map[string]T value of a nested field.
// Returns false if value is not found and an error if not a map[string]T.
func NestedTypedMap[T any](obj map[string]any, fields ...string) (map[string]T, bool, error) {
	m, found, err := NestedValueNoCopy[map[string]any](obj, fields...)
	if !found || err != nil {
		return nil, found, err
	}
	typedMap := make(map[string]T, len(m))
	for k, v := range m {
		if typed, ok := v.(T); ok {
			typedMap[k] = typed
		} else {
			return nil, false, fmt.Errorf("%v accessor error: contains non expected typed key in the map: %v is of the type %T", jsonPath(fields), v, v)
		}
	}
	return typedMap, true, nil
}

// NestedTypedSlice returns a copy of []T value of a nested field.
// Returns false if value is not found and an error if not a []T or contains non type T items in the slice.
func NestedTypedSlice[T any](obj map[string]any, fields ...string) ([]T, bool, error) {
	val, found, err := NestedValueNoCopy[[]any](obj, fields...)
	if !found || err != nil {
		return nil, found, err
	}
	typedSlice := make([]T, 0, len(val))
	for _, v := range val {
		if typed, ok := v.(T); ok {
			typedSlice = append(typedSlice, typed)
		} else {
			return nil, false, fmt.Errorf("%v accessor error: contains non expected type key in the slice: %v is of the type %T", jsonPath(fields), v, v)
		}
	}
	return typedSlice, true, nil
}

// NestedString returns the string value of a nested field.
// Returns false if value is not found and an error if not a string.
//
// Deprecated: use generic NestedValueNoCopy instead.
func NestedString(obj map[string]any, fields ...string) (string, bool, error) {
	return NestedValueNoCopy[string](obj, fields...)
}

// NestedBool returns the bool value of a nested field.
// Returns false if value is not found and an error if not a bool.
//
// Deprecated: use generic NestedValueNoCopy instead.
func NestedBool(obj map[string]any, fields ...string) (bool, bool, error) {
	return NestedValueNoCopy[bool](obj, fields...)
}

// NestedFloat64 returns the float64 value of a nested field.
// Returns false if value is not found and an error if not a float64.
//
// Deprecated: use generic NestedValueNoCopy instead.
func NestedFloat64(obj map[string]any, fields ...string) (float64, bool, error) {
	return NestedValueNoCopy[float64](obj, fields...)
}

// NestedInt64 returns the int64 value of a nested field.
// Returns false if value is not found and an error if not an int64.
//
// Deprecated: use generic NestedValueNoCopy instead.
func NestedInt64(obj map[string]any, fields ...string) (int64, bool, error) {
	return NestedValueNoCopy[int64](obj, fields...)
}

// NestedStringSlice returns a copy of []string value of a nested field.
// Returns false if value is not found and an error if not a []any or contains non-string items in the slice.
//
// Deprecated: use generic NestedTypedSlice instead.
func NestedStringSlice(obj map[string]any, fields ...string) ([]string, bool, error) {
	return NestedTypedSlice[string](obj, fields...)
}

// NestedSlice returns a deep copy of []any value of a nested field.
// Returns false if value is not found and an error if not a []any.
//
// Deprecated: use generic NestedValueCopy instead.
func NestedSlice(obj map[string]any, fields ...string) ([]any, bool, error) {
	return NestedValueCopy[[]any](obj, fields...)
}

// NestedStringMap returns a copy of map[string]string value of a nested field.
// Returns false if value is not found and an error if not a map[string]any or contains non-string values in the map.
//
// Deprecated: use generic NestedTypedMap instead.
func NestedStringMap(obj map[string]any, fields ...string) (map[string]string, bool, error) {
	return NestedTypedMap[string](obj, fields...)
}

// NestedMap returns a deep copy of map[string]any value of a nested field.
// Returns false if value is not found and an error if not a map[string]any.
//
// Deprecated: use generic NestedValueCopy instead.
func NestedMap(obj map[string]any, fields ...string) (map[string]any, bool, error) {
	return NestedValueCopy[map[string]any](obj, fields...)
}

// SetNestedField sets the value of a nested field to a deep copy of the value provided.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
func SetNestedField(obj map[string]any, value any, fields ...string) error {
	return setNestedFieldNoCopy(obj, runtime.DeepCopyJSONValue(value), fields...)
}

func setNestedFieldNoCopy(obj map[string]any, value any, fields ...string) error {
	m := obj

	for i, field := range fields[:len(fields)-1] {
		if val, ok := m[field]; ok {
			if valMap, ok := val.(map[string]any); ok {
				m = valMap
			} else {
				return fmt.Errorf("value cannot be set because %v is not a map[string]any", jsonPath(fields[:i+1]))
			}
		} else {
			newVal := make(map[string]any)
			m[field] = newVal
			m = newVal
		}
	}
	m[fields[len(fields)-1]] = value
	return nil
}

// SetNestedTypedSlice sets the T slice value of a nested field.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
func SetNestedTypedSlice[T any](obj map[string]any, value []T, fields ...string) error {
	m := make([]any, 0, len(value)) // convert []T into []any
	for _, v := range value {
		m = append(m, v)
	}
	return setNestedFieldNoCopy(obj, m, fields...)
}

// SetNestedStringSlice sets the string slice value of a nested field.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
//
// Deprecated: use generic SetNestedTypedSlice instead.
func SetNestedStringSlice(obj map[string]any, value []string, fields ...string) error {
	return SetNestedTypedSlice[string](obj, value, fields...)
}

// SetNestedSlice sets the slice value of a nested field.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
func SetNestedSlice(obj map[string]any, value []any, fields ...string) error {
	return SetNestedField(obj, value, fields...)
}

// SetNestedTypedMap sets the map[string]T value of a nested field.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
func SetNestedTypedMap[T any](obj map[string]any, value map[string]T, fields ...string) error {
	m := make(map[string]any, len(value)) // convert map[string]T into map[string]any
	for k, v := range value {
		m[k] = v
	}
	return setNestedFieldNoCopy(obj, m, fields...)
}

// SetNestedStringMap sets the map[string]string value of a nested field.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
//
// Deprecated: use generic SetNestedTypedMap instead.
func SetNestedStringMap(obj map[string]any, value map[string]string, fields ...string) error {
	return SetNestedTypedMap[string](obj, value, fields...)
}

// SetNestedMap sets the map[string]any value of a nested field.
// Returns an error if value cannot be set because one of the nesting levels is not a map[string]any.
func SetNestedMap(obj map[string]any, value map[string]any, fields ...string) error {
	return SetNestedField(obj, value, fields...)
}

// RemoveNestedField removes the nested field from the obj.
func RemoveNestedField(obj map[string]any, fields ...string) {
	m := obj
	for _, field := range fields[:len(fields)-1] {
		if x, ok := m[field].(map[string]any); ok {
			m = x
		} else {
			return
		}
	}
	delete(m, fields[len(fields)-1])
}

func getNestedString(obj map[string]any, fields ...string) string {
	val, found, err := NestedString(obj, fields...)
	if !found || err != nil {
		return ""
	}
	return val
}

func getNestedInt64Pointer(obj map[string]any, fields ...string) *int64 {
	val, found, err := NestedInt64(obj, fields...)
	if !found || err != nil {
		return nil
	}
	return &val
}

func jsonPath(fields []string) string {
	return "." + strings.Join(fields, ".")
}

func extractOwnerReference(v map[string]any) metav1.OwnerReference {
	// though this field is a *bool, but when decoded from JSON, it's
	// unmarshalled as bool.
	var controllerPtr *bool
	if controller, found, err := NestedBool(v, "controller"); err == nil && found {
		controllerPtr = &controller
	}
	var blockOwnerDeletionPtr *bool
	if blockOwnerDeletion, found, err := NestedBool(v, "blockOwnerDeletion"); err == nil && found {
		blockOwnerDeletionPtr = &blockOwnerDeletion
	}
	return metav1.OwnerReference{
		Kind:               getNestedString(v, "kind"),
		Name:               getNestedString(v, "name"),
		APIVersion:         getNestedString(v, "apiVersion"),
		UID:                types.UID(getNestedString(v, "uid")),
		Controller:         controllerPtr,
		BlockOwnerDeletion: blockOwnerDeletionPtr,
	}
}

// UnstructuredJSONScheme is capable of converting JSON data into the Unstructured
// type, which can be used for generic access to objects without a predefined scheme.
// TODO: move into serializer/json.
var UnstructuredJSONScheme runtime.Codec = unstructuredJSONScheme{}

type unstructuredJSONScheme struct{}

const unstructuredJSONSchemeIdentifier runtime.Identifier = "unstructuredJSON"

func (s unstructuredJSONScheme) Decode(data []byte, _ *schema.GroupVersionKind, obj runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
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
		return nil, &gvk, runtime.NewMissingKindErr(string(data))
	}
	// TODO(109023): require apiVersion here as well

	return obj, &gvk, nil
}

func (s unstructuredJSONScheme) Encode(obj runtime.Object, w io.Writer) error {
	if co, ok := obj.(runtime.CacheableObject); ok {
		return co.CacheEncode(s.Identifier(), s.doEncode, w)
	}
	return s.doEncode(obj, w)
}

func (unstructuredJSONScheme) doEncode(obj runtime.Object, w io.Writer) error {
	switch t := obj.(type) {
	case *Unstructured:
		return json.NewEncoder(w).Encode(t.Object)
	case *UnstructuredList:
		items := make([]any, 0, len(t.Items))
		for _, i := range t.Items {
			items = append(items, i.Object)
		}
		listObj := make(map[string]any, len(t.Object)+1)
		for k, v := range t.Object { // Make a shallow copy
			listObj[k] = v
		}
		listObj["items"] = items
		return json.NewEncoder(w).Encode(listObj)
	case *runtime.Unknown:
		// TODO: Unstructured needs to deal with ContentType.
		_, err := w.Write(t.Raw)
		return err
	default:
		return json.NewEncoder(w).Encode(t)
	}
}

// Identifier implements runtime.Encoder interface.
func (unstructuredJSONScheme) Identifier() runtime.Identifier {
	return unstructuredJSONSchemeIdentifier
}

func (s unstructuredJSONScheme) decode(data []byte) (runtime.Object, error) {
	type detector struct {
		Items gojson.RawMessage `json:"items"`
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

func (s unstructuredJSONScheme) decodeInto(data []byte, obj runtime.Object) error {
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
	m := make(map[string]any)
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}

	unstruct.Object = m

	return nil
}

func (s unstructuredJSONScheme) decodeToList(data []byte, list *UnstructuredList) error {
	type decodeList struct {
		Items []gojson.RawMessage `json:"items"`
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
	list.Items = make([]Unstructured, 0, len(dList.Items))
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
		list.Items = append(list.Items, *unstruct)
	}
	return nil
}

type jsonFallbackEncoder struct {
	encoder    runtime.Encoder
	identifier runtime.Identifier
}

func NewJSONFallbackEncoder(encoder runtime.Encoder) runtime.Encoder {
	result := map[string]string{
		"name": "fallback",
		"base": string(encoder.Identifier()),
	}
	identifier, err := gojson.Marshal(result)
	if err != nil {
		klog.Fatalf("Failed marshaling identifier for jsonFallbackEncoder: %v", err)
	}
	return &jsonFallbackEncoder{
		encoder:    encoder,
		identifier: runtime.Identifier(identifier),
	}
}

func (c *jsonFallbackEncoder) Encode(obj runtime.Object, w io.Writer) error {
	// There is no need to handle runtime.CacheableObject, as we only
	// fallback to other encoders here.
	err := c.encoder.Encode(obj, w)
	if runtime.IsNotRegisteredError(err) {
		switch obj.(type) {
		case *Unstructured, *UnstructuredList:
			return UnstructuredJSONScheme.Encode(obj, w)
		}
	}
	return err
}

// Identifier implements runtime.Encoder interface.
func (c *jsonFallbackEncoder) Identifier() runtime.Identifier {
	return c.identifier
}
