/*
Copyright 2018 The Kubernetes Authors.

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

package internal

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/value"
)

// DecodeObjectManagedFields extracts and converts the objects ManagedFields into a fieldpath.ManagedFields.
func DecodeObjectManagedFields(from runtime.Object) (fieldpath.ManagedFields, error) {
	if from == nil {
		return make(map[string]*fieldpath.VersionedSet), nil
	}
	accessor, err := meta.Accessor(from)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	accessor.SetManagedFields(nil)

	managed, err := decodeManagedFields(accessor.GetManagedFields())
	if err != nil {
		return nil, fmt.Errorf("failed to convert managed fields from API: %v", err)
	}
	return managed, err
}

// EncodeObjectManagedFields converts and stores the fieldpathManagedFields into the objects ManagedFields
func EncodeObjectManagedFields(obj runtime.Object, fields fieldpath.ManagedFields) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return fmt.Errorf("couldn't get accessor: %v", err)
	}

	managed, err := encodeManagedFields(fields)
	if err != nil {
		return fmt.Errorf("failed to convert back managed fields to API: %v", err)
	}
	accessor.SetManagedFields(managed)

	return nil
}

// decodeManagedFields converts ManagedFields from the wire format (api format)
// to the format used by sigs.k8s.io/structured-merge-diff
func decodeManagedFields(encodedManagedFields map[string]metav1.VersionedFieldSet) (managedFields fieldpath.ManagedFields, err error) {
	managedFields = make(map[string]*fieldpath.VersionedSet, len(encodedManagedFields))
	for manager, encodedVersionedSet := range encodedManagedFields {
		managedFields[manager], err = decodeVersionedSet(&encodedVersionedSet)
		if err != nil {
			return nil, fmt.Errorf("error decoding versioned set for %v: %v", manager, err)
		}
	}
	return managedFields, nil
}

func decodeVersionedSet(encodedVersionedSet *metav1.VersionedFieldSet) (versionedSet *fieldpath.VersionedSet, err error) {
	versionedSet = &fieldpath.VersionedSet{}
	versionedSet.APIVersion = fieldpath.APIVersion(encodedVersionedSet.APIVersion)
	versionedSet.Set = &fieldpath.Set{}
	err = decodeSetInto(&encodedVersionedSet.Fields, versionedSet.Set)
	if err != nil {
		return nil, fmt.Errorf("error decoding set: %v", err)
	}
	return versionedSet, nil
}

func decodeSetInto(encodedSet *metav1.FieldSet, set *fieldpath.Set) (err error) {
	m, err := decodePathElementSet(encodedSet.Members)
	if err != nil {
		return fmt.Errorf("error decoding path element set: %v", err)
	}
	set.Members = *m
	c, err := decodeSetNodeMap(encodedSet.Children)
	if err != nil {
		return fmt.Errorf("error decoding children: %v", err)
	}
	set.Children = *c
	return nil
}

func decodePathElementSet(encodedPathElementSet []metav1.FieldPathElement) (pathElementSet *fieldpath.PathElementSet, err error) {
	pathElementSet = &fieldpath.PathElementSet{}
	for _, encodedPathElement := range encodedPathElementSet {
		pathElement, err := decodePathElement(&encodedPathElement)
		if err != nil {
			return nil, fmt.Errorf("error decoding path element: %v", err)
		}
		pathElementSet.Insert(*pathElement)
	}
	return pathElementSet, nil
}

func decodeSetNodeMap(encodedSetNodeMap []metav1.FieldSetNode) (setNodeMap *fieldpath.SetNodeMap, err error) {
	setNodeMap = &fieldpath.SetNodeMap{}
	for _, encodedSetNode := range encodedSetNodeMap {
		pathElement, err := decodePathElement(&encodedSetNode.PathElement)
		if err != nil {
			return nil, fmt.Errorf("error decoding path element: %v", err)
		}
		set := setNodeMap.Descend(*pathElement)
		decodeSetInto(encodedSetNode.Set, set)
	}
	return setNodeMap, nil
}

func decodePathElement(encodedPathElement *metav1.FieldPathElement) (pathElement *fieldpath.PathElement, err error) {
	if encodedPathElement == nil {
		return nil, nil
	}
	if !validateOneOf(encodedPathElement.FieldName, encodedPathElement.Key, encodedPathElement.Value, encodedPathElement.Index) {
		return nil, fmt.Errorf("too many fields set: %v", encodedPathElement)
	}

	pathElement = &fieldpath.PathElement{}
	pathElement.FieldName = encodedPathElement.FieldName
	if encodedPathElement.Key != nil {
		pathElement.Key = make([]value.Field, len(encodedPathElement.Key))
		for i, encodedField := range encodedPathElement.Key {
			pathElement.Key[i], err = decodeField(&encodedField)
			if err != nil {
				return nil, fmt.Errorf("error decoding field: %v", err)
			}
		}
	}
	pathElement.Value, err = decodeValue(encodedPathElement.Value)
	if err != nil {
		return nil, fmt.Errorf("error decoding value: %v", err)
	}
	if encodedPathElement.Index != nil {
		v := int(*encodedPathElement.Index)
		pathElement.Index = &v
	}
	return pathElement, nil
}

func decodeField(encodedField *metav1.FieldNameValuePair) (field value.Field, err error) {
	field = value.Field{}
	if encodedField == nil {
		return field, fmt.Errorf("unexpected nil pointer")
	}
	field.Name = encodedField.Name
	v, err := decodeValue(&encodedField.Value)
	if err != nil {
		return field, fmt.Errorf("error decoding value: %v", err)
	}
	if v != nil {
		field.Value = *v
	}
	return field, nil
}

func decodeValue(encodedValue *metav1.FieldValue) (v *value.Value, err error) {
	if encodedValue == nil {
		return nil, nil
	}
	if !validateOneOf(encodedValue.FloatValue, encodedValue.IntValue, encodedValue.StringValue, encodedValue.BooleanValue) {
		return nil, fmt.Errorf("too many fields set: %v", encodedValue)
	}

	v = &value.Value{}
	if encodedValue.FloatValue != nil {
		f, err := strconv.ParseFloat(*encodedValue.FloatValue, 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing float: %v", err)
		}
		v.FloatValue = (*value.Float)(&f)
	}
	if encodedValue.IntValue != nil {
		i := int64(*encodedValue.IntValue)
		v.IntValue = (*value.Int)(&i)
	}
	v.StringValue = (*value.String)(encodedValue.StringValue)
	v.BooleanValue = (*value.Boolean)(encodedValue.BooleanValue)
	v.Null = encodedValue.Null
	return v, nil
}

func validateOneOf(fields ...interface{}) bool {
	nonNilCount := 0
	for _, field := range fields {
		if !reflect.ValueOf(field).IsNil() {
			nonNilCount++
		}
	}
	return nonNilCount <= 1
}

// encodeManagedFields converts ManagedFields from the the format used by
// sigs.k8s.io/structured-merge-diff to the the wire format (api format)
func encodeManagedFields(managedFields fieldpath.ManagedFields) (encodedManagedFields map[string]metav1.VersionedFieldSet, err error) {
	encodedManagedFields = make(map[string]metav1.VersionedFieldSet, len(managedFields))
	for manager, versionedSet := range managedFields {
		v, err := encodeVersionedSet(versionedSet)
		if err != nil {
			return nil, fmt.Errorf("error encoding versioned set for %v: %v", manager, err)
		}
		encodedManagedFields[manager] = *v
	}
	return encodedManagedFields, nil
}

func encodeVersionedSet(versionedSet *fieldpath.VersionedSet) (encodedVersionedSet *metav1.VersionedFieldSet, err error) {
	encodedVersionedSet = &metav1.VersionedFieldSet{}
	encodedVersionedSet.APIVersion = string(versionedSet.APIVersion)
	f, err := encodeSet(versionedSet.Set)
	if err != nil {
		return nil, fmt.Errorf("error encoding set: %v", err)
	}
	encodedVersionedSet.Fields = *f
	return encodedVersionedSet, nil
}

func encodeSet(set *fieldpath.Set) (encodedSet *metav1.FieldSet, err error) {
	encodedSet = &metav1.FieldSet{}
	encodedSet.Members, err = encodePathElementSet(&set.Members)
	if err != nil {
		return nil, fmt.Errorf("error encoding path element set: %v", err)
	}
	encodedSet.Children, err = encodeSetNodeMap(&set.Children)
	if err != nil {
		return nil, fmt.Errorf("error encoding children: %v", err)
	}
	return encodedSet, nil
}

func encodePathElementSet(pathElementSet *fieldpath.PathElementSet) (encodedPathElementSet []metav1.FieldPathElement, err error) {
	// Get a list of the elements in the pathElementSet and sort it
	pathElementList := make(sortablePathElementList, 0)
	pathElementSet.Iterate(func(pathElement fieldpath.PathElement) {
		pathElementList = append(pathElementList, &pathElement)
	})
	sort.Sort(pathElementList)
	if len(pathElementList) == 0 {
		return nil, nil
	}

	encodedPathElementSet = make([]metav1.FieldPathElement, len(pathElementList))
	for i, pathElement := range pathElementList {
		encodedPathElement, err := encodePathElement(pathElement)
		if err != nil {
			return nil, fmt.Errorf("error encoding path element: %v", err)
		}
		encodedPathElementSet[i] = encodedPathElement
	}
	return encodedPathElementSet, nil
}

func encodeSetNodeMap(setNodeMap *fieldpath.SetNodeMap) (encodedSetNodeMap []metav1.FieldSetNode, err error) {
	// Get a list of the keys in the setNodeMap and sort it
	pathElementList := make(sortablePathElementList, 0)
	setNodeMap.Iterate(func(pathElement fieldpath.PathElement) {
		pathElementList = append(pathElementList, &pathElement)
	})
	sort.Sort(pathElementList)
	if len(pathElementList) == 0 {
		return nil, nil
	}

	encodedSetNodeMap = make([]metav1.FieldSetNode, len(pathElementList))
	for i, pathElement := range pathElementList {
		encodedPathElement, err := encodePathElement(pathElement)
		if err != nil {
			return nil, fmt.Errorf("error encoding path element %v: %v", pathElement.String(), err)
		}
		set, ok := setNodeMap.Get(*pathElement)
		if !ok {
			return nil, fmt.Errorf("error looking up path element %v", pathElement.String())
		}
		encodedSet, err := encodeSet(set)
		if err != nil {
			return nil, fmt.Errorf("error encoding set for %v: %v", pathElement.String(), err)
		}
		encodedSetNodeMap[i] = metav1.FieldSetNode{
			PathElement: encodedPathElement,
			Set:         encodedSet,
		}
	}
	return encodedSetNodeMap, nil
}

func encodePathElement(pathElement *fieldpath.PathElement) (encodedPathElement metav1.FieldPathElement, err error) {
	encodedPathElement = metav1.FieldPathElement{}
	if pathElement == nil {
		return encodedPathElement, nil
	}
	encodedPathElement.FieldName = pathElement.FieldName
	if pathElement.Key != nil {
		encodedPathElement.Key = make([]metav1.FieldNameValuePair, len(pathElement.Key))
		for i, field := range pathElement.Key {
			encodedPathElement.Key[i], err = encodeField(&field)
			if err != nil {
				return encodedPathElement, fmt.Errorf("error encoding field: %v", err)
			}
		}
	}
	encodedPathElement.Value, err = encodeValue(pathElement.Value)
	if err != nil {
		return encodedPathElement, fmt.Errorf("error encoding value: %v", err)
	}
	if pathElement.Index != nil {
		v := int32(*pathElement.Index)
		encodedPathElement.Index = &v
	}
	return encodedPathElement, nil
}

func encodeField(field *value.Field) (encodedField metav1.FieldNameValuePair, err error) {
	encodedField = metav1.FieldNameValuePair{}
	if field == nil {
		return encodedField, fmt.Errorf("unexpected nil pointer")
	}
	encodedField.Name = field.Name
	v, err := encodeValue(&field.Value)
	if err != nil {
		return encodedField, fmt.Errorf("error encoding value: %v", err)
	}
	if v != nil {
		encodedField.Value = *v
	}
	return encodedField, nil
}

func encodeValue(v *value.Value) (encodedValue *metav1.FieldValue, err error) {
	if v == nil {
		return nil, nil
	}
	encodedValue = &metav1.FieldValue{}
	if v.FloatValue != nil {
		f := strconv.FormatFloat(float64(*v.FloatValue), 'f', -1, 64)
		encodedValue.FloatValue = &f
	}
	if v.IntValue != nil {
		i := int32(int64(*v.IntValue))
		encodedValue.IntValue = &i
	}
	encodedValue.StringValue = (*string)(v.StringValue)
	encodedValue.BooleanValue = (*bool)(v.BooleanValue)
	encodedValue.Null = v.Null
	return encodedValue, nil
}

type sortablePathElementList []*fieldpath.PathElement

func (p sortablePathElementList) Len() int           { return len(p) }
func (p sortablePathElementList) Less(i, j int) bool { return p[i].String() < p[j].String() }
func (p sortablePathElementList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
