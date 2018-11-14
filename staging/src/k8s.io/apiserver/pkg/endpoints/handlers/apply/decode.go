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

package apply

import (
	"fmt"
	"reflect"
	"strconv"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/value"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DecodeManagedFields converts ManagedFields from the wire format (api format)
// to the format used by sigs.k8s.io/structured-merge-diff
func DecodeManagedFields(encodedManagedFields map[string]metav1.VersionedFieldSet) (managedFields fieldpath.ManagedFields, err error) {
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
