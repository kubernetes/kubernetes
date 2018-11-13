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
	"sort"
	"strconv"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/value"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EncodeManagedFields converts ManagedFields from the the format used by
// sigs.k8s.io/structured-merge-diff to the the wire format (api format)
func EncodeManagedFields(managedFields fieldpath.ManagedFields) (encodedManagedFields map[string]metav1.VersionedFieldSet, err error) {
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
