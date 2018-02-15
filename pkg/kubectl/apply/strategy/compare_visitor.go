/*
Copyright 2017 The Kubernetes Authors.

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

package strategy

import (
	"reflect"

	"k8s.io/kubernetes/pkg/kubectl/apply"
)

// CreateCompareStrategy return compare strategy instance
func CreateCompareStrategy() apply.Compare {
	return compareStrategy{}
}

type compareStrategy struct {
}

// CompareMap recursively detects conflicts between recorded and remote in MapElement.
// Return false if conflicts detected.
func (v compareStrategy) CompareMap(e apply.MapElement) bool {
	if e.HasRecorded() && e.HasRemote() {
		record := e.GetRecordedMap()
		remote := e.GetRemoteMap()
		for key, value := range record {
			if val, ok := remote[key]; ok {
				if !v.hasNext(value, val) {
					return false
				}
			}
		}
	}
	return true
}

// CompareList recursively detects conflicts between recorded and remote in ListElement.
// Return false if conflicts detected.
func (v compareStrategy) CompareList(e apply.ListElement) bool {
	if e.HasRecorded() && e.HasRemote() {
		record := e.GetRecordedList()
		remote := e.GetRemoteList()
		if keys := e.GetFieldMergeKeys(); keys != nil {
			return v.compareMapList(record, remote, keys)
		}
		return v.comparePrimitiveList(record, remote)
	}
	return true
}

// compareList compares list in which element is object, like map, list, etc.
func (v compareStrategy) compareMapList(record, remote []interface{}, keys apply.MergeKeys) bool {
	if len(record) == 0 || len(remote) == 0 {
		return true
	}
	list1, list2 := sort(record, remote)
	var found bool
	for _, l2 := range list2 {
		found = false
		for _, l1 := range list1 {
			mergeKeyValueRecord, _ := keys.GetMergeKeyValue(l1)
			mergeKeyValueRemote, _ := keys.GetMergeKeyValue(l2)
			if mergeKeyValueRecord.Equal(mergeKeyValueRemote) {
				if v.hasNext(l1, l2) {
					found = true
					break
				}
			}
		}
	}
	return found
}

// comparePrimitiveList compare list in which elements are treated as primitive
func (v compareStrategy) comparePrimitiveList(record, remote []interface{}) bool {
	if len(record) == 0 || len(remote) == 0 {
		return true
	}
	list1, list2 := sort(record, remote)
	var found bool
	for _, value := range list2 {
		found = false
		for _, val := range list1 {
			if reflect.DeepEqual(value, val) {
				found = true
				break
			}
		}
	}
	return found
}

// hasNext recursively compare two values based on their type
func (v compareStrategy) hasNext(value, val interface{}) bool {
	if reflect.TypeOf(value).Kind() != reflect.TypeOf(val).Kind() {
		return false
	}
	switch value.(type) {
	case apply.ListElement:
		return value.(apply.ListElement).Compare(v) && val.(apply.ListElement).Compare(v)
	case apply.MapElement:
		return value.(apply.MapElement).Compare(v) && val.(apply.MapElement).Compare(v)
	case apply.TypeElement:
		return value.(apply.TypeElement).Compare(v) && val.(apply.TypeElement).Compare(v)
	case apply.PrimitiveElement:
		return value.(apply.PrimitiveElement).Compare(v) && val.(apply.PrimitiveElement).Compare(v)
	}
	return reflect.DeepEqual(value, val)
}

// CompareEmpty compares emptyElement and return true always
func (v compareStrategy) CompareEmpty(e apply.EmptyElement) bool {
	// Always return true, because empty element has no conflicts
	return true
}

// ComparePrimitive compares PrimitiveElement
func (v compareStrategy) ComparePrimitive(e apply.PrimitiveElement) bool {
	return reflect.DeepEqual(e.GetRecorded(), e.GetRemote())
}

// CompareType compares TypeElement
func (v compareStrategy) CompareType(e apply.TypeElement) bool {
	m := apply.MapElement{
		FieldMetaImpl:  e.FieldMetaImpl,
		MapElementData: e.MapElementData,
		Values:         e.Values,
	}
	return v.CompareMap(m)
}

// sort two lists by length
func sort(arg1, arg2 []interface{}) (l1, l2 []interface{}) {
	l1 = arg1
	l2 = arg2
	if len(arg2) > len(arg1) {
		l1 = arg2
		l2 = arg1
	}
	return l1, l2
}

func lookUpList(index int, list []interface{}) (interface{}, bool) {
	if list != nil && len(list) > index {
		return list[index], true
	}
	return nil, false
}
