/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"fmt"
	"reflect"
	"sort"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/jsonpath"
)

// Sorting preprocessor sorts list types. Non-list types are simply passed through
// WARNING: it modifies the object.
// TODO: do not modify the object.
type SortingPreprocessor struct {
	SortField string
	Delegate  ResourcePrinter
}

func (s SortingPreprocessor) Process(obj runtime.Object) (runtime.Object, error) {
	if !runtime.IsListType(obj) {
		return obj, nil
	}
	if err := s.sortObj(obj); err != nil {
		return nil, err
	}
	return obj, nil
}

func (s *SortingPreprocessor) sortObj(obj runtime.Object) error {
	objs, err := runtime.ExtractList(obj)
	if err != nil {
		return err
	}
	if len(objs) == 0 {
		return nil
	}
	parser := jsonpath.New("sorting")
	parser.Parse(s.SortField)
	values, err := parser.FindResults(reflect.ValueOf(objs[0]).Elem().Interface())
	if err != nil {
		return err
	}
	if len(values) == 0 {
		return fmt.Errorf("couldn't find any field with path: %s", s.SortField)
	}
	sorter := &RuntimeSort{
		field: s.SortField,
		objs:  objs,
	}
	sort.Sort(sorter)
	runtime.SetList(obj, sorter.objs)
	return nil
}

// RuntimeSort is an implementation of the golang sort interface that knows how to sort
// lists of runtime.Object
type RuntimeSort struct {
	field string
	objs  []runtime.Object
}

func (r *RuntimeSort) Len() int {
	return len(r.objs)
}

func (r *RuntimeSort) Swap(i, j int) {
	r.objs[i], r.objs[j] = r.objs[j], r.objs[i]
}

func (r *RuntimeSort) Less(i, j int) bool {
	iObj := r.objs[i]
	jObj := r.objs[j]

	parser := jsonpath.New("sorting")
	parser.Parse(r.field)

	iValues, err := parser.FindResults(reflect.ValueOf(iObj).Elem().Interface())
	if err != nil {
		glog.Fatalf("Failed to get i values for %#v using %s (%#v)", iObj, r.field, err)
	}
	jValues, err := parser.FindResults(reflect.ValueOf(jObj).Elem().Interface())
	if err != nil {
		glog.Fatalf("Failed to get j values for %#v using %s (%v)", jObj, r.field, err)
	}

	iField := iValues[0][0]
	jField := jValues[0][0]

	switch iField.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return iField.Int() < jField.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return iField.Uint() < jField.Uint()
	case reflect.Float32, reflect.Float64:
		return iField.Float() < jField.Float()
	case reflect.String:
		return iField.String() < jField.String()
	default:
		glog.Fatalf("Field %s in %v is an unsortable type: %s", r.field, iObj, iField.Kind().String())
	}
	// default to preserving order
	return i < j
}
