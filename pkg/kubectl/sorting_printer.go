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
	"io"
	"reflect"
	"sort"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/jsonpath"

	"github.com/golang/glog"
)

// Sorting printer sorts list types before delegating to another printer.
// Non-list types are simply passed through
type SortingPrinter struct {
	SortField string
	Delegate  ResourcePrinter
}

func (s *SortingPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	if !runtime.IsListType(obj) {
		return s.Delegate.PrintObj(obj, out)
	}

	if err := s.sortObj(obj); err != nil {
		return err
	}
	return s.Delegate.PrintObj(obj, out)
}

// TODO: implement HandledResources()
func (p *SortingPrinter) HandledResources() []string {
	return []string{}
}

func (s *SortingPrinter) sortObj(obj runtime.Object) error {
	objs, err := runtime.ExtractList(obj)
	if err != nil {
		return err
	}
	if len(objs) == 0 {
		return nil
	}
	parser := jsonpath.New("sorting")
	parser.Parse(s.SortField)

	for ix := range objs {
		item := objs[ix]
		switch u := item.(type) {
		case *runtime.Unknown:
			var err error
			if objs[ix], err = api.Codec.Decode(u.RawJSON); err != nil {
				return err
			}
		}
	}
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

func isLess(i, j reflect.Value) (bool, error) {
	switch i.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return i.Int() < j.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return i.Uint() < j.Uint(), nil
	case reflect.Float32, reflect.Float64:
		return i.Float() < j.Float(), nil
	case reflect.String:
		return i.String() < j.String(), nil
	case reflect.Ptr:
		return isLess(i.Elem(), j.Elem())
	default:
		return false, fmt.Errorf("unsortable type: %v", i.Kind())
	}
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

	less, err := isLess(iField, jField)
	if err != nil {
		glog.Fatalf("Field %s in %v is an unsortable type: %s, err: %v", r.field, iObj, iField.Kind().String(), err)
	}
	return less
}
