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

package kubectl

import (
	"fmt"
	"io"
	"reflect"
	"sort"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/integer"
	"k8s.io/kubernetes/pkg/util/jsonpath"
)

// Sorting printer sorts list types before delegating to another printer.
// Non-list types are simply passed through
type SortingPrinter struct {
	SortField string
	Delegate  ResourcePrinter
	Decoder   runtime.Decoder
}

func (s *SortingPrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

func (s *SortingPrinter) PrintObj(obj runtime.Object, out io.Writer) error {
	if !meta.IsListType(obj) {
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
	objs, err := meta.ExtractList(obj)
	if err != nil {
		return err
	}
	if len(objs) == 0 {
		return nil
	}

	sorter, err := SortObjects(s.Decoder, objs, s.SortField)
	if err != nil {
		return err
	}

	switch list := obj.(type) {
	case *v1.List:
		outputList := make([]runtime.RawExtension, len(objs))
		for ix := range objs {
			outputList[ix] = list.Items[sorter.OriginalPosition(ix)]
		}
		list.Items = outputList
		return nil
	}
	return meta.SetList(obj, objs)
}

func SortObjects(decoder runtime.Decoder, objs []runtime.Object, fieldInput string) (*RuntimeSort, error) {
	parser := jsonpath.New("sorting")

	field, err := massageJSONPath(fieldInput)
	if err != nil {
		return nil, err
	}

	if err := parser.Parse(field); err != nil {
		return nil, err
	}

	for ix := range objs {
		item := objs[ix]
		switch u := item.(type) {
		case *runtime.Unknown:
			var err error
			// decode runtime.Unknown to runtime.Unstructured for sorting.
			// we don't actually want the internal versions of known types.
			if objs[ix], _, err = decoder.Decode(u.Raw, nil, &unstructured.Unstructured{}); err != nil {
				return nil, err
			}
		}
	}

	var values [][]reflect.Value
	if unstructured, ok := objs[0].(*unstructured.Unstructured); ok {
		values, err = parser.FindResults(unstructured.Object)
	} else {
		values, err = parser.FindResults(reflect.ValueOf(objs[0]).Elem().Interface())
	}

	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("couldn't find any field with path: %s", field)
	}

	sorter := NewRuntimeSort(field, objs)
	sort.Sort(sorter)
	return sorter, nil
}

// RuntimeSort is an implementation of the golang sort interface that knows how to sort
// lists of runtime.Object
type RuntimeSort struct {
	field        string
	objs         []runtime.Object
	origPosition []int
}

func NewRuntimeSort(field string, objs []runtime.Object) *RuntimeSort {
	sorter := &RuntimeSort{field: field, objs: objs, origPosition: make([]int, len(objs))}
	for ix := range objs {
		sorter.origPosition[ix] = ix
	}
	return sorter
}

func (r *RuntimeSort) Len() int {
	return len(r.objs)
}

func (r *RuntimeSort) Swap(i, j int) {
	r.objs[i], r.objs[j] = r.objs[j], r.objs[i]
	r.origPosition[i], r.origPosition[j] = r.origPosition[j], r.origPosition[i]
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
	case reflect.Struct:
		// sort metav1.Time
		in := i.Interface()
		if t, ok := in.(metav1.Time); ok {
			return t.Before(j.Interface().(metav1.Time)), nil
		}
		// fallback to the fields comparison
		for idx := 0; idx < i.NumField(); idx++ {
			less, err := isLess(i.Field(idx), j.Field(idx))
			if err != nil || !less {
				return less, err
			}
		}
		return true, nil
	case reflect.Array, reflect.Slice:
		// note: the length of i and j may be different
		for idx := 0; idx < integer.IntMin(i.Len(), j.Len()); idx++ {
			less, err := isLess(i.Index(idx), j.Index(idx))
			if err != nil || !less {
				return less, err
			}
		}
		return true, nil

	case reflect.Interface:
		switch itype := i.Interface().(type) {
		case uint8:
			if jtype, ok := j.Interface().(uint8); ok {
				return itype < jtype, nil
			}
		case uint16:
			if jtype, ok := j.Interface().(uint16); ok {
				return itype < jtype, nil
			}
		case uint32:
			if jtype, ok := j.Interface().(uint32); ok {
				return itype < jtype, nil
			}
		case uint64:
			if jtype, ok := j.Interface().(uint64); ok {
				return itype < jtype, nil
			}
		case int8:
			if jtype, ok := j.Interface().(int8); ok {
				return itype < jtype, nil
			}
		case int16:
			if jtype, ok := j.Interface().(int16); ok {
				return itype < jtype, nil
			}
		case int32:
			if jtype, ok := j.Interface().(int32); ok {
				return itype < jtype, nil
			}
		case int64:
			if jtype, ok := j.Interface().(int64); ok {
				return itype < jtype, nil
			}
		case uint:
			if jtype, ok := j.Interface().(uint); ok {
				return itype < jtype, nil
			}
		case int:
			if jtype, ok := j.Interface().(int); ok {
				return itype < jtype, nil
			}
		case float32:
			if jtype, ok := j.Interface().(float32); ok {
				return itype < jtype, nil
			}
		case float64:
			if jtype, ok := j.Interface().(float64); ok {
				return itype < jtype, nil
			}
		case string:
			if jtype, ok := j.Interface().(string); ok {
				return itype < jtype, nil
			}
		default:
			return false, fmt.Errorf("unsortable type: %T", itype)
		}
		return false, fmt.Errorf("unsortable interface: %v", i.Kind())

	default:
		return false, fmt.Errorf("unsortable type: %v", i.Kind())
	}
}

func (r *RuntimeSort) Less(i, j int) bool {
	iObj := r.objs[i]
	jObj := r.objs[j]

	parser := jsonpath.New("sorting")
	parser.Parse(r.field)

	var iValues [][]reflect.Value
	var jValues [][]reflect.Value
	var err error

	if unstructured, ok := iObj.(*unstructured.Unstructured); ok {
		iValues, err = parser.FindResults(unstructured.Object)
	} else {
		iValues, err = parser.FindResults(reflect.ValueOf(iObj).Elem().Interface())
	}
	if err != nil {
		glog.Fatalf("Failed to get i values for %#v using %s (%#v)", iObj, r.field, err)
	}

	if unstructured, ok := jObj.(*unstructured.Unstructured); ok {
		jValues, err = parser.FindResults(unstructured.Object)
	} else {
		jValues, err = parser.FindResults(reflect.ValueOf(jObj).Elem().Interface())
	}
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

// Returns the starting (original) position of a particular index.  e.g. If OriginalPosition(0) returns 5 than the
// the item currently at position 0 was at position 5 in the original unsorted array.
func (r *RuntimeSort) OriginalPosition(ix int) int {
	if ix < 0 || ix > len(r.origPosition) {
		return -1
	}
	return r.origPosition[ix]
}
