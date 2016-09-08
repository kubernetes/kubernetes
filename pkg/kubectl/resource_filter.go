/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
)

// FilterFunc is a function that knows how to filter a specific resource kind.
// It receives a generic runtime.Object which must be type-checked by the function.
// Returns a boolean value true if a resource is filtered, or false otherwise.
type FilterFunc func(runtime.Object, PrintOptions) bool

// ResourceFilter filters api resources
type ResourceFilter interface {
	Filter(obj runtime.Object) (bool, error)
	AddFilter(reflect.Type, FilterFunc) error
	PrintFilterCount(io.Writer, string) error
}

// filterOptions is an implementation of ResourceFilter which contains
// a map of FilterFuncs for given resource types, PrintOptions for a
// resource, and a hiddenObjNum to keep track of the number of filtered resources
type filterOptions struct {
	filterMap    map[reflect.Type]FilterFunc
	options      PrintOptions
	hiddenObjNum int
}

func NewResourceFilter(opts *PrintOptions) ResourceFilter {
	filterOpts := &filterOptions{
		filterMap: make(map[reflect.Type]FilterFunc),
		options:   *opts,
	}

	filterOpts.addDefaultHandlers()
	return filterOpts
}

func (f *filterOptions) addDefaultHandlers() {
	f.AddFilter(reflect.TypeOf(&api.Pod{}), filterPods)
	f.AddFilter(reflect.TypeOf(&v1.Pod{}), filterPods)
}

func (f *filterOptions) AddFilter(objType reflect.Type, handlerFn FilterFunc) error {
	f.filterMap[objType] = handlerFn
	return nil
}

// filterPods is a FilterFunc type implementation.
// returns true if a pod should be skipped. Defaults to true for terminated pods
func filterPods(obj runtime.Object, options PrintOptions) bool {
	switch p := obj.(type) {
	case *v1.Pod:
		reason := string(p.Status.Phase)
		if p.Status.Reason != "" {
			reason = p.Status.Reason
		}
		return !options.ShowAll && (reason == string(v1.PodSucceeded) || reason == string(v1.PodFailed))
	case *api.Pod:
		reason := string(p.Status.Phase)
		if p.Status.Reason != "" {
			reason = p.Status.Reason
		}
		return !options.ShowAll && (reason == string(api.PodSucceeded) || reason == string(api.PodFailed))
	}

	return false
}

// PrintFilterCount prints an info message indicating the amount of resources
// that were skipped as a result of being filtered, and resets this count
func (f *filterOptions) PrintFilterCount(output io.Writer, res string) error {
	hiddenObjNum := f.hiddenObjNum
	f.hiddenObjNum = 0

	if !f.options.NoHeaders && !f.options.ShowAll && hiddenObjNum > 0 {
		_, err := fmt.Fprintf(output, "  info: %d completed object(s) was(were) not shown in %s list. Pass --show-all to see all objects.\n\n", hiddenObjNum, res)
		return err
	}
	return nil
}

// Filter extracts the filter handler, if one exists, for the given resource
func (f *filterOptions) Filter(obj runtime.Object) (bool, error) {
	t := reflect.TypeOf(obj)
	if filter, found := f.filterMap[t]; found {
		isFiltered := filter(obj, f.options)
		if isFiltered {
			f.hiddenObjNum++
		}
		return isFiltered, nil
	}
	return false, fmt.Errorf("error: no filter for type %#v", t)
}
