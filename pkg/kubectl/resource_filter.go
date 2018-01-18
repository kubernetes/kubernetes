/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/printers"
)

// FilterFunc is a function that knows how to filter a specific resource kind.
// It receives a generic runtime.Object which must be type-checked by the function.
// Returns a boolean value true if a resource is filtered, or false otherwise.
type FilterFunc func(runtime.Object, printers.PrintOptions) bool

// Filters is a collection of filter funcs
type Filters []FilterFunc

func NewResourceFilter() Filters {
	return []FilterFunc{
		filterPods,
	}
}

// filterPods returns true if a pod should be skipped.
// If show-all is true, the pod will be never be skipped (return false);
// otherwise, skip terminated pod.
func filterPods(obj runtime.Object, options printers.PrintOptions) bool {
	if options.ShowAll {
		return false
	}

	switch p := obj.(type) {
	case *v1.Pod:
		return p.Status.Phase == v1.PodSucceeded || p.Status.Phase == v1.PodFailed
	case *api.Pod:
		return p.Status.Phase == api.PodSucceeded || p.Status.Phase == api.PodFailed
	}
	return false
}

// Filter loops through a collection of FilterFuncs until it finds one that can filter the given resource
func (f Filters) Filter(obj runtime.Object, opts *printers.PrintOptions) (bool, error) {
	for _, filter := range f {
		if ok := filter(obj, *opts); ok {
			return true, nil
		}
	}
	return false, nil
}
