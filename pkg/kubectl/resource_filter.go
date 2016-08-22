package kubectl

import (
	"fmt"
	"reflect"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

type filterEntry struct {
	filterFunc reflect.Value
}

// FilterOptions implements PrintOptions
type FilterOptions struct {
	PrintOptions
}

type ResourceFilter struct {
	filterMap map[reflect.Type]*filterEntry
	options   FilterOptions
}

func NewResourceFilter(opts *FilterOptions) *ResourceFilter {
	filter := &ResourceFilter{
		filterMap: make(map[reflect.Type]*filterEntry),
		options:   *opts,
	}

	filter.addDefaultHandlers()
	return filter
}

func (f *ResourceFilter) addDefaultHandlers() {
	f.Handler(filterPod)
}

// Handler adds a filter handler to a ResourceFilter instance.
// See validateFilterFunc for required method signature.
func (f *ResourceFilter) Handler(filterFunc interface{}) error {
	filterFuncValue := reflect.ValueOf(filterFunc)
	if err := f.validateFilterFunc(filterFuncValue); err != nil {
		glog.Errorf("Unable to add print handler: %v", err)
		return err
	}
	objType := filterFuncValue.Type().In(0)
	f.filterMap[objType] = &filterEntry{
		filterFunc: filterFuncValue,
	}
	return nil
}

// validateFilterFunc validates the filter handler signature.
// filterFunc is the function that will be called to filter an object.
// It must be of the following type:
//  func filterFunc(object ObjectType, options FilterOptions) error
// where ObjectType is the type of the object that will be printed.
func (f *ResourceFilter) validateFilterFunc(filterFunc reflect.Value) error {
	if filterFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid filter handler. %#v is not a function", filterFunc)
	}
	funcType := filterFunc.Type()
	if funcType.NumIn() != 2 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid filter handler." +
			"Must accept 2 parameters and return 1 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*FilterOptions)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*bool)(nil)).Elem() {
		return fmt.Errorf("invalid filter handler. The expected signature is: "+
			"func handler(obj %v, options FilterOptions) error", funcType.In(0))
	}
	return nil
}

// filterPod returns true if a pod should be skipped.
// defaults to true for terminated pods
func filterPod(pod *api.Pod, options FilterOptions) bool {
	reason := string(pod.Status.Phase)
	if pod.Status.Reason != "" {
		reason = pod.Status.Reason
	}
	if !options.ShowAll && (reason == string(api.PodSucceeded) || reason == string(api.PodFailed)) {
		return true
	}
	return false
}

// Filter extracts the filter handler, if one exists, for the given resource
func (f *ResourceFilter) Filter(obj runtime.Object) (bool, error) {
	t := reflect.TypeOf(obj)
	if filter := f.filterMap[t]; filter != nil {
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(f.options)}
		resultValue := filter.filterFunc.Call(args)[0]
		return resultValue.Interface().(bool), nil
	}

	return false, fmt.Errorf("error: no filter for type %#v", obj)
}
