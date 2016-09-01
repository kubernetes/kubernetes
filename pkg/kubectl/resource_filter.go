package kubectl

import (
	"fmt"
	"io"
	"reflect"

	"k8s.io/kubernetes/pkg/api"
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
}

func (f *filterOptions) AddFilter(objType reflect.Type, handlerFn FilterFunc) error {
	f.filterMap[objType] = handlerFn
	return nil
}

// filterPods is a FilterFunc type implementation.
// returns true if a pod should be skipped. Defaults to true for terminated pods
func filterPods(obj runtime.Object, options PrintOptions) bool {
	if pod, ok := obj.(*api.Pod); ok {
		reason := string(pod.Status.Phase)
		if pod.Status.Reason != "" {
			reason = pod.Status.Reason
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
	if filter, ok := f.filterMap[t]; ok {
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(f.options)}
		filterFunc := reflect.ValueOf(filter)
		resultValue := filterFunc.Call(args)[0]
		isFiltered := resultValue.Interface().(bool)
		if isFiltered {
			f.hiddenObjNum++
		}
		return isFiltered, nil
	}

	return false, fmt.Errorf("error: no filter for type %#v", t)
}
