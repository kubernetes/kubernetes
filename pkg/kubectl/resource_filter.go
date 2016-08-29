package kubectl

import (
	"fmt"
	"io"
	"reflect"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
)

type FilterOptions struct {
	filterMap    map[reflect.Type]reflect.Value
	options      PrintOptions
	hiddenObjNum int
}

func NewResourceFilter(opts *PrintOptions) *FilterOptions {
	filterOpts := &FilterOptions{
		filterMap: make(map[reflect.Type]reflect.Value),
		options:   *opts,
	}

	filterOpts.addDefaultHandlers()
	return filterOpts
}

func (f *FilterOptions) addDefaultHandlers() {
	f.Handler(filterPods)
}

// Handler adds a filter handler to a FilterOptions instance.
// See validateFilterFunc for required method signature.
func (f *FilterOptions) Handler(handlerFn interface{}) error {
	handlerFnValue := reflect.ValueOf(handlerFn)
	if err := f.validateFilterFunc(handlerFnValue); err != nil {
		glog.Errorf("Unable to add filter handler: %v", err)
		return err
	}
	objType := handlerFnValue.Type().In(0)
	f.filterMap[objType] = handlerFnValue

	return nil
}

// validateFilterFunc validates the filter handler signature.
// filterFunc is the function that will be called to filter an object.
// It must be of the following type:
//  func filterFunc(object ObjectType, options PrintOptions) error
// where ObjectType is the type of the object that will be filtered.
func (f *FilterOptions) validateFilterFunc(filterFunc reflect.Value) error {
	if filterFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid filter handler. %#v is not a function", filterFunc)
	}
	funcType := filterFunc.Type()
	if funcType.NumIn() != 2 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid filter handler."+
			"Must accept 2 parameters and return 1 value,"+
			"but instead accepts %v parameter(s) and returns %v value(s)",
			funcType.NumIn(), funcType.NumOut())
	}
	if funcType.In(1) != reflect.TypeOf((*PrintOptions)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*bool)(nil)).Elem() {
		return fmt.Errorf("invalid filter handler. The expected signature is: "+
			"func handler(obj %v, options PrintOptions) error", funcType.In(0))
	}
	return nil
}

// FilterDeletedPods returns true if a pod should be skipped.
// defaults to true for terminated pods
func filterPods(pod *api.Pod, options PrintOptions) bool {
	reason := string(pod.Status.Phase)
	if pod.Status.Reason != "" {
		reason = pod.Status.Reason
	}
	if !options.ShowAll && (reason == string(api.PodSucceeded) || reason == string(api.PodFailed)) {
		return true
	}
	return false
}

// PrintFilterCount prints an info message indicating the amount of resources
// that were skipped as a result of being filtered, and resets this count
func (f *FilterOptions) PrintFilterCount(output io.Writer, res string) error {
	hiddenObjNum := f.hiddenObjNum
	f.hiddenObjNum = 0

	if !f.options.NoHeaders && !f.options.ShowAll && hiddenObjNum > 0 {
		_, err := fmt.Fprintf(output, "  info: %d completed object(s) was(were) not shown in %s list. Pass --show-all to see all objects.\n\n", hiddenObjNum, res)
		return err
	}
	return nil
}

// Filter extracts the filter handler, if one exists, for the given resource
func (f *FilterOptions) Filter(obj runtime.Object) (bool, error) {
	t := reflect.TypeOf(obj)
	if filter, ok := f.filterMap[t]; ok {
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(f.options)}
		resultValue := filter.Call(args)[0]
		isFiltered := resultValue.Interface().(bool)
		if isFiltered {
			f.hiddenObjNum++
		}
		return isFiltered, nil
	}

	return false, fmt.Errorf("error: no filter for type %#v", t)
}
