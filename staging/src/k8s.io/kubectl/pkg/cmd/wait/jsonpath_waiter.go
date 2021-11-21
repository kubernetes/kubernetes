package wait

import (
	errors2 "errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/util/jsonpath"
)

// JSONPathWaiter holds a JSONPath Parser which has the ability
// to check for the JSONPath condition and compare with the API server provided JSON output.
type JSONPathWaiter struct {
	jsonPathCondition string
	jsonPathParser    *jsonpath.JSONPath
	// errOut is written to if an error occurs
	errOut io.Writer
}

func NewJSONPathWaiter(jsonPathCond string, j *jsonpath.JSONPath, errOut io.Writer) Waiter {
	return &JSONPathWaiter{
		jsonPathCondition: jsonPathCond,
		jsonPathParser:    j,
		errOut:            errOut,
	}
}

func (j JSONPathWaiter) VisitResource(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	return getObjAndCheckCondition(info, o, j.isJSONPathConditionMet, j.checkCondition)
}

// IsJSONPathConditionMet fulfills the requirements of the interface ConditionFunc which provides condition check
//func (j JSONPathWaiter) IsJSONPathConditionMet(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
//	return getObjAndCheckCondition(info, o, j.isJSONPathConditionMet, j.checkCondition)
//}

func (j JSONPathWaiter) OnWaitLoopCompletion(visitedCount int, err error) error {
	if visitedCount == 0 {
		return errNoMatchingResources
	} else {
		return err
	}
}

// isJSONPathConditionMet is a helper function of IsJSONPathConditionMet
// which check the watch event and check if a JSONPathWaiter condition is met
func (j JSONPathWaiter) isJSONPathConditionMet(event watch.Event) (bool, error) {
	if event.Type == watch.Error {
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server
		err := errors.FromObject(event.Object)
		fmt.Fprintf(j.errOut, "error: An error occurred while waiting for the condition to be satisfied: %v", err)
		return false, nil
	}
	if event.Type == watch.Deleted {
		// this will chain back out, result in another get and an return false back up the chain
		return false, nil
	}
	// event runtime Object can be safely asserted to Unstructed
	// because we are working with dynamic client
	obj := event.Object.(*unstructured.Unstructured)
	return j.checkCondition(obj)
}

// checkCondition uses JSONPath parser to parse the JSON received from the API server
// and check if it matches the desired condition
func (j JSONPathWaiter) checkCondition(obj *unstructured.Unstructured) (bool, error) {
	queryObj := obj.UnstructuredContent()
	parseResults, err := j.jsonPathParser.FindResults(queryObj)
	if err != nil {
		return false, err
	}
	if err := verifyParsedJSONPath(parseResults); err != nil {
		return false, err
	}
	isConditionMet, err := compareResults(parseResults[0][0], j.jsonPathCondition)
	if err != nil {
		return false, err
	}
	return isConditionMet, nil
}

// verifyParsedJSONPath verifies the JSON received from the API server is valid.
// It will only accept a single JSON
func verifyParsedJSONPath(results [][]reflect.Value) error {
	if len(results) == 0 {
		return errors2.New("given jsonpath expression does not match any value")
	}
	if len(results) > 1 {
		return errors2.New("given jsonpath expression matches more than one list")
	}
	if len(results[0]) > 1 {
		return errors2.New("given jsonpath expression matches more than one value")
	}
	return nil
}

// compareResults will compare the reflect.Value from the result parsed by the
// JSONPath parser with the expected value given by the value
//
// Since this is coming from an unstructured this can only ever be a primitive,
// map[string]interface{}, or []interface{}.
// We do not support the last two and rely on fmt to handle conversion to string
// and compare the result with user input
func compareResults(r reflect.Value, expectedVal string) (bool, error) {
	switch r.Interface().(type) {
	case map[string]interface{}, []interface{}:
		return false, errors2.New("jsonpath leads to a nested object or list which is not supported")
	}
	s := fmt.Sprintf("%v", r.Interface())
	return strings.TrimSpace(s) == strings.TrimSpace(expectedVal), nil
}
