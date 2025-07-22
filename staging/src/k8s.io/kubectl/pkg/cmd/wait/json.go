/*
Copyright 2024 The Kubernetes Authors.

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

package wait

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/util/jsonpath"
)

// JSONPathWait holds a JSONPath Parser which has the ability
// to check for the JSONPath condition and compare with the API server provided JSON output.
type JSONPathWait struct {
	matchAnyValue  bool
	jsonPathValue  string
	jsonPathParser *jsonpath.JSONPath
	// errOut is written to if an error occurs
	errOut io.Writer
}

// IsJSONPathConditionMet fulfills the requirements of the interface ConditionFunc which provides condition check
func (j JSONPathWait) IsJSONPathConditionMet(ctx context.Context, info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	return getObjAndCheckCondition(ctx, info, o, j.isJSONPathConditionMet, j.checkCondition)
}

// isJSONPathConditionMet is a helper function of IsJSONPathConditionMet
// which check the watch event and check if a JSONPathWait condition is met
func (j JSONPathWait) isJSONPathConditionMet(event watch.Event) (bool, error) {
	if event.Type == watch.Error {
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server
		err := apierrors.FromObject(event.Object)
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
func (j JSONPathWait) checkCondition(obj *unstructured.Unstructured) (bool, error) {
	queryObj := obj.UnstructuredContent()
	parseResults, err := j.jsonPathParser.FindResults(queryObj)
	if err != nil {
		return false, err
	}
	if len(parseResults) == 0 || len(parseResults[0]) == 0 {
		return false, nil
	}
	if err := verifyParsedJSONPath(parseResults); err != nil {
		return false, err
	}
	if j.matchAnyValue {
		return true, nil
	}
	isConditionMet, err := compareResults(parseResults[0][0], j.jsonPathValue)
	if err != nil {
		return false, err
	}
	return isConditionMet, nil
}

// verifyParsedJSONPath verifies the JSON received from the API server is valid.
// It will only accept a single JSON
func verifyParsedJSONPath(results [][]reflect.Value) error {
	if len(results) > 1 {
		return errors.New("given jsonpath expression matches more than one list")
	}
	if len(results[0]) > 1 {
		return errors.New("given jsonpath expression matches more than one value")
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
		return false, errors.New("jsonpath leads to a nested object or list which is not supported")
	}
	s := fmt.Sprintf("%v", r.Interface())
	return strings.TrimSpace(s) == strings.TrimSpace(expectedVal), nil
}
