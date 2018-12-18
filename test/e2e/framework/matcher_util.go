/*
Copyright 2018 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"reflect"
	"runtime"
	"strings"

	"github.com/onsi/gomega/format"
)

type HaveOccurredMatcherAt struct {
}

func isError(a interface{}) bool {
	_, ok := a.(error)
	return ok
}

func isNil(a interface{}) bool {
	if a == nil {
		return true
	}

	switch reflect.TypeOf(a).Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return reflect.ValueOf(a).IsNil()
	}

	return false
}

func (matcher *HaveOccurredMatcherAt) Match(actual interface{}) (success bool, err error) {
	_, file, line, _ := runtime.Caller(1)
	// is purely nil?
	if actual == nil {
		return false, nil
	}

	// must be an 'error' type
	if !isError(actual) {
		return false, fmt.Errorf("Expected an error-type.  Got:\n%s\nAt: %s:%d", format.Object(actual, 1), chopPath(file), line)
	}

	// must be non-nil (or a pointer to a non-nil)
	return !isNil(actual), nil
}

// return the source filename after the last slash
func chopPath(original string) string {
	i := strings.LastIndex(original, "/")
	if i == -1 {
		return original
	} else {
		return original[i+1:]
	}
}

func (matcher *HaveOccurredMatcherAt) FailureMessage(actual interface{}) (message string) {
	_, file, line, _ := runtime.Caller(1)
	return fmt.Sprintf("Expected an error to have occurred.  Got:\n%s\nAt: %s:%d", format.Object(actual, 1), chopPath(file), line)
}

func (matcher *HaveOccurredMatcherAt) NegatedFailureMessage(actual interface{}) (message string) {
	_, file, line, _ := runtime.Caller(1)
	return fmt.Sprintf("Expected error:\n%s\n%s\n%s\nAt: %s:%d", format.Object(actual, 1), format.IndentString(actual.(error).Error(), 1), "not to have occurred", chopPath(file), line)
}
