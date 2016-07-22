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

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/types"
)

type PtrMatcher struct {
	Matcher types.GomegaMatcher

	// Failure message.
	failure string
}

func (m *PtrMatcher) Match(actual interface{}) (bool, error) {
	val := reflect.ValueOf(actual)

	// return error if actual's type is incompatible with Transform function's argument type
	if val.Kind() != reflect.Ptr {
		return false, fmt.Errorf("PtrMatcher expects a pointer but we have '%s'", val.Kind())
	}

	if !val.IsValid() || val.IsNil() {
		m.failure = format.Message(actual, "not to be <nil>")
		return false, nil
	}

	// Forward the value.
	elem := val.Elem().Interface()
	match, err := m.Matcher.Match(elem)
	if !match {
		m.failure = m.Matcher.FailureMessage(elem)
	}
	return match, err
}

func (m *PtrMatcher) FailureMessage(_ interface{}) (message string) {
	return m.failure
}

func (m *PtrMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return m.Matcher.NegatedFailureMessage(actual)
}
