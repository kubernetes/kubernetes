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

package cache

import (
	"reflect"
	"regexp"
)

// makeValidPromethusMetricName converts a string into a valid Prometheus metric name.
// A valid metric name must match the regex [a-zA-Z_:][a-zA-Z0-9_:]*
func makeValidPromethusMetricName(name string) string {
	if len(name) == 0 {
		return "_"
	}

	var (
		invalidFirstChar = regexp.MustCompile(`[^a-zA-Z_:]`)
		invalidChar      = regexp.MustCompile(`[^a-zA-Z0-9_:]`)
	)

	first := invalidFirstChar.ReplaceAllString(name[:1], "_")

	if len(name) > 1 {
		rest := invalidChar.ReplaceAllString(name[1:], "_")
		return first + rest
	}

	return first
}

// getEventHandlerTypeName returns a simplified type name for the event handler
func getEventHandlerTypeName(handler ResourceEventHandler) string {
	t := reflect.TypeOf(handler)

	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	typeName := t.Name()

	if typeName == "" {
		if t.Kind() == reflect.Func {
			typeName = "func"
		} else {
			typeName = "anonymous"
		}
	}

	return typeName
}
