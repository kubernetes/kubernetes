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

package kunstruct

import (
	"testing"
)

func TestGetFieldValue(t *testing.T) {
	factory := NewKunstructuredFactoryImpl()
	kunstructured := factory.FromMap(map[string]interface{}{
		"Kind": "Service",
		"metadata": map[string]interface{}{
			"labels": map[string]string{
				"app": "application-name",
			},
			"name": "service-name",
		},
		"spec": map[string]interface{}{
			"ports": map[string]interface{}{
				"port": "80",
			},
		},
		"this": map[string]interface{}{
			"is": map[string]interface{}{
				"aNumber":    1000,
				"aNilValue":  nil,
				"anEmptyMap": map[string]interface{}{},
				"unrecognizable": testing.InternalExample{
					Name: "fooBar",
				},
			},
		},
	})

	tests := []struct {
		name          string
		pathToField   string
		expectedValue string
		errorExpected bool
		errorMsg      string
	}{
		{
			name:          "oneField",
			pathToField:   "Kind",
			expectedValue: "Service",
			errorExpected: false,
		},
		{
			name:          "twoFields",
			pathToField:   "metadata.name",
			expectedValue: "service-name",
			errorExpected: false,
		},
		{
			name:          "threeFields",
			pathToField:   "spec.ports.port",
			expectedValue: "80",
			errorExpected: false,
		},
		{
			name:          "empty",
			pathToField:   "",
			errorExpected: true,
			errorMsg:      "no field named ''",
		},
		{
			name:          "emptyDotEmpty",
			pathToField:   ".",
			errorExpected: true,
			errorMsg:      "no field named '.'",
		},
		{
			name:          "twoFieldsOneMissing",
			pathToField:   "metadata.banana",
			errorExpected: true,
			errorMsg:      "no field named 'metadata.banana'",
		},
		{
			name:          "deeperMissingField",
			pathToField:   "this.is.aDeep.field.that.does.not.exist",
			errorExpected: true,
			errorMsg:      "no field named 'this.is.aDeep.field.that.does.not.exist'",
		},
		{
			name:          "emptyMap",
			pathToField:   "this.is.anEmptyMap",
			errorExpected: true,
			errorMsg:      ".this.is.anEmptyMap accessor error: map[] is of the type map[string]interface {}, expected string",
		},
		{
			name:          "numberAsValue",
			pathToField:   "this.is.aNumber",
			errorExpected: true,
			errorMsg:      ".this.is.aNumber accessor error: 1000 is of the type int, expected string",
		},
		{
			name:          "nilAsValue",
			pathToField:   "this.is.aNilValue",
			errorExpected: true,
			errorMsg:      ".this.is.aNilValue accessor error: <nil> is of the type <nil>, expected string",
		},
		{
			name:          "unrecognizable",
			pathToField:   "this.is.unrecognizable.Name",
			errorExpected: true,
			errorMsg:      ".this.is.unrecognizable.Name accessor error: {fooBar <nil>  false} is of the type testing.InternalExample, expected map[string]interface{}",
		},
	}

	for _, test := range tests {
		s, err := kunstructured.GetFieldValue(test.pathToField)
		if test.errorExpected {
			if err == nil {
				t.Fatalf("%q; path %q - should return error, but no error returned",
					test.name, test.pathToField)
			}
			if test.errorMsg != err.Error() {
				t.Fatalf("%q; path %q - expected error: \"%s\", got error: \"%v\"",
					test.name, test.pathToField, test.errorMsg, err.Error())
			}
			continue
		}
		if err != nil {
			t.Fatalf("%q; path %q - unexpected error %v",
				test.name, test.pathToField, err)
		}
		if test.expectedValue != s {
			t.Fatalf("%q; Got: %s expected: %s",
				test.name, s, test.expectedValue)
		}

	}
}
