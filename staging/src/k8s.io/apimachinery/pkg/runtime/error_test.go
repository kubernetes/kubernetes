/*
Copyright 2023 The Kubernetes Authors.

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

package runtime

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

type testAPIObject struct{}

func TestNotRegisteredErr(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "test", Version: "v1", Kind: "TestType"}
	target := schema.GroupVersion{
		Group:   "test",
		Version: "v2",
	}
	const schemeName = "testScheme"
	testType := reflect.TypeOf(testAPIObject{})

	tests := []struct {
		name           string
		err            error
		expectedErrStr string
	}{
		{
			name:           "with type and target",
			err:            NewNotRegisteredErrForTarget(schemeName, testType, target),
			expectedErrStr: `runtime.testAPIObject is not suitable for converting to "test/v2" in scheme "testScheme"`,
		},
		{
			name:           "with GVK and target",
			err:            NewNotRegisteredGVKErrForTarget(schemeName, gvk, target),
			expectedErrStr: `"test/v1" is not suitable for converting to "test/v2" in scheme "testScheme"`,
		},
		{
			name:           "with only type",
			err:            NewNotRegisteredErrForType(schemeName, testType),
			expectedErrStr: `no kind is registered for the type runtime.testAPIObject in scheme "testScheme"`,
		},
		{
			name:           "with only group and version",
			err:            NewNotRegisteredErrForKind(schemeName, schema.GroupVersionKind{Group: "test", Version: "v1"}),
			expectedErrStr: `no version "test/v1" has been registered in scheme "testScheme"`,
		},
		{
			name:           "with version as APIVersionInternal",
			err:            NewNotRegisteredErrForKind(schemeName, schema.GroupVersionKind{Group: "test", Version: APIVersionInternal, Kind: "TestType"}),
			expectedErrStr: `no kind "TestType" is registered for the internal version of group "test" in scheme "testScheme"`,
		},
		{
			name:           "with only gvk",
			err:            NewNotRegisteredErrForKind(schemeName, gvk),
			expectedErrStr: `no kind "TestType" is registered for version "test/v1" in scheme "testScheme"`,
		},
		{
			name:           "with gvk and target along with a non-empty genericContext",
			err:            NewNotRegisteredGVKErrForTargetWithContext("in list item 3", gvk, target), // example of a generic context.
			expectedErrStr: `"test/v1" is not suitable for converting to "test/v2" in list item 3`,
		},
		{
			name:           "with gvk, target along with an empty genericContext",
			err:            NewNotRegisteredGVKErrForTargetWithContext("", gvk, target),
			expectedErrStr: `"test/v1" is not suitable for converting to "test/v2"`,
		},
	}

	for _, testCase := range tests {
		if testCase.err.Error() != testCase.expectedErrStr {
			t.Errorf("test [%s]: unexpected err string, got: %s, expected: %s", testCase.name, testCase.err.Error(), testCase.expectedErrStr)
		}
	}
}
