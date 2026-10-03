/*
Copyright The Kubernetes Authors.

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

package pod

import (
	"context"
	"fmt"
	"maps"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
)

// RunDeclarativeValidateRuntimeOptionsTestCases checks the shared bounds on
// checkpoint and restore options, including resources that embed a pod template.
func RunDeclarativeValidateRuntimeOptionsTestCases[T runtime.Object](t *testing.T, ctx context.Context, strategy rest.RESTCreateStrategy, fldPath *field.Path, baseObj T, setOptions func(T, map[string]string)) {
	t.Helper()
	maxEntries := make(map[string]string, 64)
	for i := range 64 {
		maxEntries[fmt.Sprintf("option-%d", i)] = "value"
	}
	tooManyEntries := maps.Clone(maxEntries)
	tooManyEntries["extra"] = "value"

	testCases := map[string]struct {
		options      map[string]string
		expectedErrs field.ErrorList
	}{
		"unset": {},
		"empty": {
			options: map[string]string{},
		},
		"empty value": {
			options: map[string]string{"example.runtime/flag": ""},
		},
		"maximum entries": {
			options: maxEntries,
		},
		"too many entries": {
			options:      tooManyEntries,
			expectedErrs: field.ErrorList{field.TooMany(fldPath, 65, 64).WithOrigin("maxProperties")},
		},
		"maximum key length": {
			options: map[string]string{strings.Repeat("k", 256): "value"},
		},
		"key too long": {
			options:      map[string]string{strings.Repeat("k", 257): "value"},
			expectedErrs: field.ErrorList{field.TooLong(fldPath, "", 256).WithOrigin("maxBytes")},
		},
		"maximum value length": {
			options: map[string]string{"key": strings.Repeat("v", 4096)},
		},
		"value too long": {
			options:      map[string]string{"key": strings.Repeat("v", 4097)},
			expectedErrs: field.ErrorList{field.TooLong(fldPath.Key("key"), "", 4096).WithOrigin("maxBytes")},
		},
		"maximum multibyte key and value lengths": {
			options: map[string]string{strings.Repeat("é", 128): strings.Repeat("é", 2048)},
		},
		"key byte length exceeds limit": {
			options:      map[string]string{strings.Repeat("é", 128) + "k": "value"},
			expectedErrs: field.ErrorList{field.TooLong(fldPath, "", 256).WithOrigin("maxBytes")},
		},
		"value byte length exceeds limit": {
			options:      map[string]string{"key": strings.Repeat("é", 2048) + "v"},
			expectedErrs: field.ErrorList{field.TooLong(fldPath.Key("key"), "", 4096).WithOrigin("maxBytes")},
		},
	}
	for name, tc := range testCases {
		t.Run(fldPath.String()+": "+name, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			setOptions(obj, tc.options)
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.expectedErrs)
		})
	}
}
