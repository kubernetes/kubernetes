/*
Copyright 2015 The Kubernetes Authors.

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

package v1

import (
	"testing"

	"k8s.io/apimachinery/pkg/internal/encodingtest"
)

func groupVersionUnmarshalJSONCases() []encodingtest.UnmarshalCase[GroupVersion] {
	//nolint:govet // Unkeyed fields are fine for test cases.
	return []encodingtest.UnmarshalCase[GroupVersion]{
		{`"v1"`, GroupVersion{"", "v1"}},
		{`"apps/v1"`, GroupVersion{"apps", "v1"}},
		{`""`, GroupVersion{}},
		{`null`, GroupVersion{}},
		{`"apps/"`, GroupVersion{"apps", ""}},
		{`"apps\/v1"`, GroupVersion{"apps", "v1"}},
	}
}

func TestGroupVersionUnmarshalJSON(t *testing.T) {
	encodingtest.UnmarshalJSON(t, groupVersionUnmarshalJSONCases())
	encodingtest.UnmarshalJSONErrors[GroupVersion](t, []string{
		`"a/b/c"`,
		`5`,
		`true`,
		`{}`,
		`[]`,
	})

}

func BenchmarkGroupVersionUnmarshalJSON(b *testing.B) {
	encodingtest.BenchmarkUnmarshalJSON(b, groupVersionUnmarshalJSONCases())
}

func groupVersionMarshalJSONCases() []encodingtest.MarshalCase[GroupVersion] {
	//nolint:govet // Unkeyed fields are fine for test cases.
	return []encodingtest.MarshalCase[GroupVersion]{
		{GroupVersion{"", "v1"}, `"v1"`},
		{GroupVersion{"apps", "v1"}, `"apps/v1"`},
		{GroupVersion{}, `""`},
		{GroupVersion{"apps", ""}, `"apps/"`},
	}
}

func TestGroupVersionMarshalJSON(t *testing.T) {
	encodingtest.MarshalJSON(t, groupVersionMarshalJSONCases())
	encodingtest.MarshalJSONErrors(t, []GroupVersion{
		{"a/b", "v1"},
		{"apps", "v1/beta"},
	})
}

func BenchmarkGroupVersionMarshalJSON(b *testing.B) {
	encodingtest.BenchmarkMarshalJSON(b, groupVersionMarshalJSONCases())
}
