/*
Copyright 2022 The Kubernetes Authors.

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

package v2_test

import (
	"bytes"
	"testing"
	"text/template"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
	v2 "k8s.io/kubectl/pkg/explain/v2"
)

func TestFuncs(t *testing.T) {
	testcases := []struct {
		Name     string
		FuncName string
		Source   string
		Context  any
		Expect   string
		Error    string
	}{
		{
			Name:     "err",
			FuncName: "fail",
			Source:   `{{fail .}}`,
			Context:  "this is a test",
			Error:    "this is a test",
		},
		{
			Name:     "basic",
			FuncName: "wrap",
			Source:   `{{wrap 3 .}}`,
			Context:  "this is a really good test",
			Expect:   "this\nis\na\nreally\ngood\ntest",
		},
		{
			Name:     "basic",
			FuncName: "split",
			Source:   `{{split . "/"}}`,
			Context:  "this/is/a/slash/separated/thing",
			Expect:   "[this is a slash separated thing]",
		},
		{
			Name:     "basic",
			FuncName: "join",
			Source:   `{{join "/" "this" "is" "a" "slash" "separated" "thing"}}`,
			Expect:   "this/is/a/slash/separated/thing",
		},
		{
			Name:     "basic",
			FuncName: "include",
			Source:   `{{define "myTemplate"}}{{.}}{{end}}{{$var := include "myTemplate" .}}{{$var}}`,
			Context:  "hello, world!",
			Expect:   "hello, world!",
		},
		{
			Name:     "nil",
			FuncName: "first",
			Source:   `{{first .}}`,
			Context:  nil,
			Error:    "list is empty",
		},
		{
			Name:     "empty",
			FuncName: "first",
			Source:   `{{first .}}`,
			Context:  []string{},
			Error:    "list is empty",
		},
		{
			Name:     "basic",
			FuncName: "first",
			Source:   `{{first .}}`,
			Context:  []string{"first", "second", "third"},
			Expect:   "first",
		},
		{
			Name:     "wrongtype",
			FuncName: "first",
			Source:   `{{first .}}`,
			Context:  "test",
			Error:    "first cannot be used on type: string",
		},
		{
			Name:     "nil",
			FuncName: "last",
			Source:   `{{last .}}`,
			Context:  nil,
			Error:    "list is empty",
		},
		{
			Name:     "empty",
			FuncName: "last",
			Source:   `{{last .}}`,
			Context:  []string{},
			Error:    "list is empty",
		},
		{
			Name:     "basic",
			FuncName: "last",
			Source:   `{{last .}}`,
			Context:  []string{"first", "second", "third"},
			Expect:   "third",
		},
		{
			Name:     "wrongtype",
			FuncName: "last",
			Source:   `{{last .}}`,
			Context:  "test",
			Error:    "last cannot be used on type: string",
		},
		{
			Name:     "none",
			FuncName: "indent",
			Source:   `{{indent 0 .}}`,
			Context:  "this is a string",
			Expect:   "this is a string",
		},
		{
			Name:     "some",
			FuncName: "indent",
			Source:   `{{indent 2 .}}`,
			Context:  "this is a string",
			Expect:   "  this is a string",
		},
		{
			Name:     "empty",
			FuncName: "dict",
			Source:   `{{dict | toJson}}`,
			Expect:   "{}",
		},
		{
			Name:     "single value",
			FuncName: "dict",
			Source:   `{{dict "key" "value" | toJson}}`,
			Expect:   `{"key":"value"}`,
		},
		{
			Name:     "twoValues",
			FuncName: "dict",
			Source:   `{{dict "key1" "val1" "key2" "val2" | toJson}}`,
			Expect:   `{"key1":"val1","key2":"val2"}`,
		},
		{
			Name:     "oddNumberArgs",
			FuncName: "dict",
			Source:   `{{dict "key1" 1 "key2" | toJson}}`,
			Error:    "error calling dict: expected even # of arguments",
		},
		{
			Name:     "IntegerValue",
			FuncName: "dict",
			Source:   `{{dict "key1" 1 | toJson}}`,
			Expect:   `{"key1":1}`,
		},
		{
			Name:     "MixedValues",
			FuncName: "dict",
			Source:   `{{dict "key1" 1 "key2" "val2" "key3" (dict "key1" "val1") | toJson}}`,
			Expect:   `{"key1":1,"key2":"val2","key3":{"key1":"val1"}}`,
		},
		{
			Name:     "nil",
			FuncName: "contains",
			Source:   `{{contains . "value"}}`,
			Context:  nil,
			Expect:   `false`,
		},
		{
			Name:     "empty",
			FuncName: "contains",
			Source:   `{{contains . "value"}}`,
			Context:  []string{},
			Expect:   `false`,
		},
		{
			Name:     "basic",
			FuncName: "contains",
			Source:   `{{contains . "value"}}`,
			Context:  []string{"value"},
			Expect:   `true`,
		},
		{
			Name:     "struct",
			FuncName: "contains",
			Source:   `{{contains $.haystack $.needle}}`,
			Context: map[string]any{
				"needle": schema.GroupVersionKind{Group: "testgroup.k8s.io", Version: "v1", Kind: "Kind"},
				"haystack": []schema.GroupVersionKind{
					{Group: "randomgroup.k8s.io", Version: "v1", Kind: "OtherKind"},
					{Group: "testgroup.k8s.io", Version: "v1", Kind: "OtherKind"},
					{Group: "testgroup.k8s.io", Version: "v1", Kind: "Kind"},
				},
			},
			Expect: `true`,
		},
		{
			Name:     "nil",
			FuncName: "set",
			Source:   `{{set nil "key" "value" | toJson}}`,
			Expect:   `{"key":"value"}`,
		},
		{
			Name:     "empty",
			FuncName: "set",
			Source:   `{{set (dict) "key" "value" | toJson}}`,
			Expect:   `{"key":"value"}`,
		},
		{
			Name:     "OddArgs",
			FuncName: "set",
			Source:   `{{set (dict) "key" "value" "key2" | toJson}}`,
			Error:    `expected even number of arguments`,
		},
		{
			Name:     "NonStringKey",
			FuncName: "set",
			Source:   `{{set (dict) 1 "value" | toJson}}`,
			Error:    `keys must be strings`,
		},
		{
			Name:     "NilKey",
			FuncName: "set",
			Source:   `{{set (dict) nil "value" | toJson}}`,
			Error:    `keys must be strings`,
		},
		{
			Name:     "NilValue",
			FuncName: "set",
			Source:   `{{set (dict) "key" nil | toJson}}`,
			Expect:   `{"key":null}`,
		},
		{
			Name:     "OverwriteKey",
			FuncName: "set",
			Source:   `{{set (dict "key1" "val1" "key2" "val2") "key1" nil | toJson}}`,
			Expect:   `{"key1":null,"key2":"val2"}`,
		},
		{
			Name:     "OverwriteKeyWithLefover",
			FuncName: "set",
			Source:   `{{set (dict "key1" "val1" "key2" "val2" "key3" "val3") "key1" nil | toJson}}`,
			Expect:   `{"key1":null,"key2":"val2","key3":"val3"}`,
		},
		{
			Name:     "basic",
			FuncName: "add",
			Source:   `{{add 1 2}}`,
			Expect:   `3`,
		},
		{
			Name:     "basic",
			FuncName: "sub",
			Source:   `{{sub 1 2}}`,
			Expect:   `-1`,
		},
		{
			Name:     "basic",
			FuncName: "mul",
			Source:   `{{mul 2 3}}`,
			Expect:   `6`,
		},
		{
			Name:     "basic",
			FuncName: "resolveRef",
			Source:   `{{resolveRef "#/components/schemas/myTypeName" . | toJson}}`,
			Context: map[string]any{
				"components": map[string]any{
					"schemas": map[string]any{
						"myTypeName": map[string]any{
							"key": "val",
						},
					},
				},
			},
			Expect: `{"key":"val"}`,
		},
		{
			Name:     "basicNameWithDots",
			FuncName: "resolveRef",
			Source:   `{{resolveRef "#/components/schemas/myTypeName.with.dots" . | toJson}}`,
			Context: map[string]any{
				"components": map[string]any{
					"schemas": map[string]any{
						"myTypeName.with.dots": map[string]any{
							"key": "val",
						},
					},
				},
			},
			Expect: `{"key":"val"}`,
		},
		{
			Name:     "notFound",
			FuncName: "resolveRef",
			Source:   `{{resolveRef "#/components/schemas/otherTypeName" . | toJson}}`,
			Context: map[string]any{
				"components": map[string]any{
					"schemas": map[string]any{
						"myTypeName": map[string]any{
							"key": "val",
						},
					},
				},
			},
			Expect: `null`,
		},
		{
			Name:     "url",
			FuncName: "resolveRef",
			Source:   `{{resolveRef "http://swagger.com/swagger.json#/components/schemas/myTypeName" . | toJson}}`,
			Context: map[string]any{
				"components": map[string]any{
					"schemas": map[string]any{
						"myTypeName": map[string]any{
							"key": "val",
						},
					},
				},
			},
			Expect: `null`,
		},
	}

	for _, tcase := range testcases {
		t.Run(tcase.FuncName+"/"+tcase.Name, func(t *testing.T) {

			tmpl, err := v2.WithBuiltinTemplateFuncs(template.New("me")).Parse(tcase.Source)
			require.NoError(t, err)

			buf := bytes.NewBuffer(nil)
			err = tmpl.Execute(buf, tcase.Context)

			if len(tcase.Error) > 0 {
				require.ErrorContains(t, err, tcase.Error)
			} else if output := buf.String(); len(tcase.Expect) > 0 {
				require.NoError(t, err)
				require.Contains(t, output, tcase.Expect)
			}
		})
	}
}
