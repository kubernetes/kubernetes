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

package openapi_test

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubectl/pkg/util/openapi"
)

func TestSupportsDryRun(t *testing.T) {
	doc, err := fakeSchema.OpenAPISchema()
	if err != nil {
		t.Fatalf("Failed to get OpenAPI Schema: %v", err)
	}

	tests := []struct {
		gvk      schema.GroupVersionKind
		success  bool
		supports bool
	}{
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "Pod",
			},
			success:  true,
			supports: true,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "UnknownKind",
			},
			success:  false,
			supports: false,
		},
		{
			gvk: schema.GroupVersionKind{
				Group:   "",
				Version: "v1",
				Kind:    "NodeProxyOptions",
			},
			success:  true,
			supports: false,
		},
	}

	for _, test := range tests {
		supports, err := openapi.SupportsDryRun(doc, test.gvk)
		if supports != test.supports || ((err == nil) != test.success) {
			errStr := "nil"
			if test.success == false {
				errStr = "err"
			}
			t.Errorf("SupportsDryRun(doc, %v) = (%v, %v), expected (%v, %v)",
				test.gvk,
				supports, err,
				test.supports, errStr,
			)
		}
	}
}
