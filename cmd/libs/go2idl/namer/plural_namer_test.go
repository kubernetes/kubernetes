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

package namer

import (
	"testing"

	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

func TestPluralNamer(t *testing.T) {
	exceptions := map[string]string{
		// The type name is already in the plural form
		"Endpoints": "endpoints",
	}
	public := NewPublicPluralNamer(exceptions)
	private := NewPrivatePluralNamer(exceptions)

	cases := []struct {
		typeName        string
		expectedPrivate string
		expectedPublic  string
	}{
		{
			"Pod",
			"pods",
			"Pods",
		},
		{
			"Entry",
			"entries",
			"Entries",
		},
		{
			"Endpoints",
			"endpoints",
			"Endpoints",
		},
		{
			"Bus",
			"buses",
			"Buses",
		},
	}
	for _, c := range cases {
		testType := &types.Type{Name: types.Name{Name: c.typeName}}
		if e, a := c.expectedPrivate, private.Name(testType); e != a {
			t.Errorf("Unexpected result from private plural namer. Expected: %s, Got: %s", e, a)
		}
		if e, a := c.expectedPublic, public.Name(testType); e != a {
			t.Errorf("Unexpected result from public plural namer. Expected: %s, Got: %s", e, a)
		}
	}
}
