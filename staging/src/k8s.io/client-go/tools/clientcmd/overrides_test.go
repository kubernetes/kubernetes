/*
Copyright 2017 The Kubernetes Authors.

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

package clientcmd

import (
	"testing"

	"github.com/spf13/pflag"
)

func TestNamespacePrefixStrip(t *testing.T) {
	testData := map[string]string{
		"namespaces/foo": "foo",
		"NAMESPACES/foo": "foo",
		"NameSpaces/foo": "foo",
		"namespace/foo":  "foo",
		"NAMESPACE/foo":  "foo",
		"nameSpace/foo":  "foo",
		"ns/foo":         "foo",
		"NS/foo":         "foo",
		"namespaces/":    "namespaces/",
		"namespace/":     "namespace/",
		"ns/":            "ns/",
	}

	for before, after := range testData {
		overrides := &ConfigOverrides{}
		fs := &pflag.FlagSet{}
		BindOverrideFlags(overrides, fs, RecommendedConfigOverrideFlags(""))
		fs.Parse([]string{"--namespace", before})

		if overrides.Context.Namespace != after {
			t.Fatalf("Expected %s, got %s", after, overrides.Context.Namespace)
		}
	}
}
