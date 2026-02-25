/*
Copyright 2025 The Kubernetes Authors.

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

package api

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
)

func TestOpenAPIDefinitionNames(t *testing.T) {
	scheme := runtime.NewScheme()
	for _, builder := range groups {
		if err := builder.AddToScheme(scheme); err != nil {
			t.Fatalf("unexpected error adding to scheme: %v", err)
		}
	}

	kinds := scheme.AllKnownTypes()
	for gvk := range kinds {
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		t.Run(gvk.String(), func(t *testing.T) {
			example, err := scheme.New(gvk)
			if err != nil {
				t.Fatalf("unexpected error creating example: %v", err)
			}

			namer, ok := example.(OpenAPIModelNamer)
			if !ok {
				t.Fatalf("type %v does not implement OpenAPICanonicalTypeName\n", gvk)
			}
			lookupName := namer.OpenAPIModelName()

			rtype := reflect.TypeOf(example).Elem()
			reflectName := ToRESTFriendlyName(rtype.PkgPath() + "." + rtype.Name())

			if lookupName != reflectName {
				t.Errorf("expected %v, got %v", reflectName, lookupName)
			}
		})
	}
}

type OpenAPIModelNamer interface {
	OpenAPIModelName() string
}

func ToRESTFriendlyName(name string) string {
	nameParts := strings.Split(name, "/")
	// Reverse first part. e.g., io.k8s... instead of k8s.io...
	if len(nameParts) > 0 && strings.Contains(nameParts[0], ".") {
		parts := strings.Split(nameParts[0], ".")
		for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
			parts[i], parts[j] = parts[j], parts[i]
		}
		nameParts[0] = strings.Join(parts, ".")
	}
	return strings.Join(nameParts, ".")
}
