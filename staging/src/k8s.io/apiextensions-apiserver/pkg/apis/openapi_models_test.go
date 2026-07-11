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
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kube-openapi/pkg/util"
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

			namer, ok := example.(util.OpenAPIModelNamer)
			if !ok {
				t.Fatalf("type %v does not implement OpenAPICanonicalTypeName\n", gvk)
			}
			lookupName := namer.OpenAPIModelName()

			rtype := reflect.TypeOf(example).Elem()
			reflectName := util.ToRESTFriendlyName(rtype.PkgPath() + "." + rtype.Name())

			if lookupName != reflectName {
				t.Errorf("expected %v, got %v", reflectName, lookupName)
			}
		})
	}
}
