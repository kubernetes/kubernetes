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

package openapitest_test

import (
	"testing"

	"k8s.io/client-go/openapi/openapitest"
	"k8s.io/kube-openapi/pkg/spec3"
	kjson "sigs.k8s.io/json"
)

func TestOpenAPIEmbeddedTest(t *testing.T) {
	client := openapitest.NewEmbeddedFileClient()

	// make sure we get paths
	paths, err := client.Paths()
	if err != nil {
		t.Fatalf("error fetching paths: %v", err)
	}
	if len(paths) == 0 {
		t.Error("empty paths")
	}

	// spot check specific paths
	expectedPaths := []string{
		"api/v1",
		"apis/apps/v1",
		"apis/batch/v1",
		"apis/networking.k8s.io/v1alpha1",
		"apis/discovery.k8s.io/v1",
	}
	for _, p := range expectedPaths {
		if _, ok := paths[p]; !ok {
			t.Fatalf("expected %s", p)
		}
	}

	// make sure all paths can load
	for path, gv := range paths {
		data, err := gv.Schema("application/json")
		if err != nil {
			t.Fatalf("error reading schema for %v: %v", path, err)
		}
		o := &spec3.OpenAPI{}
		stricterrs, err := kjson.UnmarshalStrict(data, o)
		if err != nil {
			t.Fatalf("error unmarshaling schema for %v: %v", path, err)
		}
		if len(stricterrs) > 0 {
			t.Fatalf("strict errors unmarshaling schema for %v: %v", path, stricterrs)
		}
	}
}

func TestOpenAPITest(t *testing.T) {
	client := openapitest.NewFileClient("testdata")

	// make sure we get paths
	paths, err := client.Paths()
	if err != nil {
		t.Fatalf("error fetching paths: %v", err)
	}
	if len(paths) == 0 {
		t.Error("empty paths")
	}

	// spot check specific paths
	expectedPaths := []string{
		"api/v1",
		"apis/apps/v1",
		"apis/batch/v1",
		"apis/networking.k8s.io/v1alpha1",
		"apis/discovery.k8s.io/v1",
	}
	for _, p := range expectedPaths {
		if _, ok := paths[p]; !ok {
			t.Fatalf("expected %s", p)
		}
	}

	// make sure all paths can load
	for path, gv := range paths {
		data, err := gv.Schema("application/json")
		if err != nil {
			t.Fatalf("error reading schema for %v: %v", path, err)
		}
		o := &spec3.OpenAPI{}
		stricterrs, err := kjson.UnmarshalStrict(data, o)
		if err != nil {
			t.Fatalf("error unmarshaling schema for %v: %v", path, err)
		}
		if len(stricterrs) > 0 {
			t.Fatalf("strict errors unmarshaling schema for %v: %v", path, stricterrs)
		}
	}
}
