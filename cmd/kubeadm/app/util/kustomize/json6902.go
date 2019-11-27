/*
Copyright 2019 The Kubernetes Authors.

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

// Package kustomize contains helpers for working with embedded kustomize commands
package kustomize

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/patch"
)

// json6902 represents a json6902 patch
type json6902 struct {
	// Target refers to a Kubernetes object that the json patch will be applied to
	*patch.Target

	// Patch contain the json patch as a string
	Patch string
}

// json6902Slice is a slice of json6902 patches.
type json6902Slice []*json6902

// newJSON6902FromFile returns a json6902 patch from a file
func newJSON6902FromFile(f patch.Json6902, ldr ifc.Loader, file string) (*json6902, error) {
	patch, err := ldr.Load(file)
	if err != nil {
		return nil, err
	}

	return &json6902{
		Target: f.Target,
		Patch:  string(patch),
	}, nil
}

// filterByResource returns all the json6902 patches in the json6902Slice corresponding to a given resource
func (s *json6902Slice) filterByResource(r *unstructured.Unstructured) json6902Slice {
	var result json6902Slice
	for _, p := range *s {
		if p.Group == r.GroupVersionKind().Group &&
			p.Version == r.GroupVersionKind().Version &&
			p.Kind == r.GroupVersionKind().Kind &&
			p.Namespace == r.GetNamespace() &&
			p.Name == r.GetName() {
			result = append(result, p)
		}
	}
	return result
}
