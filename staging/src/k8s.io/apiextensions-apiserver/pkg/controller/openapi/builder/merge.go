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

package builder

import (
	"fmt"
	"strings"

	"k8s.io/kube-openapi/pkg/aggregator"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const metadataGV = "io.k8s.apimachinery.pkg.apis.meta.v1"
const autoscalingGV = "io.k8s.api.autoscaling.v1"

// MergeSpecs aggregates all OpenAPI specs, reusing the metadata of the first, static spec as the basis.
// The static spec has the highest priority, and its paths and definitions won't get overlapped by
// user-defined CRDs. None of the input is mutated, but input and output share data structures.
func MergeSpecs(staticSpec *spec.Swagger, crdSpecs ...*spec.Swagger) (*spec.Swagger, error) {
	// create shallow copy of staticSpec, but replace paths and definitions because we modify them.
	specToReturn := *staticSpec
	if staticSpec.Definitions != nil {
		specToReturn.Definitions = make(spec.Definitions, len(staticSpec.Definitions))
		for k, s := range staticSpec.Definitions {
			specToReturn.Definitions[k] = s
		}
	}
	if staticSpec.Parameters != nil {
		specToReturn.Parameters = make(map[string]spec.Parameter, len(staticSpec.Parameters))
		for k, s := range staticSpec.Parameters {
			specToReturn.Parameters[k] = s
		}
	}
	if staticSpec.Paths != nil {
		specToReturn.Paths = &spec.Paths{
			Paths: make(map[string]spec.PathItem, len(staticSpec.Paths.Paths)),
		}
		for k, p := range staticSpec.Paths.Paths {
			specToReturn.Paths.Paths[k] = p
		}
	}

	crdSpec := &spec.Swagger{}
	for _, s := range crdSpecs {
		// merge specs without checking conflicts, since the naming controller prevents
		// conflicts between user-defined CRDs
		mergeSpec(crdSpec, s)
	}

	// The static spec has the highest priority. Resolve conflicts to prevent user-defined
	// CRDs potentially overlapping the built-in apiextensions API
	if err := aggregator.MergeSpecsIgnorePathConflictRenamingDefinitionsAndParameters(&specToReturn, crdSpec); err != nil {
		return nil, err
	}
	return &specToReturn, nil
}

// mergeSpec copies paths, parameters and definitions from source to dest, mutating dest, but not source.
// We assume that conflicts do not matter.
func mergeSpec(dest, source *spec.Swagger) {
	if source == nil || source.Paths == nil {
		return
	}
	if dest.Paths == nil {
		dest.Paths = &spec.Paths{}
	}
	for k, v := range source.Definitions {
		if dest.Definitions == nil {
			dest.Definitions = make(spec.Definitions, len(source.Definitions))
		}
		dest.Definitions[k] = v
	}
	for k, v := range source.Parameters {
		if dest.Parameters == nil {
			dest.Parameters = make(map[string]spec.Parameter, len(source.Parameters))
		}
		dest.Parameters[k] = v
	}
	for k, v := range source.Paths.Paths {
		if dest.Paths.Paths == nil {
			dest.Paths.Paths = make(map[string]spec.PathItem, len(source.Paths.Paths))
		}
		dest.Paths.Paths[k] = v
	}
}

// MergeSpecsV3 merges OpenAPI v3 specs for CRDs
// Conflicts belonging to the meta.v1 or autoscaling.v1 group versions are skipped as all CRDs reference those types
// Other conflicts will result in an error
func MergeSpecsV3(crdSpecs ...*spec3.OpenAPI) (*spec3.OpenAPI, error) {
	crdSpec := &spec3.OpenAPI{}
	if len(crdSpecs) > 0 {
		crdSpec.Version = crdSpecs[0].Version
		crdSpec.Info = crdSpecs[0].Info
	}
	for _, s := range crdSpecs {
		err := mergeSpecV3(crdSpec, s)
		if err != nil {
			return nil, err
		}
	}
	return crdSpec, nil
}

// mergeSpecV3 copies paths and definitions from source to dest, mutating dest, but not source.
// Conflicts belonging to the meta.v1 or autoscaling.v1 group versions are skipped as all CRDs reference those types
// Other conflicts will result in an error
func mergeSpecV3(dest, source *spec3.OpenAPI) error {
	if source == nil || source.Paths == nil {
		return nil
	}
	if dest.Paths == nil {
		dest.Paths = &spec3.Paths{}
	}

	for k, v := range source.Components.Schemas {
		if dest.Components == nil {
			dest.Components = &spec3.Components{}
		}
		if dest.Components.Schemas == nil {
			dest.Components.Schemas = map[string]*spec.Schema{}
		}
		if _, exists := dest.Components.Schemas[k]; exists {
			if strings.HasPrefix(k, metadataGV) || strings.HasPrefix(k, autoscalingGV) {
				continue
			}
			return fmt.Errorf("OpenAPI V3 merge schema conflict on %s", k)
		}
		dest.Components.Schemas[k] = v
	}
	for k, v := range source.Paths.Paths {
		if dest.Paths.Paths == nil {
			dest.Paths.Paths = map[string]*spec3.Path{}
		}
		dest.Paths.Paths[k] = v
	}
	return nil
}
