package aggregator

import (
	"fmt"
	"strings"

	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// mergeSpecsV3 to prevent a dependency on apiextensions-apiserver, this function is copied from https://github.com/kubernetes/kubernetes/blob/2c6c4566eff972d6c1320b5f8ad795f88c822d09/staging/src/k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder/merge.go#L105
// mergeSpecsV3 merges OpenAPI v3 specs for CRDs
// Conflicts belonging to the meta.v1 or autoscaling.v1 group versions are skipped as all CRDs reference those types
// Other conflicts will result in an error
func mergeSpecsV3(crdSpecs ...*spec3.OpenAPI) (*spec3.OpenAPI, error) {
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

const metadataGV = "io.k8s.apimachinery.pkg.apis.meta.v1"
const autoscalingGV = "io.k8s.api.autoscaling.v1"

// mergeSpecV3 to prevent a dependency on apiextensions-apiserver, this function is copied from https://github.com/kubernetes/kubernetes/blob/2c6c4566eff972d6c1320b5f8ad795f88c822d09/staging/src/k8s.io/apiextensions-apiserver/pkg/controller/openapi/builder/merge.go#L123
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
