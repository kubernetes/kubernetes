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

package apiserver

import (
	"fmt"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
)

// getCRDSchemaForVersion returns the validation schema for given version in given CRD Spec.
func getCRDSchemaForVersion(crdSpec *apiextensions.CustomResourceDefinitionSpec, version string) (*apiextensions.CustomResourceValidation, error) {
	for _, v := range crdSpec.Versions {
		if version != v.Name {
			continue
		}
		if v.Schema != nil {
			// For backwards compatibility with existing code path, we wrap the
			// OpenAPIV3Schema into a CustomResourceValidation struct
			return &apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: v.Schema,
			}, nil
		}
		return crdSpec.Validation, nil
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinitionSpec", version)
}
