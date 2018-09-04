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

// getCRDSubresourcesForVersion returns the subresources for given version in given CRD Spec.
func getCRDSubresourcesForVersion(crdSpec *apiextensions.CustomResourceDefinitionSpec, version string) (*apiextensions.CustomResourceSubresources, error) {
	for _, v := range crdSpec.Versions {
		if version != v.Name {
			continue
		}
		if v.Subresources != nil {
			return v.Subresources, nil
		}
		return crdSpec.Subresources, nil
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinitionSpec", version)
}

// getCRDColumnsForVersion returns the columns for given version in given CRD Spec.
func getCRDColumnsForVersion(crdSpec *apiextensions.CustomResourceDefinitionSpec, version string) ([]apiextensions.CustomResourceColumnDefinition, error) {
	for _, v := range crdSpec.Versions {
		if version != v.Name {
			continue
		}
		if v.AdditionalPrinterColumns != nil {
			return v.AdditionalPrinterColumns, nil
		}
		return crdSpec.AdditionalPrinterColumns, nil
	}
	return nil, fmt.Errorf("version %s not found in CustomResourceDefinitionSpec", version)
}
