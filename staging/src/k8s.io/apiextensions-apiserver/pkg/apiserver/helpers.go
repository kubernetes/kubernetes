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

package apiserver

import (
	"fmt"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var swaggerMetadataDescriptions = metav1.ObjectMeta{}.SwaggerDoc()

// getColumnsForVersion returns the columns for given version or nil.
// NOTE: the newly logically-defaulted columns is not pointing to the original CRD object.
// One cannot mutate the original CRD columns using the logically-defaulted columns. Please iterate through
// the original CRD object instead.
func getColumnsForVersion(crd *apiextensionsv1.CustomResourceDefinition, version string) ([]apiextensionsv1.CustomResourceColumnDefinition, error) {
	for _, v := range crd.Spec.Versions {
		if version == v.Name {
			return serveDefaultColumnsIfEmpty(v.AdditionalPrinterColumns), nil
		}
	}
	return nil, fmt.Errorf("version %s not found in apiextensionsv1.CustomResourceDefinition: %v", version, crd.Name)
}

// getScaleColumnsForVersion returns 2 columns for the desired and actual number of replicas.
func getScaleColumnsForVersion(crd *apiextensionsv1.CustomResourceDefinition, version string) ([]apiextensionsv1.CustomResourceColumnDefinition, error) {
	for _, v := range crd.Spec.Versions {
		if version != v.Name {
			continue
		}
		var cols []apiextensionsv1.CustomResourceColumnDefinition
		if v.Subresources != nil && v.Subresources.Scale != nil {
			if v.Subresources.Scale.SpecReplicasPath != "" {
				cols = append(cols, apiextensionsv1.CustomResourceColumnDefinition{
					Name:        "Desired",
					Type:        "integer",
					Description: "Number of desired replicas",
					JSONPath:    ".spec.replicas",
				})
			}
			if v.Subresources.Scale.StatusReplicasPath != "" {
				cols = append(cols, apiextensionsv1.CustomResourceColumnDefinition{
					Name:        "Available",
					Type:        "integer",
					Description: "Number of actual replicas",
					JSONPath:    ".status.replicas",
				})
			}
		}
		cols = append(cols, apiextensionsv1.CustomResourceColumnDefinition{
			Name:        "Age",
			Type:        "date",
			Description: swaggerMetadataDescriptions["creationTimestamp"],
			JSONPath:    ".metadata.creationTimestamp",
		})
		return cols, nil
	}
	return nil, fmt.Errorf("version %s not found in apiextensionsv1.CustomResourceDefinition: %v", version, crd.Name)
}

// serveDefaultColumnsIfEmpty applies logically defaulting to columns, if the input columns is empty.
// NOTE: in this way, the newly logically-defaulted columns is not pointing to the original CRD object.
// One cannot mutate the original CRD columns using the logically-defaulted columns. Please iterate through
// the original CRD object instead.
func serveDefaultColumnsIfEmpty(columns []apiextensionsv1.CustomResourceColumnDefinition) []apiextensionsv1.CustomResourceColumnDefinition {
	if len(columns) > 0 {
		return columns
	}
	return []apiextensionsv1.CustomResourceColumnDefinition{
		{Name: "Age", Type: "date", Description: swaggerMetadataDescriptions["creationTimestamp"], JSONPath: ".metadata.creationTimestamp"},
	}
}
