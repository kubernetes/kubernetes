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

package crd

import (
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	csiapiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	"reflect"
)

// NOTE: the CRD functions here and the associated unit tests are non-ideal temporary measures in
// release 1.12 in order to aid manual CRD installation. This installation will be automated in
// subsequent releases and as a result this package will be removed.

// CSIDriverCRD returns the CustomResourceDefinition for CSIDriver object.
func CSIDriverCRD() *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: csiapiv1alpha1.CsiDriverResourcePlural + "." + csiapiv1alpha1.GroupName,
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   csiapiv1alpha1.GroupName,
			Version: csiapiv1alpha1.SchemeGroupVersion.Version,
			Scope:   apiextensionsv1beta1.ClusterScoped,
			Validation: &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
					Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
						"spec": {
							Description: "Specification of the CSI Driver.",
							Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
								"attachRequired": {
									Description: "Indicates this CSI volume driver requires an attach operation," +
										" and that Kubernetes should call attach and wait for any attach operation to" +
										" complete before proceeding to mount.",
									Type: "boolean",
								},
								"podInfoOnMountVersion": {
									Description: "Indicates this CSI volume driver requires additional pod" +
										" information (like podName, podUID, etc.) during mount operations.",
									Type: "string",
								},
							},
						},
					},
				},
			},
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: csiapiv1alpha1.CsiDriverResourcePlural,
				Kind:   reflect.TypeOf(csiapiv1alpha1.CSIDriver{}).Name(),
			},
		},
	}
}

// CSINodeInfoCRD returns the CustomResourceDefinition for CSINodeInfo object.
func CSINodeInfoCRD() *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: csiapiv1alpha1.CsiNodeInfoResourcePlural + "." + csiapiv1alpha1.GroupName,
		},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   csiapiv1alpha1.GroupName,
			Version: csiapiv1alpha1.SchemeGroupVersion.Version,
			Scope:   apiextensionsv1beta1.ClusterScoped,
			Validation: &apiextensionsv1beta1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
					Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
						"csiDrivers": {
							Description: "List of CSI drivers running on the node and their properties.",
							Type:        "array",
							Items: &apiextensionsv1beta1.JSONSchemaPropsOrArray{
								Schema: &apiextensionsv1beta1.JSONSchemaProps{
									Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
										"driver": {
											Description: "The CSI driver that this object refers to.",
											Type:        "string",
										},
										"nodeID": {
											Description: "The node from the driver point of view.",
											Type:        "string",
										},
										"topologyKeys": {
											Description: "List of keys supported by the driver.",
											Type:        "array",
											Items: &apiextensionsv1beta1.JSONSchemaPropsOrArray{
												Schema: &apiextensionsv1beta1.JSONSchemaProps{
													Type: "string",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural: csiapiv1alpha1.CsiNodeInfoResourcePlural,
				Kind:   reflect.TypeOf(csiapiv1alpha1.CSINodeInfo{}).Name(),
			},
		},
	}
}
