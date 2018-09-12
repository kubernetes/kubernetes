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

// Package attachdetachcrd implements logic to generate CRDs required by the
// attach/detach controller.
package attachdetachcrd

import (
	"reflect"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	csiapiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	"k8s.io/kubernetes/pkg/controller/crdinstaller/crdgenerator"
	"k8s.io/kubernetes/pkg/features"
)

// NewAttachDetachControllerCRDGenerator returns a new instance of
// ControllerCRDGenerator.
func NewAttachDetachControllerCRDGenerator() crdgenerator.ControllerCRDGenerator {
	return &attachDetachControllerCRDGenerator{}
}

var _ crdgenerator.ControllerCRDGenerator = (*attachDetachControllerCRDGenerator)(nil)

type attachDetachControllerCRDGenerator struct {
}

// GetCRDs returns the CRDs required by the attach/detach controller.
func (adcCRDGen *attachDetachControllerCRDGenerator) GetCRDs() []*apiextensionsv1beta1.CustomResourceDefinition {
	var attachDetachCRDs []*apiextensionsv1beta1.CustomResourceDefinition

	// Install required CSI CRDs on API server
	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		attachDetachCRDs = append(attachDetachCRDs, adcCRDGen.createCSIDriverCRD())
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) {
		attachDetachCRDs = append(attachDetachCRDs, adcCRDGen.createCSINodeInfo())
	}

	return attachDetachCRDs
}

// createCSIDriverCRD generates a CRD for the Container Storage Interface (CSI) CSIDriver
// object.
func (adcCRDGen *attachDetachControllerCRDGenerator) createCSIDriverCRD() *apiextensionsv1beta1.CustomResourceDefinition {
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

// createCSINodeInfo generates a CRD holding information about all CSI drivers installed on a node.
func (adcCRDGen *attachDetachControllerCRDGenerator) createCSINodeInfo() *apiextensionsv1beta1.CustomResourceDefinition {
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
