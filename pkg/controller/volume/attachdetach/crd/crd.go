package crd

import (
	"reflect"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	csiapiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
)

func CSIDriver() *apiextensionsv1beta1.CustomResourceDefinition {
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

func CSINodeInfo() *apiextensionsv1beta1.CustomResourceDefinition {
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
