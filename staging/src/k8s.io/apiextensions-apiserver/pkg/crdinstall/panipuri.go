package crdinstall

import (
	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var panipuriCRD = &v1.CustomResourceDefinition{
	TypeMeta: metav1.TypeMeta{
		Kind:       "CustomResourceDefinition",
		APIVersion: v1.SchemeGroupVersion.String(),
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "panipuris.chaat.com",
	},
	Spec: v1.CustomResourceDefinitionSpec{
		Group: "chaat.com",
		Names: v1.CustomResourceDefinitionNames{
			Plural:     "panipuris",
			Singular:   "panipuri",
			ShortNames: []string{"pp"},
			Kind:       "PaniPuri",
		},
		Scope: "Namespaced",
		Versions: []v1.CustomResourceDefinitionVersion{
			panipuriCRDv1,
		},
		PreserveUnknownFields: false,
	},
	Status: v1.CustomResourceDefinitionStatus{},
}

var panipuriCRDv1 = v1.CustomResourceDefinitionVersion{
	Name:    "v1",
	Served:  true,
	Storage: true,
	Schema: &v1.CustomResourceValidation{
		OpenAPIV3Schema: &v1.JSONSchemaProps{
			Type: "object",
			Properties: map[string]v1.JSONSchemaProps{
				"spec":   panipuriCRDSv1pec,
				"status": panipuriCRDv1Status,
			},
		},
	},
	Subresources:             panipuriCRDv1Subresources,
	AdditionalPrinterColumns: panipuriCRDv1PrinterAdditionalColumns,
}

var panipuriCRDSv1pec = v1.JSONSchemaProps{
	Type: "object",
	Properties: map[string]v1.JSONSchemaProps{
		"includeAloo": {
			Type: "boolean",
		},
		"perPlate": {
			Type: "integer",
		},
	},
}

var panipuriCRDv1Status = v1.JSONSchemaProps{
	Type: "object",
	Properties: map[string]v1.JSONSchemaProps{
		"served": {
			Type: "integer",
		},
		"labelSelector": {
			Type: "string",
		},
	},
}

var panipuriCRDv1Subresources = &v1.CustomResourceSubresources{
	Status: &v1.CustomResourceSubresourceStatus{},
	Scale: &v1.CustomResourceSubresourceScale{
		SpecReplicasPath:   ".spec.perPlate",
		StatusReplicasPath: ".status.served",
	},
}

var panipuriCRDv1PrinterAdditionalColumns = []v1.CustomResourceColumnDefinition{
	{
		Name:        "Served",
		Type:        "integer",
		Description: "The number of puris already served",
		JSONPath:    ".status.served",
	},
	{
		Name:        "PerPlate",
		Type:        "integer",
		Description: "Maximum panipuris per plate",
		JSONPath:    ".spec.perPlate",
	},
}
