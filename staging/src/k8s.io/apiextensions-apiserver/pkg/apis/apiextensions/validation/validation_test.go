/*
Copyright 2017 The Kubernetes Authors.

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

package validation

import (
	"math/rand"
	"strings"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsfuzzer "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"
)

type validationMatch struct {
	path      *field.Path
	errorType field.ErrorType
}

func required(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeRequired}
}
func invalid(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeInvalid}
}
func invalidIndex(index int, path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...).Index(index), errorType: field.ErrorTypeInvalid}
}
func unsupported(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeNotSupported}
}
func immutable(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeInvalid}
}
func forbidden(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeForbidden}
}

func (v validationMatch) matches(err *field.Error) bool {
	return err.Type == v.errorType && err.Field == v.path.String()
}

func strPtr(s string) *string { return &s }

func TestValidateCustomResourceDefinition(t *testing.T) {
	singleVersionList := []apiextensions.CustomResourceDefinitionVersion{
		{
			Name:    "version",
			Served:  true,
			Storage: true,
		},
	}
	tests := []struct {
		name             string
		resource         *apiextensions.CustomResourceDefinition
		requestGV        schema.GroupVersion
		errors           []validationMatch
		enabledFeatures  []featuregate.Feature
		disabledFeatures []featuregate.Feature
	}{
		{
			name: "invalid types allowed via v1beta1",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{
						Name:    "version",
						Served:  true,
						Storage: true,
					}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "bogus"}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
		},
		{
			name: "invalid types disallowed via v1",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "bogus"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				unsupported("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "webhookconfig: invalid port 0",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							Service: &apiextensions.ServiceReference{
								Name:      "n",
								Namespace: "ns",
								Port:      0,
							},
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "conversion", "webhookClientConfig", "service", "port"),
			},
		},
		{
			name: "webhookconfig: invalid port 65536",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							Service: &apiextensions.ServiceReference{
								Name:      "n",
								Namespace: "ns",
								Port:      65536,
							},
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "conversion", "webhookClientConfig", "service", "port"),
			},
		},
		{
			name: "webhookconfig: both service and URL provided",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
							Service: &apiextensions.ServiceReference{
								Name:      "n",
								Namespace: "ns",
							},
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				required("spec", "conversion", "webhookClientConfig"),
			},
		},
		{
			name: "webhookconfig: blank URL",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr(""),
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "conversion", "webhookClientConfig", "url"),
				invalid("spec", "conversion", "webhookClientConfig", "url"),
			},
		},
		{
			name: "webhookconfig_should_not_be_set",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				forbidden("spec", "conversion", "webhookClientConfig"),
			},
		},
		{
			name: "ConversionReviewVersions_should_not_be_set",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy:                 apiextensions.ConversionStrategyType("None"),
						ConversionReviewVersions: []string{"v1beta1"},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				forbidden("spec", "conversion", "conversionReviewVersions"),
			},
		},
		{
			name: "webhookconfig: invalid ConversionReviewVersion",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"invalid-version"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "conversion", "conversionReviewVersions"),
			},
		},
		{
			name: "webhookconfig: invalid ConversionReviewVersion version string",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"0v"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalidIndex(0, "spec", "conversion", "conversionReviewVersions"),
				invalid("spec", "conversion", "conversionReviewVersions"),
			},
		},
		{
			name: "webhookconfig: at least one valid ConversionReviewVersion",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"invalid-version", "v1beta1"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "webhookconfig: duplicate ConversionReviewVersion",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"v1beta1", "v1beta1"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalidIndex(1, "spec", "conversion", "conversionReviewVersions"),
			},
		},
		{
			name: "missing_webhookconfig",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				required("spec", "conversion", "webhookClientConfig"),
			},
		},
		{
			name: "invalid_conversion_strategy",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("non_existing_conversion"),
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				unsupported("spec", "conversion", "strategy"),
			},
		},
		{
			name: "none conversion without preserveUnknownFields=false",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version1",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version1"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "webhook conversion without preserveUnknownFields=false",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version1",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"v1beta1"},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version1"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("spec", "conversion", "strategy"),
			},
		},
		{
			name: "webhook conversion with preserveUnknownFields=false, conversionReviewVersions=[v1beta1]",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version1",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"v1beta1"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version1"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "webhook conversion with preserveUnknownFields=false, conversionReviewVersions=[v1]",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version1",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"v1"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version1"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "no_storage_version",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: false,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("spec", "versions"),
			},
		},
		{
			name: "multiple_storage_version",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: true,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("spec", "versions"),
				invalid("status", "storedVersions"),
			},
		},
		{
			name: "missing_storage_version_in_stored_versions",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: false,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: true,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("status", "storedVersions"),
			},
		},
		{
			name: "empty_stored_version",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("status", "storedVersions"),
			},
		},
		{
			name: "mismatched name",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.not.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural: "plural",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("status", "storedVersions"),
				invalid("metadata", "name"),
				invalid("spec", "versions"),
				required("spec", "scope"),
				required("spec", "names", "singular"),
				required("spec", "names", "kind"),
				required("spec", "names", "listKind"),
			},
		},
		{
			name: "missing values",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
			},
			errors: []validationMatch{
				invalid("status", "storedVersions"),
				invalid("metadata", "name"),
				invalid("spec", "versions"),
				required("spec", "group"),
				required("spec", "scope"),
				required("spec", "names", "plural"),
				required("spec", "names", "singular"),
				required("spec", "names", "kind"),
				required("spec", "names", "listKind"),
			},
		},
		{
			name: "bad names 01",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group",
					Version: "ve()*rsion",
					Scope:   apiextensions.ResourceScope("foo"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "pl()*ural",
						Singular: "value()*a",
						Kind:     "value()*a",
						ListKind: "value()*a",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "pl()*ural",
						Singular: "value()*a",
						Kind:     "value()*a",
						ListKind: "value()*a",
					},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("status", "storedVersions"),
				invalid("metadata", "name"),
				invalid("spec", "group"),
				unsupported("spec", "scope"),
				invalid("spec", "names", "plural"),
				invalid("spec", "names", "singular"),
				invalid("spec", "names", "kind"),
				invalid("spec", "names", "listKind"), // invalid format
				invalid("spec", "names", "listKind"), // kind == listKind
				invalid("status", "acceptedNames", "plural"),
				invalid("status", "acceptedNames", "singular"),
				invalid("status", "acceptedNames", "kind"),
				invalid("status", "acceptedNames", "listKind"), // invalid format
				invalid("status", "acceptedNames", "listKind"), // kind == listKind
				invalid("spec", "versions"),
				invalid("spec", "version"),
			},
		},
		{
			name: "bad names 02",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.c(*&om",
					Version:  "version",
					Versions: singleVersionList,
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "matching",
						ListKind: "matching",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "matching",
						ListKind: "matching",
					},
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("metadata", "name"),
				invalid("spec", "group"),
				required("spec", "scope"),
				invalid("spec", "names", "listKind"),
				invalid("status", "acceptedNames", "listKind"),
			},
		},
		{
			name: "additionalProperties and properties forbidden",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Properties: map[string]apiextensions.JSONSchemaProps{
								"foo": {},
							},
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{Allows: false},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "additionalProperties"),
			},
		},
		{
			name: "additionalProperties without properties allowed (map[string]string)",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("None"),
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Allows: true,
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "per-version fields may not all be set to identical values (top-level field should be used instead)",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: validValidationSchema,
							},
							Subresources:             &apiextensions.CustomResourceSubresources{},
							AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "Alpha", Type: "string", JSONPath: ".spec.alpha"}},
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: validValidationSchema,
							},
							Subresources:             &apiextensions.CustomResourceSubresources{},
							AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "Alpha", Type: "string", JSONPath: ".spec.alpha"}},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				// Per-version schema/subresources/columns may not all be set to identical values.
				// Note that the test will fail if we de-duplicate the expected errors below.
				invalid("spec", "versions"),
				invalid("spec", "versions"),
				invalid("spec", "versions"),
			},
		},
		{
			name: "x-kubernetes-preserve-unknown-field: false",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							XPreserveUnknownFields: pointer.BoolPtr(false),
						},
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-preserve-unknown-fields"),
			},
		},
		{
			name: "preserveUnknownFields with unstructural global schema",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validUnstructuralValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "properties[spec]", "type"),
				required("spec", "validation", "openAPIV3Schema", "properties[status]", "type"),
				required("spec", "validation", "openAPIV3Schema", "items", "type"),
			},
		},
		{
			name: "preserveUnknownFields with unstructural schema in one version",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: validValidationSchema,
							},
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: validUnstructuralValidationSchema,
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				required("spec", "versions[1]", "schema", "openAPIV3Schema", "properties[spec]", "type"),
				required("spec", "versions[1]", "schema", "openAPIV3Schema", "properties[status]", "type"),
				required("spec", "versions[1]", "schema", "openAPIV3Schema", "items", "type"),
			},
		},
		{
			name: "preserveUnknownFields with no schema in one version",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: validValidationSchema,
							},
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: nil,
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "versions[1]", "schema", "openAPIV3Schema"),
			},
		},
		{
			name: "preserveUnknownFields with no schema at all",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
							Schema:  nil,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: nil,
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "versions[0]", "schema", "openAPIV3Schema"),
				required("spec", "versions[1]", "schema", "openAPIV3Schema"),
			},
		},
		{
			name: "no schema via v1",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
							Schema:  nil,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: nil,
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "versions[0]", "schema", "openAPIV3Schema"),
				required("spec", "versions[1]", "schema", "openAPIV3Schema"),
			},
		},
		{
			name: "preserveUnknownFields: true via v1",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors:    []validationMatch{invalid("spec.preserveUnknownFields")},
		},
		{
			name: "labelSelectorPath outside of .spec and .status",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version0",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							// null labelSelectorPath
							Name:    "version0",
							Served:  true,
							Storage: true,
							Subresources: &apiextensions.CustomResourceSubresources{
								Scale: &apiextensions.CustomResourceSubresourceScale{
									SpecReplicasPath:   ".spec.replicas",
									StatusReplicasPath: ".status.replicas",
								},
							},
						},
						{
							// labelSelectorPath under .status
							Name:    "version1",
							Served:  true,
							Storage: false,
							Subresources: &apiextensions.CustomResourceSubresources{
								Scale: &apiextensions.CustomResourceSubresourceScale{
									SpecReplicasPath:   ".spec.replicas",
									StatusReplicasPath: ".status.replicas",
									LabelSelectorPath:  strPtr(".status.labelSelector"),
								},
							},
						},
						{
							// labelSelectorPath under .spec
							Name:    "version2",
							Served:  true,
							Storage: false,
							Subresources: &apiextensions.CustomResourceSubresources{
								Scale: &apiextensions.CustomResourceSubresourceScale{
									SpecReplicasPath:   ".spec.replicas",
									StatusReplicasPath: ".status.replicas",
									LabelSelectorPath:  strPtr(".spec.labelSelector"),
								},
							},
						},
						{
							// labelSelectorPath outside of .spec and .status
							Name:    "version3",
							Served:  true,
							Storage: false,
							Subresources: &apiextensions.CustomResourceSubresources{
								Scale: &apiextensions.CustomResourceSubresourceScale{
									SpecReplicasPath:   ".spec.replicas",
									StatusReplicasPath: ".status.replicas",
									LabelSelectorPath:  strPtr(".labelSelector"),
								},
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version0"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("spec", "versions[3]", "subresources", "scale", "labelSelectorPath"),
			},
		},
		{
			name: "defaults with enabled feature gate via v1beta1",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type:    "number",
									Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
			requestGV:       apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "default"), // disallowed via v1beta1
			},
		},
		{
			name: "defaults with enabled feature gate via v1",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type:    "number",
									Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
			requestGV:       apiextensionsv1.SchemeGroupVersion,
		},
		{
			name: "x-kubernetes-int-or-string without structural",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Properties: map[string]apiextensions.JSONSchemaProps{
								"intorstring": {
									XIntOrString: true,
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "type"),
			},
		},
		{
			name: "x-kubernetes-preserve-unknown-fields without structural",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Properties: map[string]apiextensions.JSONSchemaProps{
								"raw": {
									XPreserveUnknownFields: pointer.BoolPtr(true),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "type"),
			},
		},
		{
			name: "x-kubernetes-embedded-resource without structural",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Properties: map[string]apiextensions.JSONSchemaProps{
								"embedded": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {Type: "string"},
									},
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "type"),
			},
		},
		{
			name: "x-kubernetes-embedded-resource with pruning and empty properties",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"nil": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties:        nil,
								},
								"empty": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties:        map[string]apiextensions.JSONSchemaProps{},
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "properties[nil]", "properties"),
				required("spec", "validation", "openAPIV3Schema", "properties[empty]", "properties"),
			},
		},
		{
			name: "x-kubernetes-embedded-resource inside resource meta",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"embedded": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type:                   "object",
											XEmbeddedResource:      true,
											XPreserveUnknownFields: pointer.BoolPtr(true),
										},
										"apiVersion": {
											Type: "string",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"foo": {
													Type:                   "object",
													XEmbeddedResource:      true,
													XPreserveUnknownFields: pointer.BoolPtr(true),
												},
											},
										},
										"kind": {
											Type: "string",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"foo": {
													Type:                   "object",
													XEmbeddedResource:      true,
													XPreserveUnknownFields: pointer.BoolPtr(true),
												},
											},
										},
									},
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[metadata]", "x-kubernetes-embedded-resource"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[apiVersion]", "properties[foo]", "x-kubernetes-embedded-resource"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[kind]", "properties[foo]", "x-kubernetes-embedded-resource"),
			},
		},
		{
			name: "defaults with enabled feature gate, unstructural schema",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {Default: jsonPtr(42.0)},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "properties[a]", "type"),
				required("spec", "validation", "openAPIV3Schema", "type"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "defaults with enabled feature gate, structural schema",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type:    "number",
									Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors:          []validationMatch{},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "defaults in value validation with enabled feature gate, structural schema",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "number",
									Not: &apiextensions.JSONSchemaProps{
										Default: jsonPtr(42.0),
									},
									AnyOf: []apiextensions.JSONSchemaProps{
										{
											Default: jsonPtr(42.0),
										},
									},
									AllOf: []apiextensions.JSONSchemaProps{
										{
											Default: jsonPtr(42.0),
										},
									},
									OneOf: []apiextensions.JSONSchemaProps{
										{
											Default: jsonPtr(42.0),
										},
									},
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "not", "default"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "allOf[0]", "default"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "anyOf[0]", "default"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "oneOf[0]", "default"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "invalid defaults with enabled feature gate, structural schema",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"foo": "abc",
										"bar": int64(42.0),
									}),
								},
								"b": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"foo": "abc",
									}),
								},
								"c": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"foo": int64(42),
									}),
								},
								"d": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"good": {
											Type:    "string",
											Pattern: "a",
										},
										"bad": {
											Type:    "string",
											Pattern: "b",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"good": "a",
										"bad":  "a",
									}),
								},
								"e": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"preserveUnknownFields": {
											Type: "object",
											Default: jsonPtr(map[string]interface{}{
												"foo": "abc",
												// this is under x-kubernetes-preserve-unknown-fields
												"bar": int64(42.0),
											}),
										},
										"nestedProperties": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"foo": {
													Type: "string",
												},
											},
											Default: jsonPtr(map[string]interface{}{
												"foo": "abc",
												"bar": int64(42.0),
											}),
										},
									},
									XPreserveUnknownFields: pointer.BoolPtr(true),
								},
								// x-kubernetes-embedded-resource: true
								"embedded-fine": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"foo":        "abc",
										"apiVersion": "foo/v1",
										"kind":       "v1",
										"metadata": map[string]interface{}{
											"name": "foo",
										},
									}),
								},
								"embedded-preserve": {
									Type:                   "object",
									XEmbeddedResource:      true,
									XPreserveUnknownFields: pointer.BoolPtr(true),
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"foo":        "abc",
										"apiVersion": "foo/v1",
										"kind":       "v1",
										"metadata": map[string]interface{}{
											"name": "foo",
										},
										"bar": int64(42),
									}),
								},
								"embedded-preserve-unpruned-objectmeta": {
									Type:                   "object",
									XEmbeddedResource:      true,
									XPreserveUnknownFields: pointer.BoolPtr(true),
									Properties: map[string]apiextensions.JSONSchemaProps{
										"foo": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"foo":        "abc",
										"apiVersion": "foo/v1",
										"kind":       "v1",
										"metadata": map[string]interface{}{
											"name": "foo",
											// allow: unknown fields under metadata are not rejected during CRD validation, but only pruned in storage creation
											"unspecified": "bar",
										},
										"bar": int64(42),
									}),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "properties[a]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[c]", "default", "foo"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[d]", "default", "bad"),
				// we also expected unpruned and valid defaults under x-kubernetes-preserve-unknown-fields. We could be more
				// strict here, but want to encourage proper specifications by forbidding other defaults.
				invalid("spec", "validation", "openAPIV3Schema", "properties[e]", "properties[preserveUnknownFields]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[e]", "properties[nestedProperties]", "default"),
			},

			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "additionalProperties at resource root",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"embedded1": {
									Type:              "object",
									XEmbeddedResource: true,
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{Type: "string"},
									},
								},
								"embedded2": {
									Type:                   "object",
									XEmbeddedResource:      true,
									XPreserveUnknownFields: pointer.BoolPtr(true),
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{Type: "string"},
									},
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded1]", "additionalProperties"),
				required("spec", "validation", "openAPIV3Schema", "properties[embedded1]", "properties"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded2]", "additionalProperties"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			// TODO: remove in a follow-up. This blocks is here for easy review.
			name: "v1.15 era tests for metadata defaults",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1",
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											// forbidden: no default for top-level metadata
											Default: jsonPtr(map[string]interface{}{
												"name": "foo",
											}),
										},
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"metadata": {
													Type: "object",
													Default: jsonPtr(map[string]interface{}{
														"name": "foo",
														// allow: unknown fields under metadata are not rejected during CRD validation, but only pruned in storage creation
														"unknown": int64(42),
													}),
												},
											},
										},
									},
								},
							},
						},
						{
							Name:    "v2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													// forbidden: no default in top-level metadata
													Default: jsonPtr("foo"),
												},
											},
										},
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type:    "string",
													Default: jsonPtr("v1"),
												},
												"kind": {
													Type:    "string",
													Default: jsonPtr("Pod"),
												},
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type:    "string",
															Default: jsonPtr("foo"),
														},
													},
												},
											},
										},
									},
								},
							},
						},
						{
							Name:    "v3",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type:    "string",
													Default: jsonPtr("v1"),
												},
												"kind": {
													Type: "string",
													// invalid: non-validating value in TypeMeta
													Default: jsonPtr("%"),
												},
												"metadata": {
													Type: "object",
													Default: jsonPtr(map[string]interface{}{
														"labels": map[string]interface{}{
															// invalid: non-validating nested field in ObjectMeta
															"bar": "x y",
														},
													}),
												},
											},
										},
									},
								},
							},
						},
						{
							Name:    "v4",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type: "string",
															// invalid: wrongly typed nested fields in ObjectMeta
															Default: jsonPtr(int64(42)),
														},
														"labels": {
															Type: "object",
															Properties: map[string]apiextensions.JSONSchemaProps{
																"bar": {
																	Type: "string",
																	// invalid: wrong typed nested fields in ObjectMeta
																	Default: jsonPtr(int64(42)),
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
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"v1"},
				},
			},
			errors: []validationMatch{
				// Forbidden: must not be set in top-level metadata
				forbidden("spec", "versions[0]", "schema", "openAPIV3Schema", "properties[metadata]", "default"),

				// Forbidden: must not be set in top-level metadata
				forbidden("spec", "versions[1]", "schema", "openAPIV3Schema", "properties[metadata]", "properties[name]", "default"),

				// Invalid value: "x y"
				invalid("spec", "versions[2]", "schema", "openAPIV3Schema", "properties[embedded]", "properties[metadata]", "default"),
				// Invalid value: "%": kind: Invalid value: "%"
				invalid("spec", "versions[2]", "schema", "openAPIV3Schema", "properties[embedded]", "properties[kind]", "default"),

				// Invalid value: wrongly typed
				invalid("spec", "versions[3]", "schema", "openAPIV3Schema", "properties[embedded]", "properties[metadata]", "properties[labels]", "properties[bar]", "default"),
				// Invalid value: wrongly typed
				invalid("spec", "versions[3]", "schema", "openAPIV3Schema", "properties[embedded]", "properties[metadata]", "properties[name]", "default"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "default inside additionalSchema",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1",
							Served:  true,
							Storage: true,
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"embedded": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"annotations": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															// forbidden: no default under additionalProperties inside of metadata
															Default: jsonPtr("abc"),
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
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"v1"},
				},
			},
			errors: []validationMatch{
				// Forbidden: must not be set inside additionalProperties applying to object metadata
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[metadata]", "properties[annotations]", "additionalProperties", "default"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "top-level metadata default",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1",
							Served:  true,
							Storage: true,
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"metadata": {
									Type: "object",
									// forbidden: no default for top-level metadata
									Default: jsonPtr(map[string]interface{}{
										"name": "foo",
									}),
								},
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"v1"},
				},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[metadata]", "default"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "embedded metadata defaults",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "v1",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1",
							Served:  true,
							Storage: true,
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"embedded": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											Default: jsonPtr(map[string]interface{}{
												"name": "foo",
											}),
										},
									},
								},

								"allowed-in-object-defaults": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type:    "string",
											Default: jsonPtr("v1"),
										},
										"kind": {
											Type:    "string",
											Default: jsonPtr("Pod"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type:    "string",
													Default: jsonPtr("foo"),
												},
											},
											// allowed: unknown fields outside metadata
											Default: jsonPtr(map[string]interface{}{
												"unknown": int64(42),
											}),
										},
									},
								},
								"allowed-object-defaults": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									XEmbeddedResource: true,
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "v1",
										"kind":       "Pod",
										"metadata": map[string]interface{}{
											"name":    "foo",
											"unknown": int64(42),
										},
									}),
								},
								"allowed-spanning-object-defaults": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "v1",
											"kind":       "Pod",
											"metadata": map[string]interface{}{
												"name":    "foo",
												"unknown": int64(42),
											},
										},
									}),
								},

								"unknown-field-object-defaults": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "v1",
										"kind":       "Pod",
										"metadata": map[string]interface{}{
											"name": "foo",
											// allowed: unspecified field in ObjectMeta
											"unknown": int64(42),
										},
										// forbidden: unspecified field
										"unknown": int64(42),
									}),
								},
								"unknown-field-spanning-object-defaults": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "v1",
											"kind":       "Pod",
											"metadata": map[string]interface{}{
												"name": "foo",
												// allowed: unspecified field in ObjectMeta
												"unknown": int64(42),
											},
											// forbidden: unspecified field
											"unknown": int64(42),
										},
										// forbidden: unspecified field
										"unknown": int64(42),
									}),
								},

								"x-preserve-unknown-fields-unknown-field-object-defaults": {
									Type:                   "object",
									XEmbeddedResource:      true,
									XPreserveUnknownFields: pointer.BoolPtr(true),
									Properties:             map[string]apiextensions.JSONSchemaProps{},
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "v1",
										"kind":       "Pod",
										"metadata": map[string]interface{}{
											"name": "foo",
											// allowed: unspecified field in ObjectMeta
											"unknown": int64(42),
										},
										// allowed: because x-kubernetes-preserve-unknown-fields: true
										"unknown": int64(42),
									}),
								},
								"x-preserve-unknown-fields-unknown-field-spanning-object-defaults": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:                   "object",
											XEmbeddedResource:      true,
											XPreserveUnknownFields: pointer.BoolPtr(true),
											Properties:             map[string]apiextensions.JSONSchemaProps{},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "v1",
											"kind":       "Pod",
											"metadata": map[string]interface{}{
												"name": "foo",
												// allowed: unspecified field in ObjectMeta
												"unknown": int64(42),
											},
											// allowed: because x-kubernetes-preserve-unknown-fields: true
											"unknown": int64(42),
										},
									}),
								},
								"x-preserve-unknown-fields-unknown-field-outside": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:                   "object",
											XEmbeddedResource:      true,
											XPreserveUnknownFields: pointer.BoolPtr(true),
											Properties:             map[string]apiextensions.JSONSchemaProps{},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "v1",
											"kind":       "Pod",
											"metadata": map[string]interface{}{
												"name": "foo",
												// allowed: unspecified field in ObjectMeta
												"unknown": int64(42),
											},
											// allowed: because x-kubernetes-preserve-unknown-fields: true
											"unknown": int64(42),
										},
										// forbidden: unspecified field
										"unknown": int64(42),
									}),
								},

								"wrongly-typed-in-object-defaults": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											// invalid: wrong type
											Default: jsonPtr(int64(42)),
										},
										"kind": {
											Type: "string",
											// invalid: wrong type
											Default: jsonPtr(int64(42)),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													// invalid: wrong type
													Default: jsonPtr(int64(42)),
												},
												"annotations": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
														},
													},
													// invalid: wrong type
													Default: jsonPtr(int64(42)),
												},
											},
										},
									},
								},
								"wrongly-typed-object-defaults-apiVersion": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										// invalid: wrong type
										"apiVersion": int64(42),
									}),
								},
								"wrongly-typed-object-defaults-kind": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										// invalid: wrong type
										"kind": int64(42),
									}),
								},
								"wrongly-typed-object-defaults-name": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"metadata": map[string]interface{}{
											// invalid: wrong type
											"name": int64(42),
										},
									}),
								},
								"wrongly-typed-object-defaults-labels": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"metadata": map[string]interface{}{
											"labels": map[string]interface{}{
												// invalid: wrong type
												"foo": int64(42),
											},
										},
									}),
								},
								"wrongly-typed-object-defaults-annotations": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"metadata": map[string]interface{}{
											// invalid: wrong type
											"annotations": int64(42),
										},
									}),
								},
								"wrongly-typed-object-defaults-metadata": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										// invalid: wrong type
										"metadata": int64(42),
									}),
								},

								"wrongly-typed-spanning-object-defaults-apiVersion": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											// invalid: wrong type
											"apiVersion": int64(42),
										},
									}),
								},
								"wrongly-typed-spanning-object-defaults-kind": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											// invalid: wrong type
											"kind": int64(42),
										},
									}),
								},
								"wrongly-typed-spanning-object-defaults-name": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"metadata": map[string]interface{}{
												"name": int64(42),
											},
										},
									}),
								},
								"wrongly-typed-spanning-object-defaults-labels": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"metadata": map[string]interface{}{
												"labels": map[string]interface{}{
													// invalid: wrong type
													"foo": int64(42),
												},
											},
										},
									}),
								},
								"wrongly-typed-spanning-object-defaults-annotations": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"metadata": map[string]interface{}{
												// invalid: wrong type
												"annotations": int64(42),
											},
										},
									}),
								},
								"wrongly-typed-spanning-object-defaults-metadata": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"metadata": int64(42),
										},
									}),
								},

								"invalid-in-object-defaults": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"kind": {
											Type: "string",
											// invalid
											Default: jsonPtr("%"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													// invalid
													Default: jsonPtr("%"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
														},
													},
													// invalid
													Default: jsonPtr(map[string]interface{}{
														"foo": "x y",
													}),
												},
											},
										},
									},
								},
								"invalid-object-defaults-kind": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										// invalid: wrongly typed
										"kind": "%",
									}),
								},
								"invalid-object-defaults-name": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										"kind":       "Foo",
										"metadata": map[string]interface{}{
											// invalid: wrongly typed
											"name": "%",
										},
									}),
								},
								"invalid-object-defaults-labels": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"something": {
											Type: "string",
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										"kind":       "Foo",
										"metadata": map[string]interface{}{
											"labels": map[string]interface{}{
												// invalid: wrongly typed
												"foo": "x y",
											},
										},
									}),
								},
								"invalid-spanning-object-defaults-kind": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											// invalid: wrongly typed
											"kind": "%",
										},
									}),
								},
								"invalid-spanning-object-defaults-name": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											"kind":       "Foo",
											"metadata": map[string]interface{}{
												// invalid: wrongly typed
												"name": "%",
											},
										},
									}),
								},
								"invalid-spanning-object-defaults-labels": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"something": {
													Type: "string",
												},
											},
										},
									},
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											"kind":       "Foo",
											"metadata": map[string]interface{}{
												"labels": map[string]interface{}{
													// invalid: wrongly typed
													"foo": "x y",
												},
											},
										},
									}),
								},

								"in-object-defaults-with-valid-constraints": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											// valid
											Default: jsonPtr("foo/v1"),
											Enum:    jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											// valid
											Default: jsonPtr("Foo"),
											Enum:    jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													// valid
													Default: jsonPtr("foo"),
													Enum:    jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
													// valid
													Default: jsonPtr(map[string]interface{}{
														"foo": "foo",
													}),
												},
											},
										},
									},
								},
								"metadata-defaults-with-valid-constraints": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
											// valid
											Default: jsonPtr(map[string]interface{}{
												"name": "foo",
												"labels": map[string]interface{}{
													"foo": "foo",
												},
											}),
										},
									},
								},
								"object-defaults-with-valid-constraints": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											Enum: jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											Enum: jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
										},
									},
									// valid
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										"kind":       "Foo",
										"metadata": map[string]interface{}{
											"name": "foo",
											"labels": map[string]interface{}{
												"foo": "foo",
											},
										},
									}),
								},
								"spanning-defaults-with-valid-constraints": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type: "string",
													Enum: jsonSlice("foo/v1"),
												},
												"kind": {
													Type: "string",
													Enum: jsonSlice("Foo"),
												},
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type: "string",
															Enum: jsonSlice("foo"),
														},
														"labels": {
															Type: "object",
															AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
																Schema: &apiextensions.JSONSchemaProps{
																	Type: "string",
																	Enum: jsonSlice("foo"),
																},
															},
														},
													},
												},
											},
										},
									},
									// valid
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											"kind":       "Foo",
											"metadata": map[string]interface{}{
												"name": "foo",
												"labels": map[string]interface{}{
													"foo": "foo",
												},
											},
										},
									}),
								},

								"in-object-defaults-with-invalid-constraints": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type:        "string",
											Description: "BREAK",
											// invalid
											Default: jsonPtr("bar/v1"),
											Enum:    jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											// invalid
											Default: jsonPtr("Bar"),
											Enum:    jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													// invalid
													Default: jsonPtr("bar"),
													Enum:    jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
													// invalid
													Default: jsonPtr(map[string]interface{}{
														"foo": "bar",
													}),
												},
											},
										},
									},
								},
								"metadata-defaults-with-invalid-constraints-name": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
											// invalid name
											Default: jsonPtr(map[string]interface{}{
												"name": "bar",
											}),
										},
									},
								},
								"metadata-defaults-with-invalid-constraints-labels": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
											// invalid labels
											Default: jsonPtr(map[string]interface{}{
												"name": "foo",
												"labels": map[string]interface{}{
													"foo": "bar",
												},
											}),
										},
									},
								},
								"object-defaults-with-invalid-constraints-name": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											Enum: jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											Enum: jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										"kind":       "Foo",
										"metadata": map[string]interface{}{
											"name": "bar",
										},
									}),
								},
								"object-defaults-with-invalid-constraints-labels": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											Enum: jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											Enum: jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										"kind":       "Foo",
										"metadata": map[string]interface{}{
											"name": "foo",
											"labels": map[string]interface{}{
												"foo": "bar",
											},
										},
									}),
								},
								"object-defaults-with-invalid-constraints-apiVersion": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											Enum: jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											Enum: jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "bar/v1",
										"kind":       "Foo",
										"metadata": map[string]interface{}{
											"name": "foo",
										},
									}),
								},
								"object-defaults-with-invalid-constraints-kind": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											Enum: jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											Enum: jsonSlice("Foo"),
										},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type: "string",
													Enum: jsonSlice("foo"),
												},
												"labels": {
													Type: "object",
													AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
														Schema: &apiextensions.JSONSchemaProps{
															Type: "string",
															Enum: jsonSlice("foo"),
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"apiVersion": "foo/v1",
										"kind":       "Bar",
										"metadata": map[string]interface{}{
											"name": "foo",
										},
									}),
								},
								"spanning-defaults-with-invalid-constraints-name": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type: "string",
													Enum: jsonSlice("foo/v1"),
												},
												"kind": {
													Type: "string",
													Enum: jsonSlice("Foo"),
												},
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type: "string",
															Enum: jsonSlice("foo"),
														},
														"labels": {
															Type: "object",
															AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
																Schema: &apiextensions.JSONSchemaProps{
																	Type: "string",
																	Enum: jsonSlice("foo"),
																},
															},
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											"kind":       "Foo",
											"metadata": map[string]interface{}{
												"name": "bar",
												"labels": map[string]interface{}{
													"foo": "foo",
												},
											},
										},
									}),
								},
								"spanning-defaults-with-invalid-constraints-labels": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type: "string",
													Enum: jsonSlice("foo/v1"),
												},
												"kind": {
													Type: "string",
													Enum: jsonSlice("Foo"),
												},
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type: "string",
															Enum: jsonSlice("foo"),
														},
														"labels": {
															Type: "object",
															AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
																Schema: &apiextensions.JSONSchemaProps{
																	Type: "string",
																	Enum: jsonSlice("foo"),
																},
															},
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											"kind":       "Foo",
											"metadata": map[string]interface{}{
												"name": "foo",
												"labels": map[string]interface{}{
													"foo": "bar",
												},
											},
										},
									}),
								},
								"spanning-defaults-with-invalid-constraints-apiVersion": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type: "string",
													Enum: jsonSlice("foo/v1"),
												},
												"kind": {
													Type: "string",
													Enum: jsonSlice("Foo"),
												},
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type: "string",
															Enum: jsonSlice("foo"),
														},
														"labels": {
															Type: "object",
															AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
																Schema: &apiextensions.JSONSchemaProps{
																	Type: "string",
																	Enum: jsonSlice("foo"),
																},
															},
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "bar/v1",
											"kind":       "Foo",
											"metadata": map[string]interface{}{
												"name": "foo",
												"labels": map[string]interface{}{
													"foo": "foo",
												},
											},
										},
									}),
								},
								"spanning-defaults-with-invalid-constraints-kind": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type: "string",
													Enum: jsonSlice("foo/v1"),
												},
												"kind": {
													Type: "string",
													Enum: jsonSlice("Foo"),
												},
												"metadata": {
													Type: "object",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type: "string",
															Enum: jsonSlice("foo"),
														},
														"labels": {
															Type: "object",
															AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
																Schema: &apiextensions.JSONSchemaProps{
																	Type: "string",
																	Enum: jsonSlice("foo"),
																},
															},
														},
													},
												},
											},
										},
									},
									// invalid
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"apiVersion": "foo/v1",
											"kind":       "Bar",
											"metadata": map[string]interface{}{
												"name": "foo",
												"labels": map[string]interface{}{
													"foo": "foo",
												},
											},
										},
									}),
								},

								"object-defaults-with-missing-typemeta": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {
											Type: "string",
											Enum: jsonSlice("foo/v1"),
										},
										"kind": {
											Type: "string",
											Enum: jsonSlice("Foo"),
										},
									},
									// invalid: kind and apiVersion are missing
									Default: jsonPtr(map[string]interface{}{
										"metadata": map[string]interface{}{
											"name": "bar",
										},
									}),
								},
								"spanning-defaults-with-missing-typemeta": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"embedded": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {
													Type: "string",
													Enum: jsonSlice("foo/v1"),
												},
												"kind": {
													Type: "string",
													Enum: jsonSlice("Foo"),
												},
											},
										},
									},
									// invalid: kind and apiVersion are missing
									Default: jsonPtr(map[string]interface{}{
										"embedded": map[string]interface{}{
											"metadata": map[string]interface{}{
												"name": "bar",
											},
										},
									}),
								},
							},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"v1"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "properties[unknown-field-object-defaults]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[unknown-field-spanning-object-defaults]", "default"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[x-preserve-unknown-fields-unknown-field-outside]", "default"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-in-object-defaults]", "properties[kind]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-in-object-defaults]", "properties[apiVersion]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-in-object-defaults]", "properties[metadata]", "properties[name]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-in-object-defaults]", "properties[metadata]", "properties[annotations]", "default"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-object-defaults-metadata]", "default", "metadata"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-object-defaults-apiVersion]", "default", "apiVersion"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-object-defaults-kind]", "default", "kind"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-object-defaults-name]", "default", "metadata"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-object-defaults-labels]", "default", "metadata"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-object-defaults-annotations]", "default", "metadata"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-spanning-object-defaults-metadata]", "default", "embedded", "metadata"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-spanning-object-defaults-apiVersion]", "default", "embedded", "apiVersion"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-spanning-object-defaults-kind]", "default", "embedded", "kind"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-spanning-object-defaults-name]", "default", "embedded", "metadata"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-spanning-object-defaults-labels]", "default", "embedded", "metadata"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[wrongly-typed-spanning-object-defaults-annotations]", "default", "embedded", "metadata"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-in-object-defaults]", "properties[metadata]", "properties[name]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-in-object-defaults]", "properties[metadata]", "properties[labels]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-in-object-defaults]", "properties[kind]", "default"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-object-defaults-kind]", "default", "kind"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-object-defaults-name]", "default", "metadata", "name"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-object-defaults-labels]", "default", "metadata", "labels"),

				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-spanning-object-defaults-kind]", "default", "embedded", "kind"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-spanning-object-defaults-name]", "default", "embedded", "metadata", "name"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid-spanning-object-defaults-labels]", "default", "embedded", "metadata", "labels"),

				unsupported("spec", "validation", "openAPIV3Schema", "properties[in-object-defaults-with-invalid-constraints]", "properties[apiVersion]", "default"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[in-object-defaults-with-invalid-constraints]", "properties[kind]", "default"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[in-object-defaults-with-invalid-constraints]", "properties[metadata]", "properties[name]", "default"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[in-object-defaults-with-invalid-constraints]", "properties[metadata]", "properties[labels]", "default", "foo"),

				unsupported("spec", "validation", "openAPIV3Schema", "properties[metadata-defaults-with-invalid-constraints-name]", "properties[metadata]", "default", "name"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[metadata-defaults-with-invalid-constraints-labels]", "properties[metadata]", "default", "labels", "foo"),

				unsupported("spec", "validation", "openAPIV3Schema", "properties[object-defaults-with-invalid-constraints-name]", "default", "metadata", "name"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[object-defaults-with-invalid-constraints-labels]", "default", "metadata", "labels", "foo"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[object-defaults-with-invalid-constraints-apiVersion]", "default", "apiVersion"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[object-defaults-with-invalid-constraints-kind]", "default", "kind"),

				unsupported("spec", "validation", "openAPIV3Schema", "properties[spanning-defaults-with-invalid-constraints-kind]", "default", "embedded", "kind"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[spanning-defaults-with-invalid-constraints-labels]", "default", "embedded", "metadata", "labels", "foo"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[spanning-defaults-with-invalid-constraints-apiVersion]", "default", "embedded", "apiVersion"),
				unsupported("spec", "validation", "openAPIV3Schema", "properties[spanning-defaults-with-invalid-constraints-name]", "default", "embedded", "metadata", "name"),

				required("spec", "validation", "openAPIV3Schema", "properties[object-defaults-with-missing-typemeta]", "default", "apiVersion"),
				required("spec", "validation", "openAPIV3Schema", "properties[object-defaults-with-missing-typemeta]", "default", "kind"),

				required("spec", "validation", "openAPIV3Schema", "properties[spanning-defaults-with-missing-typemeta]", "default", "embedded", "apiVersion"),
				required("spec", "validation", "openAPIV3Schema", "properties[spanning-defaults-with-missing-typemeta]", "default", "embedded", "kind"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "contradicting meta field types",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"apiVersion": {Type: "number"},
								"kind":       {Type: "number"},
								"metadata": {
									Type: "number",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {
											Type:    "string",
											Pattern: "abc",
										},
										"generateName": {
											Type:    "string",
											Pattern: "abc",
										},
										"generation": {
											Type: "integer",
										},
									},
								},
								"valid": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {Type: "string"},
										"kind":       {Type: "string"},
										"metadata": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type:    "string",
													Pattern: "abc",
												},
												"generateName": {
													Type:    "string",
													Pattern: "abc",
												},
												"generation": {
													Type:    "integer",
													Minimum: float64Ptr(42.0), // does not make sense, but is allowed for nested ObjectMeta
												},
											},
										},
									},
								},
								"invalid": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {Type: "number"},
										"kind":       {Type: "number"},
										"metadata": {
											Type: "number",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"name": {
													Type:    "string",
													Pattern: "abc",
												},
												"generateName": {
													Type:    "string",
													Pattern: "abc",
												},
												"generation": {
													Type:    "integer",
													Minimum: float64Ptr(42.0), // does not make sense, but is allowed for nested ObjectMeta
												},
											},
										},
									},
								},
								"nested": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"invalid": {
											Type:              "object",
											XEmbeddedResource: true,
											Properties: map[string]apiextensions.JSONSchemaProps{
												"apiVersion": {Type: "number"},
												"kind":       {Type: "number"},
												"metadata": {
													Type: "number",
													Properties: map[string]apiextensions.JSONSchemaProps{
														"name": {
															Type:    "string",
															Pattern: "abc",
														},
														"generateName": {
															Type:    "string",
															Pattern: "abc",
														},
														"generation": {
															Type:    "integer",
															Minimum: float64Ptr(42.0), // does not make sense, but is allowed for nested ObjectMeta
														},
													},
												},
											},
										},
									},
								},
								"noEmbeddedObject": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"apiVersion": {Type: "number"},
										"kind":       {Type: "number"},
										"metadata":   {Type: "number"},
									},
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[metadata]"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[apiVersion]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[kind]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[metadata]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid]", "properties[apiVersion]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid]", "properties[kind]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[invalid]", "properties[metadata]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[nested]", "properties[invalid]", "properties[apiVersion]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[nested]", "properties[invalid]", "properties[kind]", "type"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[nested]", "properties[invalid]", "properties[metadata]", "type"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, gate := range tc.enabledFeatures {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, gate, true)()
			}
			for _, gate := range tc.disabledFeatures {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, gate, false)()
			}

			// duplicate defaulting behaviour
			if tc.resource.Spec.Conversion != nil && tc.resource.Spec.Conversion.Strategy == apiextensions.WebhookConverter && len(tc.resource.Spec.Conversion.ConversionReviewVersions) == 0 {
				tc.resource.Spec.Conversion.ConversionReviewVersions = []string{"v1beta1"}
			}
			errs := ValidateCustomResourceDefinition(tc.resource, tc.requestGV)
			seenErrs := make([]bool, len(errs))

			for _, expectedError := range tc.errors {
				found := false
				for i, err := range errs {
					if expectedError.matches(err) && !seenErrs[i] {
						found = true
						seenErrs[i] = true
						break
					}
				}

				if !found {
					t.Errorf("expected %v at %v, got %v", expectedError.errorType, expectedError.path.String(), errs)
				}
			}

			for i, seen := range seenErrs {
				if !seen {
					t.Errorf("unexpected error: %v", errs[i])
				}
			}
		})
	}
}

func TestValidateCustomResourceDefinitionUpdate(t *testing.T) {
	tests := []struct {
		name             string
		old              *apiextensions.CustomResourceDefinition
		resource         *apiextensions.CustomResourceDefinition
		requestGV        schema.GroupVersion
		errors           []validationMatch
		enabledFeatures  []featuregate.Feature
		disabledFeatures []featuregate.Feature
	}{
		{
			name: "invalid type updates allowed via v1beta1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "bogus"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
		},
		{
			name: "invalid types updates disallowed via v1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "bogus"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				unsupported("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "invalid types updates allowed via v1 if old object has invalid types",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "bogus"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "bogus2"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
		},
		{
			name: "non-atomic items in lists of type set allowed if pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:       "object", // non-atomic
										Properties: map[string]apiextensions.JSONSchemaProps{},
									},
								},
							}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:       "object", // non-atomic
										Properties: map[string]apiextensions.JSONSchemaProps{},
									},
								},
							}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
		},
		{
			name: "reject non-atomic items in lists of type set if not pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:       "object", // non-atomic
										Properties: map[string]apiextensions.JSONSchemaProps{},
									},
								},
							}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "properties[bar]", "items", "x-kubernetes-map-type"),
			},
		},
		{
			name: "structural to non-structural updates allowed via v1beta1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "integer"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {}}, // untyped object
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
		},
		{
			name: "structural to non-structural updates not allowed via v1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "integer"}},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {}}, // untyped object
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "absent schema to non-structural updates not allowed via v1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Scope:                 apiextensions.ResourceScope("Cluster"),
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {}}, // untyped object
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "non-structural updates allowed via v1 if old object has non-structural schema",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true, Schema: &apiextensions.CustomResourceValidation{
							OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
								Type:       "object",
								Properties: map[string]apiextensions.JSONSchemaProps{"foo": {}}, // untyped object, non-structural
							},
						}},
						{Name: "version2", Served: true, Storage: false, Schema: &apiextensions.CustomResourceValidation{
							OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
								Type:       "object",
								Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "number"}}, // structural
							},
						}},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true, Schema: &apiextensions.CustomResourceValidation{
							OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
								Type:       "object",
								Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Description: "b"}}, // untyped object, non-structural
							},
						}},
						{Name: "version2", Served: true, Storage: false, Schema: &apiextensions.CustomResourceValidation{
							OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
								Type:       "object",
								Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Description: "a"}}, // untyped object, non-structural
							},
						}},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "webhookconfig: should pass on invalid ConversionReviewVersion with old invalid versions",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"invalid-version"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"invalid-version_0, invalid-version"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "webhookconfig: should fail on invalid ConversionReviewVersion with old valid versions",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"invalid-version"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"v1beta1", "invalid-version"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "conversion", "conversionReviewVersions"),
			},
		},
		{
			name: "webhookconfig: should fail on invalid ConversionReviewVersion with missing old versions",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
						ConversionReviewVersions: []string{"invalid-version"},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Conversion: &apiextensions.CustomResourceConversion{
						Strategy: apiextensions.ConversionStrategyType("Webhook"),
						WebhookClientConfig: &apiextensions.WebhookClientConfig{
							URL: strPtr("https://example.com/webhook"),
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "conversion", "conversionReviewVersions"),
			},
		},
		{
			name: "unchanged",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "unchanged-established",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					Conditions: []apiextensions.CustomResourceDefinitionCondition{
						{Type: apiextensions.Established, Status: apiextensions.ConditionTrue},
					},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "version-deleted",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					StoredVersions: []string{"version", "version2"},
					Conditions: []apiextensions.CustomResourceDefinitionCondition{
						{Type: apiextensions.Established, Status: apiextensions.ConditionTrue},
					},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					StoredVersions: []string{"version", "version2"},
				},
			},
			errors: []validationMatch{
				invalid("status", "storedVersions[1]"),
			},
		},
		{
			name: "changes",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					Conditions: []apiextensions.CustomResourceDefinitionCondition{
						{Type: apiextensions.Established, Status: apiextensions.ConditionFalse},
					},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "abc.com",
					Version: "version2",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version2",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Namespaced"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural2",
						Singular: "singular2",
						Kind:     "kind2",
						ListKind: "listkind2",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural2",
						Singular: "singular2",
						Kind:     "kind2",
						ListKind: "listkind2",
					},
					StoredVersions: []string{"version2"},
				},
			},
			errors: []validationMatch{
				immutable("spec", "group"),
				immutable("spec", "names", "plural"),
			},
		},
		{
			name: "changes-established",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "kind",
						ListKind: "listkind",
					},
					Conditions: []apiextensions.CustomResourceDefinitionCondition{
						{Type: apiextensions.Established, Status: apiextensions.ConditionTrue},
					},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "abc.com",
					Version: "version2",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version2",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.ResourceScope("Namespaced"),
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural2",
						Singular: "singular2",
						Kind:     "kind2",
						ListKind: "listkind2",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					AcceptedNames: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural2",
						Singular: "singular2",
						Kind:     "kind2",
						ListKind: "listkind2",
					},
					StoredVersions: []string{"version2"},
				},
			},
			errors: []validationMatch{
				immutable("spec", "group"),
				immutable("spec", "scope"),
				immutable("spec", "names", "kind"),
				immutable("spec", "names", "plural"),
			},
		},
		{
			name: "top-level and per-version fields are mutually exclusive",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:         "version",
							Served:       true,
							Storage:      true,
							Subresources: &apiextensions.CustomResourceSubresources{},
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
						{
							Name:    "version2",
							Served:  true,
							Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: validValidationSchema,
							},
							Subresources:             &apiextensions.CustomResourceSubresources{},
							AdditionalPrinterColumns: []apiextensions.CustomResourceColumnDefinition{{Name: "Alpha", Type: "string", JSONPath: ".spec.alpha"}},
						},
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validValidationSchema,
					},
					Subresources: &apiextensions.CustomResourceSubresources{},
					Scope:        apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				forbidden("spec", "validation"),
				forbidden("spec", "subresources"),
			},
		},
		{
			name: "switch off preserveUnknownFields with structural schema before and after",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validUnstructuralValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "properties[spec]", "type"),
				required("spec", "validation", "openAPIV3Schema", "properties[status]", "type"),
				required("spec", "validation", "openAPIV3Schema", "items", "type"),
			},
		},
		{
			name: "switch off preserveUnknownFields without structural schema before, but with after",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validUnstructuralValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{},
		},
		{
			name: "switch on preserveUnknownFields without structural schema",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: validUnstructuralValidationSchema,
					},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "switch to preserveUnknownFields: true is allowed via v1beta1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "switch to preserveUnknownFields: true is forbidden via v1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors:    []validationMatch{invalid("spec.preserveUnknownFields")},
		},
		{
			name: "keep preserveUnknownFields: true is allowed via v1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "schema not required via v1beta1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1beta1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "schema not required via v1 if old object is missing schema",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "schema not required via v1 if old object is missing schema for some versions",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}}},
						{Name: "version2", Served: true, Storage: false},
					},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true},
						{Name: "version2", Served: true, Storage: false},
					},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors:    []validationMatch{},
		},
		{
			name: "schema required via v1 if old object has top-level schema",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation:            &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object"}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:                 "group.com",
					Version:               "version",
					Versions:              []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec.versions[0].schema.openAPIV3Schema"),
			},
		},
		{
			name: "schema required via v1 if all versions of old object have schema",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object", Description: "1"}}},
						{Name: "version2", Served: true, Storage: false, Schema: &apiextensions.CustomResourceValidation{OpenAPIV3Schema: &apiextensions.JSONSchemaProps{Type: "object", Description: "2"}}},
					},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true},
						{Name: "version2", Served: true, Storage: false},
					},
					Scope:                 apiextensions.NamespaceScoped,
					Names:                 apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			requestGV: apiextensionsv1.SchemeGroupVersion,
			errors: []validationMatch{
				required("spec.versions[0].schema.openAPIV3Schema"),
				required("spec.versions[1].schema.openAPIV3Schema"),
			},
		},
		{
			name: "setting defaults with enabled feature gate via v1beta1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "number",
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type:    "number",
									Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV:       apiextensionsv1beta1.SchemeGroupVersion,
			errors:          []validationMatch{forbidden("spec.validation.openAPIV3Schema.properties[a].default")},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "setting defaults with enabled feature gate via v1",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "number",
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "plural.group.com",
					ResourceVersion: "42",
				},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "version",
							Served:  true,
							Storage: true,
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type:    "number",
									Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			requestGV:       apiextensionsv1.SchemeGroupVersion,
			errors:          []validationMatch{},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
		{
			name: "add default with enabled feature gate, structural schema, without pruning",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "number",
									//Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "42"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Scope:    apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type:    "number",
									Default: jsonPtr(42.0),
								},
							},
						},
					},
					PreserveUnknownFields: pointer.BoolPtr(true),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "preserveUnknownFields"),
			},
			enabledFeatures: []featuregate.Feature{features.CustomResourceDefaulting},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, gate := range tc.enabledFeatures {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, gate, true)()
			}
			for _, gate := range tc.disabledFeatures {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, gate, false)()
			}

			errs := ValidateCustomResourceDefinitionUpdate(tc.resource, tc.old, tc.requestGV)
			seenErrs := make([]bool, len(errs))

			for _, expectedError := range tc.errors {
				found := false
				for i, err := range errs {
					if expectedError.matches(err) && !seenErrs[i] {
						found = true
						seenErrs[i] = true
						break
					}
				}

				if !found {
					t.Errorf("expected %v at %v, got %v", expectedError.errorType, expectedError.path.String(), errs)
				}
			}

			for i, seen := range seenErrs {
				if !seen {
					t.Errorf("unexpected error: %v", errs[i])
				}
			}
		})
	}
}

func TestValidateCustomResourceDefinitionValidation(t *testing.T) {
	tests := []struct {
		name          string
		input         apiextensions.CustomResourceValidation
		statusEnabled bool
		opts          validationOptions
		wantError     bool
	}{
		{
			name:      "empty",
			input:     apiextensions.CustomResourceValidation{},
			wantError: false,
		},
		{
			name:          "empty with status",
			input:         apiextensions.CustomResourceValidation{},
			statusEnabled: true,
			wantError:     false,
		},
		{
			name: "root type without status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "string",
				},
			},
			statusEnabled: false,
			wantError:     false,
		},
		{
			name: "root type having invalid value, with status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "string",
				},
			},
			statusEnabled: true,
			wantError:     true,
		},
		{
			name: "non-allowed root field with status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					AnyOf: []apiextensions.JSONSchemaProps{
						{
							Description: "First schema",
						},
						{
							Description: "Second schema",
						},
					},
				},
			},
			statusEnabled: true,
			wantError:     true,
		},
		{
			name: "all allowed fields at the root of the schema with status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: validValidationSchema,
			},
			statusEnabled: true,
			wantError:     false,
		},
		{
			name: "null type",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Properties: map[string]apiextensions.JSONSchemaProps{
						"null": {
							Type: "null",
						},
					},
				},
			},
			wantError: true,
		},
		{
			name: "nullable at the root",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					Nullable: true,
				},
			},
			wantError: true,
		},
		{
			name: "nullable without type",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Properties: map[string]apiextensions.JSONSchemaProps{
						"nullable": {
							Nullable: true,
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "nullable with types",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Properties: map[string]apiextensions.JSONSchemaProps{
						"object": {
							Type:     "object",
							Nullable: true,
						},
						"array": {
							Type:     "array",
							Nullable: true,
						},
						"number": {
							Type:     "number",
							Nullable: true,
						},
						"string": {
							Type:     "string",
							Nullable: true,
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "must be structural, but isn't",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{},
			},
			opts:      validationOptions{requireStructuralSchema: true},
			wantError: true,
		},
		{
			name: "must be structural",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
				},
			},
			opts:      validationOptions{requireStructuralSchema: true},
			wantError: false,
		},
		{
			name: "require valid types, valid",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
				},
			},
			opts:      validationOptions{requireValidPropertyType: true, requireStructuralSchema: true},
			wantError: false,
		},
		{
			name: "require valid types, invalid",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "null",
				},
			},
			opts:      validationOptions{requireValidPropertyType: true, requireStructuralSchema: true},
			wantError: true,
		},
		{
			name: "require valid types, invalid",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "bogus",
				},
			},
			opts:      validationOptions{requireValidPropertyType: true, requireStructuralSchema: true},
			wantError: true,
		},
		{
			name: "invalid type with list type extension set",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "object",
					XListType: strPtr("set"),
				},
			},
			wantError: true,
		},
		{
			name: "unset type with list type extension set",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XListType: strPtr("set"),
				},
			},
			wantError: true,
		},
		{
			name: "invalid list type extension",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("invalid"),
				},
			},
			wantError: true,
		},
		{
			name: "invalid list type extension with list map keys extension non-empty",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("set"),
					XListMapKeys: []string{"key"},
				},
			},
			wantError: true,
		},
		{
			name: "unset list type extension with list map keys extension non-empty",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XListMapKeys: []string{"key"},
				},
			},
			wantError: true,
		},
		{
			name: "empty list map keys extension with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("map"),
				},
			},
			wantError: true,
		},
		{
			name: "no items schema with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
				},
			},
			wantError: true,
		},
		{
			name: "multiple schema items with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						JSONSchemas: []apiextensions.JSONSchemaProps{
							{
								Type: "string",
							}, {
								Type: "integer",
							},
						},
					},
				},
			},
			wantError: true,
		},
		{
			name: "non object item with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "string",
						},
					},
				},
			},
			wantError: true,
		},
		{
			name: "items with key missing from properties with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
			},
			wantError: true,
		},
		{
			name: "items with non scalar key property type with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"key": {
									Type: "object",
								},
							},
						},
					},
				},
			},
			wantError: true,
		},
		{
			name: "duplicate map keys with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key", "key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"key": {
									Type: "string",
								},
							},
						},
					},
				},
			},
			wantError: true,
		},
		{
			name: "allowed schema with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"keyA", "keyB"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"keyA": {
									Type: "string",
								},
								"keyB": {
									Type: "integer",
								},
							},
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "allowed list-type atomic",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("atomic"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "string",
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "allowed list-type atomic with non-atomic items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("atomic"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:       "object",
							Properties: map[string]apiextensions.JSONSchemaProps{},
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "allowed list-type set with scalar items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "string",
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "allowed list-type set with atomic map items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:     "object",
							XMapType: strPtr("atomic"),
							Properties: map[string]apiextensions.JSONSchemaProps{
								"foo": {Type: "string"},
							},
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "invalid list-type set with non-atomic map items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:     "object",
							XMapType: strPtr("granular"),
							Properties: map[string]apiextensions.JSONSchemaProps{
								"foo": {Type: "string"},
							},
						},
					},
				},
			},
			opts:      validationOptions{requireAtomicSetType: true},
			wantError: true,
		},
		{
			name: "invalid list-type set with unspecified map-type for map items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"foo": {Type: "string"},
							},
						},
					},
				},
			},
			opts:      validationOptions{requireAtomicSetType: true},
			wantError: true,
		},
		{
			name: "allowed list-type set with atomic list items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:      "array",
							XListType: strPtr("atomic"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "allowed list-type set with unspecified list-type in list items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "invalid list-type set with with non-atomic list items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:      "array",
							XListType: strPtr("set"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
				},
			},
			opts:      validationOptions{requireAtomicSetType: true},
			wantError: true,
		},
		{
			name: "invalid type with map type extension (granular)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "array",
					XMapType: strPtr("granular"),
				},
			},
			wantError: true,
		},
		{
			name: "unset type with map type extension (granular)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XMapType: strPtr("granular"),
				},
			},
			wantError: true,
		},
		{
			name: "invalid type with map type extension (atomic)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "array",
					XMapType: strPtr("atomic"),
				},
			},
			wantError: true,
		},
		{
			name: "unset type with map type extension (atomic)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XMapType: strPtr("atomic"),
				},
			},
			wantError: true,
		},
		{
			name: "invalid map type",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					XMapType: strPtr("badMapType"),
				},
			},
			wantError: true,
		},
		{
			name: "allowed type with map type extension (granular)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					XMapType: strPtr("granular"),
				},
			},
			wantError: false,
		},
		{
			name: "allowed type with map type extension (atomic)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					XMapType: strPtr("atomic"),
				},
			},
			wantError: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := validateCustomResourceDefinitionValidation(&tt.input, tt.statusEnabled, tt.opts, field.NewPath("spec", "validation"))
			if !tt.wantError && len(got) > 0 {
				t.Errorf("Expected no error, but got: %v", got)
			} else if tt.wantError && len(got) == 0 {
				t.Error("Expected error, but got none")
			}
		})
	}
}

func TestSchemaHasDefaults(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	if err := apiextensions.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	seed := rand.Int63()
	t.Logf("seed: %d", seed)
	fuzzerFuncs := fuzzer.MergeFuzzerFuncs(apiextensionsfuzzer.Funcs)
	f := fuzzer.FuzzerFor(fuzzerFuncs, rand.NewSource(seed), codecs)

	for i := 0; i < 10000; i++ {
		// fuzz internal types
		schema := &apiextensions.JSONSchemaProps{}
		f.Fuzz(schema)

		v1beta1Schema := &apiextensionsv1beta1.JSONSchemaProps{}
		if err := apiextensionsv1beta1.Convert_apiextensions_JSONSchemaProps_To_v1beta1_JSONSchemaProps(schema, v1beta1Schema, nil); err != nil {
			t.Fatal(err)
		}

		bs, err := json.Marshal(v1beta1Schema)
		if err != nil {
			t.Fatal(err)
		}

		expected := strings.Contains(strings.Replace(string(bs), `"default":null`, `"deleted":null`, -1), `"default":`)
		if got := schemaHasDefaults(schema); got != expected {
			t.Errorf("expected %v, got %v for: %s", expected, got, string(bs))
		}
	}
}

var example = apiextensions.JSON(`"This is an example"`)

var validValidationSchema = &apiextensions.JSONSchemaProps{
	Description:      "This is a description",
	Type:             "object",
	Format:           "date-time",
	Title:            "This is a title",
	Maximum:          float64Ptr(10),
	ExclusiveMaximum: true,
	Minimum:          float64Ptr(5),
	ExclusiveMinimum: true,
	MaxLength:        int64Ptr(10),
	MinLength:        int64Ptr(5),
	Pattern:          "^[a-z]$",
	MaxItems:         int64Ptr(10),
	MinItems:         int64Ptr(5),
	MultipleOf:       float64Ptr(3),
	Required:         []string{"spec", "status"},
	Properties: map[string]apiextensions.JSONSchemaProps{
		"spec": {
			Type: "object",
			Items: &apiextensions.JSONSchemaPropsOrArray{
				Schema: &apiextensions.JSONSchemaProps{
					Description: "This is a schema nested under Items",
					Type:        "string",
				},
			},
		},
		"status": {
			Type: "object",
		},
	},
	ExternalDocs: &apiextensions.ExternalDocumentation{
		Description: "This is an external documentation description",
	},
	Example: &example,
}

var validUnstructuralValidationSchema = &apiextensions.JSONSchemaProps{
	Description:      "This is a description",
	Type:             "object",
	Format:           "date-time",
	Title:            "This is a title",
	Maximum:          float64Ptr(10),
	ExclusiveMaximum: true,
	Minimum:          float64Ptr(5),
	ExclusiveMinimum: true,
	MaxLength:        int64Ptr(10),
	MinLength:        int64Ptr(5),
	Pattern:          "^[a-z]$",
	MaxItems:         int64Ptr(10),
	MinItems:         int64Ptr(5),
	MultipleOf:       float64Ptr(3),
	Required:         []string{"spec", "status"},
	Items: &apiextensions.JSONSchemaPropsOrArray{
		Schema: &apiextensions.JSONSchemaProps{
			Description: "This is a schema nested under Items",
		},
	},
	Properties: map[string]apiextensions.JSONSchemaProps{
		"spec":   {},
		"status": {},
	},
	ExternalDocs: &apiextensions.ExternalDocumentation{
		Description: "This is an external documentation description",
	},
	Example: &example,
}

func float64Ptr(f float64) *float64 {
	return &f
}

func int64Ptr(f int64) *int64 {
	return &f
}

func jsonPtr(x interface{}) *apiextensions.JSON {
	ret := apiextensions.JSON(x)
	return &ret
}

func jsonSlice(l ...interface{}) []apiextensions.JSON {
	if len(l) == 0 {
		return nil
	}
	ret := make([]apiextensions.JSON, 0, len(l))
	for _, x := range l {
		ret = append(ret, x)
	}
	return ret
}
