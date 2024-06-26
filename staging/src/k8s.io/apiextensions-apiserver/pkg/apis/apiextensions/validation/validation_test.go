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
	"context"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsfuzzer "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	celschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema/cel"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/apiserver/pkg/cel/library"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/pointer"
	"k8s.io/utils/ptr"
)

type validationMatch struct {
	path           *field.Path
	errorType      field.ErrorType
	containsString string
}

func required(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeRequired}
}
func invalid(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeInvalid}
}
func invalidtypecode(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeTypeInvalid}
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
func duplicate(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeDuplicate}
}
func tooMany(path ...string) validationMatch {
	return validationMatch{path: field.NewPath(path[0], path[1:]...), errorType: field.ErrorTypeTooMany}
}

func (v validationMatch) matches(err *field.Error) bool {
	return err.Type == v.errorType && err.Field == v.path.String() && strings.Contains(err.Error(), v.containsString)
}

func (v validationMatch) contains(s string) validationMatch {
	v.containsString = s
	return v
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
		name     string
		resource *apiextensions.CustomResourceDefinition
		errors   []validationMatch
	}{
		{
			name: "invalid types disallowed",
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
					PreserveUnknownFields: nil,
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version1"},
				},
			},
			errors: []validationMatch{},
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
					PreserveUnknownFields: nil,
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version1"},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
			},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"foo": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"bar": {Type: "object"},
									},
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{Allows: false},
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
				forbidden("spec", "validation", "openAPIV3Schema", "properties[foo]", "additionalProperties"),
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"foo": {
									Type: "object",
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Allows: true,
										Schema: &apiextensions.JSONSchemaProps{
											Type: "string",
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
			errors: []validationMatch{},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
			errors: []validationMatch{
				required("spec", "versions[0]", "schema", "openAPIV3Schema"),
				required("spec", "versions[1]", "schema", "openAPIV3Schema"),
			},
		},
		{
			name: "schema is required",
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
			errors: []validationMatch{
				required("spec", "versions[0]", "schema", "openAPIV3Schema"),
				required("spec", "versions[1]", "schema", "openAPIV3Schema"),
			},
		},
		{
			name: "preserveUnknownFields: true",
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
			errors: []validationMatch{invalid("spec.preserveUnknownFields")},
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
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version0"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "versions[3]", "subresources", "scale", "labelSelectorPath"),
			},
		},
		{
			name: "defaults with enabled feature gate",
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
							Type:                   "object",
							XPreserveUnknownFields: pointer.BoolPtr(true),
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
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
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
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[metadata]", "x-kubernetes-embedded-resource"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[apiVersion]", "properties[foo]", "x-kubernetes-embedded-resource"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[embedded]", "properties[kind]", "properties[foo]", "x-kubernetes-embedded-resource"),
			},
		},
		{
			name: "x-kubernetes-validations access metadata name",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Versions: singleVersionList,
					Subresources: &apiextensions.CustomResourceSubresources{
						Status: &apiextensions.CustomResourceSubresourceStatus{},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: pointer.BoolPtr(false),
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type:                   "object",
							XPreserveUnknownFields: pointer.BoolPtr(true),
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "size(self.metadata.name) > 3",
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"metadata": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {
											Type: "string",
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
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
			errors: []validationMatch{},
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
				invalidtypecode("spec", "validation", "openAPIV3Schema", "properties[c]", "default", "foo"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[d]", "default", "bad"),
				// we also expected unpruned and valid defaults under x-kubernetes-preserve-unknown-fields. We could be more
				// strict here, but want to encourage proper specifications by forbidding other defaults.
				invalid("spec", "validation", "openAPIV3Schema", "properties[e]", "properties[preserveUnknownFields]", "default"),
				invalid("spec", "validation", "openAPIV3Schema", "properties[e]", "properties[nestedProperties]", "default"),
			},
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
		},
		{
			name: "x-kubernetes-validations should be forbidden under oneOf/anyOf/allOf/not, structural schema",
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
										XValidations: apiextensions.ValidationRules{
											{
												Rule: "should be forbidden",
											},
										},
									},
									AnyOf: []apiextensions.JSONSchemaProps{
										{
											XValidations: apiextensions.ValidationRules{
												{
													Rule: "should be forbidden",
												},
											},
										},
									},
									AllOf: []apiextensions.JSONSchemaProps{
										{
											XValidations: apiextensions.ValidationRules{
												{
													Rule: "should be forbidden",
												},
											},
										},
									},
									OneOf: []apiextensions.JSONSchemaProps{
										{
											XValidations: apiextensions.ValidationRules{
												{
													Rule: "should be forbidden",
												},
											},
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
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "not", "x-kubernetes-validations"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "allOf[0]", "x-kubernetes-validations"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "anyOf[0]", "x-kubernetes-validations"),
				forbidden("spec", "validation", "openAPIV3Schema", "properties[a]", "oneOf[0]", "x-kubernetes-validations"),
			},
		},
		{
			name: "x-kubernetes-validations should have valid reason and fieldPath",
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
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.a > 0",
									Reason: func() *apiextensions.FieldValueErrorReason {
										r := apiextensions.FieldValueErrorReason("InternalError")
										return &r
									}(),
									FieldPath: ".a",
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "number",
									XValidations: apiextensions.ValidationRules{
										{
											Rule: "true",
											Reason: func() *apiextensions.FieldValueErrorReason {
												r := apiextensions.FieldValueRequired
												return &r
											}(),
										},
										{
											Rule: "true",
											Reason: func() *apiextensions.FieldValueErrorReason {
												r := apiextensions.FieldValueInvalid
												return &r
											}(),
										},
										{
											Rule: "true",
											Reason: func() *apiextensions.FieldValueErrorReason {
												r := apiextensions.FieldValueDuplicate
												return &r
											}(),
										},
										{
											Rule: "true",
											Reason: func() *apiextensions.FieldValueErrorReason {
												r := apiextensions.FieldValueForbidden
												return &r
											}(),
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				unsupported("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[0]", "reason"),
			},
		},
		{
			name: "x-kubernetes-validations should have valid fieldPath for array",
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
							XValidations: apiextensions.ValidationRules{
								{
									Rule:      "true",
									FieldPath: ".foo['b.c']['c\\a']",
								},
								{
									Rule:      "true",
									FieldPath: "['a.c']",
								},
								{
									Rule:      "true",
									FieldPath: ".a.c",
								},
								{
									Rule:      "true",
									FieldPath: ".list[0]",
								},
								{
									Rule:      "true",
									FieldPath: "   ",
								},
								{
									Rule:      "true",
									FieldPath: ".",
								},
								{
									Rule:      "true",
									FieldPath: "..",
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a.c": {
									Type: "number",
								},
								"foo": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"b.c": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"c\a": {
													Type: "number",
												},
											},
										},
									},
								},
								"list": {
									Type: "array",
									Items: &apiextensions.JSONSchemaPropsOrArray{
										Schema: &apiextensions.JSONSchemaProps{
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"a": {
													Type: "number",
												},
											},
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[2]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[3]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[4]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[4]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[5]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[6]", "fieldPath"),
			},
		},
		{
			name: "x-kubernetes-validations have invalid fieldPath",
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
							XValidations: apiextensions.ValidationRules{
								{
									Rule:      "self.a.b.c > 0.0",
									FieldPath: ".list[0].b",
								},
								{
									Rule:      "self.a.b.c > 0.0",
									FieldPath: ".list[0.b",
								},
								{
									Rule:      "self.a.b.c > 0.0",
									FieldPath: ".list0].b",
								},
								{
									Rule:      "self.a.b.c > 0.0",
									FieldPath: ".a.c",
								},
								{
									Rule:      "self.a.b.c > 0.0",
									FieldPath: ".a.b.d",
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"a": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"b": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"c": {
													Type: "number",
												},
											},
										},
									},
								},
								"list": {
									Type: "array",
									Items: &apiextensions.JSONSchemaPropsOrArray{
										Schema: &apiextensions.JSONSchemaProps{
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"a": {
													Type: "number",
												},
											},
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[0]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[1]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[2]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[3]", "fieldPath"),
				invalid("spec", "validation", "openAPIV3Schema", "x-kubernetes-validations[4]", "fieldPath"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			// duplicate defaulting behaviour
			if tc.resource.Spec.Conversion != nil && tc.resource.Spec.Conversion.Strategy == apiextensions.WebhookConverter && len(tc.resource.Spec.Conversion.ConversionReviewVersions) == 0 {
				tc.resource.Spec.Conversion.ConversionReviewVersions = []string{"v1beta1"}
			}
			ctx := context.TODO()
			errs := ValidateCustomResourceDefinition(ctx, tc.resource)
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

func TestSelectableFields(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, apiextensionsfeatures.CustomResourceFieldSelectors, true)
	singleVersionList := []apiextensions.CustomResourceDefinitionVersion{
		{
			Name:    "version",
			Served:  true,
			Storage: true,
		},
	}
	tests := []struct {
		name     string
		resource *apiextensions.CustomResourceDefinition
		errors   []validationMatch
	}{
		{
			name: "selectableFields with jsonPaths that do not refer to a field in the schema are invalid",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "string"}},
									Required:   []string{"foo"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{{JSONPath: ".foo"}, {JSONPath: ".xyz"}},
						},
						{Name: "version2", Served: true, Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "integer"}},
									Required:   []string{"foo"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{{JSONPath: ".xyz"}, {JSONPath: ".foo"}, {JSONPath: ".abc"}},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: ptr.To(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "versions[0]", "selectableFields[1].jsonPath"),
				invalid("spec", "versions[1]", "selectableFields[0].jsonPath"),
				invalid("spec", "versions[1]", "selectableFields[2].jsonPath"),
			},
		},
		{
			name: "in top level schema, selectableFields with jsonPaths that do not refer to a field in the schema are invalid",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Version:  "version",
					Versions: singleVersionList,
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"spec": {
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "string"}},
									Required:   []string{"foo"},
								},
								"status": {
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"phase": {Type: "string"}},
									Required:   []string{"phase"},
								},
							},
						},
					},
					SelectableFields: []apiextensions.SelectableField{{JSONPath: ".spec.foo"}, {JSONPath: ".spec.xyz"}, {JSONPath: ".status.phase"}},
					Scope:            apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: ptr.To(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "selectableFields[1].jsonPath"),
			},
		},
		{
			name: "selectableFields with jsonPaths that do not refer to fields that are not strings, booleans or integers are invalid",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "string"}, "obj": {Type: "object"}},
									Required:   []string{"foo", "obj"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{{JSONPath: ".foo"}, {JSONPath: ".obj"}},
						},
						{Name: "version2", Served: true, Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "integer"}, "obj": {Type: "object"}, "bool": {Type: "boolean"}},
									Required:   []string{"foo", "obj", "bool"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{{JSONPath: ".obj"}, {JSONPath: ".foo"}, {JSONPath: ".bool"}},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: ptr.To(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				invalid("spec", "versions[0]", "selectableFields[1].jsonPath"),
				invalid("spec", "versions[1]", "selectableFields[0].jsonPath"),
			},
		},
		{
			name: "selectableFields with duplicate jsonPaths are invalid",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "string"}},
									Required:   []string{"foo"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{{JSONPath: ".foo"}, {JSONPath: ".foo"}},
						},
						{Name: "version2", Served: true, Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "integer"}},
									Required:   []string{"foo"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{{JSONPath: ".foo"}, {JSONPath: ".foo"}},
						},
					},
					Scope: apiextensions.NamespaceScoped,
					Names: apiextensions.CustomResourceDefinitionNames{
						Plural:   "plural",
						Singular: "singular",
						Kind:     "Plural",
						ListKind: "PluralList",
					},
					PreserveUnknownFields: ptr.To(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				duplicate("spec", "versions[0]", "selectableFields[1].jsonPath"),
				duplicate("spec", "versions[1]", "selectableFields[1].jsonPath"),
			},
		},
		{
			name: "too many selectableFields are not allowed",
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Version: "version",
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{Name: "version", Served: true, Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"a1": {Type: "string"}, "a2": {Type: "string"}, "a3": {Type: "string"},
										"a4": {Type: "string"}, "a5": {Type: "string"}, "a6": {Type: "string"},
										"a7": {Type: "string"}, "a8": {Type: "string"}, "a9": {Type: "string"},
									},
									Required: []string{"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"},
								},
							},
							SelectableFields: []apiextensions.SelectableField{
								{JSONPath: ".a1"}, {JSONPath: ".a2"}, {JSONPath: ".a3"},
								{JSONPath: ".a4"}, {JSONPath: ".a5"}, {JSONPath: ".a6"},
								{JSONPath: ".a7"}, {JSONPath: ".a8"}, {JSONPath: ".a9"},
							},
						},
						{Name: "version2", Served: true, Storage: false,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type:       "object",
									Properties: map[string]apiextensions.JSONSchemaProps{"foo": {Type: "integer"}},
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
					PreserveUnknownFields: ptr.To(false),
				},
				Status: apiextensions.CustomResourceDefinitionStatus{
					StoredVersions: []string{"version"},
				},
			},
			errors: []validationMatch{
				tooMany("spec", "versions[0]", "selectableFields"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			// duplicate defaulting behaviour
			if tc.resource.Spec.Conversion != nil && tc.resource.Spec.Conversion.Strategy == apiextensions.WebhookConverter && len(tc.resource.Spec.Conversion.ConversionReviewVersions) == 0 {
				tc.resource.Spec.Conversion.ConversionReviewVersions = []string{"v1beta1"}
			}
			ctx := context.TODO()
			errs := ValidateCustomResourceDefinition(ctx, tc.resource)
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

func TestValidateFieldPath(t *testing.T) {
	schema := apiextensions.JSONSchemaProps{
		Type: "object",
		Properties: map[string]apiextensions.JSONSchemaProps{
			"foo": {
				Type: "object",
				AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
					Schema: &apiextensions.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensions.JSONSchemaProps{
							"f1": {
								Type: "number",
							},
						},
					},
				},
			},
			"a": {
				Type: "object",
				Properties: map[string]apiextensions.JSONSchemaProps{
					"bbb": {
						Type: "object",
						Properties: map[string]apiextensions.JSONSchemaProps{
							"c": {
								Type: "number",
							},
							"34": {
								Type: "number",
							},
						},
					},
					"bbb.c": {
						Type: "object",
						Properties: map[string]apiextensions.JSONSchemaProps{
							"a-b34": {
								Type: "number",
							},
						},
					},
				},
			},
			"list": {
				Type: "array",
				Items: &apiextensions.JSONSchemaPropsOrArray{
					Schema: &apiextensions.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensions.JSONSchemaProps{
							"a": {
								Type: "number",
							},
							"a-b.34": {
								Type: "number",
							},
						},
					},
				},
			},
		},
	}

	path := field.NewPath("")

	tests := []struct {
		name            string
		fieldPath       string
		pathOfFieldPath *field.Path
		schema          *apiextensions.JSONSchemaProps
		errMsg          string
	}{
		{
			name:            "Valid .a",
			fieldPath:       ".a",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .a.b",
			fieldPath:       ".a.bbb",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .foo.f1",
			fieldPath:       ".foo.f1",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Invalid map syntax .a.b",
			fieldPath:       ".a['bbb']",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .a['bbb.c']",
			fieldPath:       ".a['bbb.c']",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .a['bbb.c'].a-b34",
			fieldPath:       ".a['bbb.c'].a-b34",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .a['bbb.c']['a-b34']",
			fieldPath:       ".a['bbb.c']['a-b34']",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .a.bbb.c",
			fieldPath:       ".a.bbb.c",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .a.bbb.34",
			fieldPath:       ".a.bbb['34']",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Invalid map key",
			fieldPath:       ".a.foo",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "does not refer to a valid field",
		},
		{
			name:            "Malformed map key",
			fieldPath:       ".a.bbb[0]",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "expected single quoted string but got 0",
		},
		{
			name:            "number in field names",
			fieldPath:       ".a.bbb.34",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Special field names",
			fieldPath:       ".a.bbb['34']",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Valid .list",
			fieldPath:       ".list",
			pathOfFieldPath: path,
			schema:          &schema,
		},
		{
			name:            "Invalid .list[1]",
			fieldPath:       ".list[1]",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "expected single quoted string but got 1",
		},
		{
			name:            "Unsopported .list.a",
			fieldPath:       ".list.a",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "does not refer to a valid field",
		},
		{
			name:            "Unsupported .list['a-b.34']",
			fieldPath:       ".list['a-b.34']",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "does not refer to a valid field",
		},
		{
			name:            "Invalid .list.a-b.34",
			fieldPath:       ".list.a-b.34",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "does not refer to a valid field",
		},
		{
			name:            "Missing leading dot",
			fieldPath:       "a",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "expected [ or . but got: a",
		},
		{
			name:            "Nonexistent field",
			fieldPath:       ".c",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "does not refer to a valid field",
		},
		{
			name:            "Duplicate dots",
			fieldPath:       ".a..b",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "does not refer to a valid field",
		},
		{
			name:            "Negative array index",
			fieldPath:       ".list[-1]",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "expected single quoted string but got -1",
		},
		{
			name:            "Floating-point array index",
			fieldPath:       ".list[1.0]",
			pathOfFieldPath: path,
			schema:          &schema,
			errMsg:          "expected single quoted string but got 1",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ss, err := structuralschema.NewStructural(tc.schema)
			if err != nil {
				t.Fatalf("error when converting schema to structural schema: %v", err)
			}
			_, _, err = celschema.ValidFieldPath(tc.fieldPath, ss)
			if err == nil && tc.errMsg != "" {
				t.Errorf("expected err contains: %v but get nil", tc.errMsg)
			}
			if err != nil && tc.errMsg == "" {
				t.Errorf("unexpected error: %v", err)
			}
			if err != nil && !strings.Contains(err.Error(), tc.errMsg) {
				t.Errorf("expected error to contain: %v, but get: %v", tc.errMsg, err)
			}
		})
	}
}

func TestValidateCustomResourceDefinitionUpdate(t *testing.T) {
	tests := []struct {
		name     string
		old      *apiextensions.CustomResourceDefinition
		resource *apiextensions.CustomResourceDefinition
		errors   []validationMatch
	}{
		{
			name: "invalid types updates disallowed",
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
			errors: []validationMatch{
				unsupported("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "invalid types updates allowed if old object has invalid types",
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
			errors: []validationMatch{
				invalid("spec", "validation", "openAPIV3Schema", "properties[bar]", "items", "x-kubernetes-map-type"),
			},
		},
		{
			name: "structural to non-structural updates not allowed",
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
			errors: []validationMatch{
				required("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "absent schema to non-structural updates not allowed",
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
			errors: []validationMatch{
				required("spec.validation.openAPIV3Schema.properties[foo].type"),
			},
		},
		{
			name: "non-structural updates allowed if old object has non-structural schema",
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
			errors: []validationMatch{},
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
			name: "switch to preserveUnknownFields: true is forbidden",
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
			errors: []validationMatch{invalid("spec.preserveUnknownFields")},
		},
		{
			name: "keep preserveUnknownFields: true is allowed",
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
			errors: []validationMatch{},
		},
		{
			name: "schema not required if old object is missing schema",
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
			errors: []validationMatch{},
		},
		{
			name: "schema not required if old object is missing schema for some versions",
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
			errors: []validationMatch{},
		},
		{
			name: "schema required if old object has top-level schema",
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
			errors: []validationMatch{
				required("spec.versions[0].schema.openAPIV3Schema"),
			},
		},
		{
			name: "schema required if all versions of old object have schema",
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
			errors: []validationMatch{
				required("spec.versions[0].schema.openAPIV3Schema"),
				required("spec.versions[1].schema.openAPIV3Schema"),
			},
		},
		{
			name: "setting defaults with enabled feature gate",
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
			errors: []validationMatch{},
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
		},
		{
			name: "allow non-required key with no default in list of type map if pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
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
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
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
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: nil,
		},
		{
			name: "reject non-required key with no default in list of type map if not pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "object",
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type:    "string",
												Default: jsonPtr("stuff"),
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
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
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: []validationMatch{
				required("spec", "validation", "openAPIV3Schema", "properties[bar]", "items", "properties[key]", "default"),
			},
		},
		{
			name: "allow nullable key in list of type map if pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type:     "string",
												Nullable: true,
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type:     "string",
												Nullable: true,
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: nil,
		},
		{
			name: "reject nullable key in list of type map if not pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type: "string",
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type:     "string",
												Nullable: true,
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[bar]", "items", "properties[key]", "nullable"),
			},
		},
		{
			name: "allow nullable item in list of type map if pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Nullable: true,
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type: "string",
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Nullable: true,
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type: "string",
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: nil,
		},
		{
			name: "reject nullable item in list of type map if not pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type: "string",
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:         "array",
								XListType:    strPtr("map"),
								XListMapKeys: []string{"key"},
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "object",
										Nullable: true,
										Required: []string{"key"},
										Properties: map[string]apiextensions.JSONSchemaProps{
											"key": {
												Type: "string",
											},
										},
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[bar]", "items", "nullable"),
			},
		},
		{
			name: "allow nullable items in list of type set if pre-existing",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "string",
										Nullable: true,
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "string",
										Nullable: true,
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: nil,
		},
		{
			name: "reject nullable items in list of type set if not pre-exisiting",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"foo": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type: "string",
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
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
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{"bar": {
								Type:      "array",
								XListType: strPtr("set"),
								Items: &apiextensions.JSONSchemaPropsOrArray{
									Schema: &apiextensions.JSONSchemaProps{
										Type:     "string",
										Nullable: true,
									},
								},
							}},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			},
			errors: []validationMatch{
				forbidden("spec", "validation", "openAPIV3Schema", "properties[bar]", "items", "nullable"),
			},
		},
		{
			name: "suppress per-expression cost limit in pre-existing versions",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1",
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
												},
											},
										},
									},
								},
							},
						},
						{
							Name: "v2",
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
												},
											},
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"v1"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1", // unchanged
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
												},
											},
										},
									},
								},
							},
						},
						{
							Name: "v2", // touched
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7] + self[8]`,
												},
											},
										},
									},
								},
							},
						},
						{
							Name: "v3", // new
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
												},
											},
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"v1"}},
			},
			errors: []validationMatch{
				// versions[0] is exempted because it existed in oldObject
				forbidden("spec", "versions[1]", "schema", "openAPIV3Schema", "properties[f]", "x-kubernetes-validations[0]", "messageExpression"),
				forbidden("spec", "versions[2]", "schema", "openAPIV3Schema", "properties[f]", "x-kubernetes-validations[0]", "messageExpression"),
			},
		},
		{
			name: "suppress per-expression cost limit in new object during top-level schema to Versions extraction",
			old: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:   "group.com",
					Scope:   apiextensions.ResourceScope("Cluster"),
					Names:   apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Version: "v1",
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"f": {
									Type: "array",
									Items: &apiextensions.JSONSchemaPropsOrArray{
										Schema: &apiextensions.JSONSchemaProps{
											Type: "string",
										},
									},
									XValidations: apiextensions.ValidationRules{
										{
											Rule:              "true",
											MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"v1"}},
			},
			resource: &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group: "group.com",
					Scope: apiextensions.ResourceScope("Cluster"),
					Names: apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{
						{
							Name:    "v1", // unchanged, was top-level
							Served:  true,
							Storage: true,
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
												},
											},
										},
									},
								},
							},
						},
						{
							Name: "v2", // new
							Schema: &apiextensions.CustomResourceValidation{
								OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"f": {
											Type: "array",
											Items: &apiextensions.JSONSchemaPropsOrArray{
												Schema: &apiextensions.JSONSchemaProps{
													Type: "string",
												},
											},
											XValidations: apiextensions.ValidationRules{
												{
													Rule:              "true",
													MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7] + self[8]`,
												},
											},
										},
									},
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"v1"}},
			},
			errors: []validationMatch{
				// versions[0] is exempted because it existed in oldObject as top-level schema.
				forbidden("spec", "versions[1]", "schema", "openAPIV3Schema", "properties[f]", "x-kubernetes-validations[0]", "messageExpression"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.TODO()
			errs := ValidateCustomResourceDefinitionUpdate(ctx, tc.resource, tc.old)
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

func TestValidateCustomResourceDefinitionValidationRuleCompatibility(t *testing.T) {
	allRuleValidationsErrors := []validationMatch{
		invalid("spec", "validation", "openAPIV3Schema", "properties[x]", "x-kubernetes-validations[0]", "rule"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[obj]", "x-kubernetes-validations[0]", "rule"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[obj]", "properties[a]", "x-kubernetes-validations[0]", "rule"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[array]", "x-kubernetes-validations[0]", "rule"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[array]", "items", "x-kubernetes-validations[0]", "rule"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[map]", "x-kubernetes-validations[0]", "rule"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[map]", "additionalProperties", "x-kubernetes-validations[0]", "rule"),
	}
	allMessageExpressionValidationsErrors := []validationMatch{
		invalid("spec", "validation", "openAPIV3Schema", "properties[x]", "x-kubernetes-validations[0]", "messageExpression"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[obj]", "x-kubernetes-validations[0]", "messageExpression"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[obj]", "properties[a]", "x-kubernetes-validations[0]", "messageExpression"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[array]", "x-kubernetes-validations[0]", "messageExpression"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[array]", "items", "x-kubernetes-validations[0]", "messageExpression"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[map]", "x-kubernetes-validations[0]", "messageExpression"),
		invalid("spec", "validation", "openAPIV3Schema", "properties[map]", "additionalProperties", "x-kubernetes-validations[0]", "messageExpression"),
	}

	tests := []struct {
		name                     string
		storedRule               string
		updatedRule              string
		storedMessageExpression  string
		updatedMessageExpression string
		errors                   []validationMatch
	}{
		{
			name:                     "functions declared for storage mode allowed if expressions are unchanged from what is stored",
			storedRule:               "test() == true",
			updatedRule:              "test() == true",
			storedMessageExpression:  "'test: %s'.format([test()])",
			updatedMessageExpression: "'test: %s'.format([test()])",
		},
		{
			name:                     "functions declared for storage mode not allowed if rule expression is changed",
			storedRule:               "test() == false",
			updatedRule:              "test() == true", // rule was changed
			storedMessageExpression:  "'test: %s'.format([test()])",
			updatedMessageExpression: "'test: %s'.format([test()])",
			errors:                   allRuleValidationsErrors,
		},
		{
			name:                     "functions declared for storage mode not allowed if message expression is changed",
			storedRule:               "test() == true",
			updatedRule:              "test() == true",
			storedMessageExpression:  "'test: %s'.format([test()])",
			updatedMessageExpression: "'test - updated: %s'.format([test()])", // messageExpression was changed
			errors:                   allMessageExpressionValidationsErrors,
		},
	}

	// Include the test library, which includes the test() function in the storage environment during test
	base := environment.MustBaseEnvSet(version.MajorMinor(1, 998), true)
	envSet, err := base.Extend(environment.VersionedOptions{
		IntroducedVersion: version.MajorMinor(1, 999),
		EnvOptions:        []cel.EnvOption{library.Test()},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range tests {
		fn := func(rule, messageExpression string) *apiextensions.CustomResourceDefinition {
			validationRules := []apiextensions.ValidationRule{
				{
					Rule:              rule,
					MessageExpression: messageExpression,
				},
			}
			return &apiextensions.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
				Spec: apiextensions.CustomResourceDefinitionSpec{
					Group:    "group.com",
					Scope:    apiextensions.ResourceScope("Cluster"),
					Names:    apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
					Versions: []apiextensions.CustomResourceDefinitionVersion{{Name: "version", Served: true, Storage: true}},
					Validation: &apiextensions.CustomResourceValidation{
						OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"x": {
									Type:         "string",
									XValidations: validationRules,
								},
								"obj": {
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"a": {
											Type:         "string",
											XValidations: validationRules,
										},
									},
									XValidations: validationRules,
								},
								"array": {
									Type:     "array",
									MaxItems: pointer.Int64(1),
									Items: &apiextensions.JSONSchemaPropsOrArray{
										Schema: &apiextensions.JSONSchemaProps{
											Type:         "string",
											XValidations: validationRules,
										},
									},
									XValidations: validationRules,
								},
								"map": {
									Type:          "object",
									MaxProperties: pointer.Int64(1),
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{
											Type:         "string",
											XValidations: validationRules,
										},
									},
									XValidations: validationRules,
								},
							},
						},
					},
				},
				Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: []string{"version"}},
			}
		}
		old := fn(tc.storedRule, tc.storedMessageExpression)
		resource := fn(tc.updatedRule, tc.updatedMessageExpression)

		t.Run(tc.name, func(t *testing.T) {
			ctx := context.TODO()
			errs := validateCustomResourceDefinitionUpdate(ctx, resource, old, validationOptions{
				preexistingExpressions: findPreexistingExpressions(&old.Spec),
				celEnvironmentSet:      envSet,
			})
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
		name           string
		input          apiextensions.CustomResourceValidation
		statusEnabled  bool
		opts           validationOptions
		expectedErrors []validationMatch
	}{
		{
			name:  "empty",
			input: apiextensions.CustomResourceValidation{},
		},
		{
			name:          "empty with status",
			input:         apiextensions.CustomResourceValidation{},
			statusEnabled: true,
		},
		{
			name: "root type without status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "string",
				},
			},
			statusEnabled: false,
		},
		{
			name: "root type having invalid value, with status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "string",
				},
			},
			statusEnabled: true,
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.type"),
			},
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
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema"),
			},
		},
		{
			name: "all allowed fields at the root of the schema with status",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: validValidationSchema,
			},
			statusEnabled: true,
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
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.properties[null].type"),
			},
		},
		{
			name: "nullable at the root",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					Nullable: true,
				},
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.nullable"),
			},
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
		},
		{
			name: "must be structural, but isn't",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{},
			},
			opts: validationOptions{requireStructuralSchema: true},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "must be structural",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
				},
			},
			opts: validationOptions{requireStructuralSchema: true},
		},
		{
			name: "require valid types, valid",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
				},
			},
			opts: validationOptions{requireValidPropertyType: true, requireStructuralSchema: true},
		},
		{
			name: "require valid types, invalid",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "null",
				},
			},
			opts: validationOptions{requireValidPropertyType: true, requireStructuralSchema: true},
			expectedErrors: []validationMatch{
				// Invalid value: "null": must be object at the root
				unsupported("spec.validation.openAPIV3Schema.type"),
				// Forbidden: type cannot be set to null, use nullable as an alternative
				forbidden("spec.validation.openAPIV3Schema.type"),
				// Unsupported value: "null": supported values: "array", "boolean", "integer", "number", "object", "string"
				invalid("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "require valid types, invalid",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "bogus",
				},
			},
			opts: validationOptions{requireValidPropertyType: true, requireStructuralSchema: true},
			expectedErrors: []validationMatch{
				unsupported("spec.validation.openAPIV3Schema.type"),
				invalid("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "invalid type with list type extension set",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "object",
					XListType: strPtr("set"),
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "unset type with list type extension set",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XListType: strPtr("set"),
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "invalid list type extension",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("invalid"),
				},
			},
			expectedErrors: []validationMatch{
				unsupported("spec.validation.openAPIV3Schema.x-kubernetes-list-type"),
			},
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
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-list-type"),
			},
		},
		{
			name: "unset list type extension with list map keys extension non-empty",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XListMapKeys: []string{"key"},
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.x-kubernetes-list-type"),
			},
		},
		{
			name: "empty list map keys extension with list type extension map",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("map"),
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.x-kubernetes-list-map-keys"),
				required("spec.validation.openAPIV3Schema.items"),
			},
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
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.items"),
			},
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
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.items"),
				invalid("spec.validation.openAPIV3Schema.items"),
			},
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
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.items.type"),
			},
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
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-list-map-keys"),
			},
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
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.items.properties[key].type"),
			},
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
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-list-map-keys"),
			},
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
			opts: validationOptions{requireAtomicSetType: true},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.items.x-kubernetes-map-type"),
			},
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
			opts: validationOptions{requireAtomicSetType: true},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.items.x-kubernetes-map-type"),
			},
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
			opts: validationOptions{requireAtomicSetType: true},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.items.x-kubernetes-list-type"),
			},
		},
		{
			name: "invalid type with map type extension (granular)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "array",
					XMapType: strPtr("granular"),
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "unset type with map type extension (granular)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XMapType: strPtr("granular"),
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "invalid type with map type extension (atomic)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "array",
					XMapType: strPtr("atomic"),
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "unset type with map type extension (atomic)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					XMapType: strPtr("atomic"),
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.type"),
			},
		},
		{
			name: "invalid map type",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					XMapType: strPtr("badMapType"),
				},
			},
			expectedErrors: []validationMatch{
				unsupported("spec.validation.openAPIV3Schema.x-kubernetes-map-type"),
			},
		},
		{
			name: "allowed type with map type extension (granular)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					XMapType: strPtr("granular"),
				},
			},
		},
		{
			name: "allowed type with map type extension (atomic)",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:     "object",
					XMapType: strPtr("atomic"),
				},
			},
		},
		{
			name: "invalid map with non-required key and no default",
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
									Type: "string",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.items.properties[key].default"),
			},
		},
		{
			name: "allowed map with required key and no default",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:     "object",
							Required: []string{"key"},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"key": {
									Type: "string",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
		},
		{
			name: "allowed map with non-required key and default",
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
									Type:    "string",
									Default: jsonPtr("stuff"),
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				allowDefaults:                      true,
				requireMapListKeysMapSetValidation: true,
			},
		},
		{
			name: "invalid map with nullable key",
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
									Type:     "string",
									Nullable: true,
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.items.properties[key].default"),
				forbidden("spec.validation.openAPIV3Schema.items.properties[key].nullable"),
			},
		},
		{
			name: "invalid map with nullable items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"key"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:     "object",
							Nullable: true,
							Properties: map[string]apiextensions.JSONSchemaProps{
								"key": {
									Type: "string",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.items.nullable"),
				required("spec.validation.openAPIV3Schema.items.properties[key].default"),
			},
		},
		{
			name: "valid map with some required, some defaulted, and non-key fields",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "array",
					XListType:    strPtr("map"),
					XListMapKeys: []string{"a"},
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type:     "object",
							Required: []string{"a", "c"},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"key": {
									Type: "string",
								},
								"a": {
									Type: "string",
								},
								"b": {
									Type:    "string",
									Default: jsonPtr("stuff"),
								},
								"c": {
									Type: "int",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.items.properties[b].default"),
			},
		},
		{
			name: "invalid set with nullable items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Nullable: true,
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.items.nullable"),
			},
		},
		{
			name: "allowed set with non-nullable items",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:      "array",
					XListType: strPtr("set"),
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Nullable: false,
						},
					},
				},
			},
			opts: validationOptions{
				requireMapListKeysMapSetValidation: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for scalar element",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"subRoot": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "self.startsWith('s')",
									Message: "subRoot should start with 's'.",
								},
								{
									Rule:    "self.endsWith('s')",
									Message: "subRoot should end with 's'.",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for object",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self.minReplicas <= self.maxReplicas",
							Message: "minReplicas should be no greater than maxReplicas",
						},
					},
					Properties: map[string]apiextensions.JSONSchemaProps{
						"minReplicas": {
							Type: "integer",
						},
						"maxReplicas": {
							Type: "integer",
						},
					},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "invalid x-kubernetes-validations with empty rule",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					XValidations: apiextensions.ValidationRules{
						{Message: "empty rule"},
					},
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.x-kubernetes-validations[0].rule"),
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations with empty validators",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type:         "object",
					XValidations: apiextensions.ValidationRules{},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "invalid rule in x-kubernetes-validations",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"subRoot": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "self == true",
									Message: "subRoot should be true.",
								},
								{
									Rule:    "self.endsWith('s')",
									Message: "subRoot should end with 's'.",
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[subRoot].x-kubernetes-validations[0].rule"),
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for nested object under multiple fields",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "self.minReplicas <= self.maxReplicas",
							Message: "minReplicas should be no greater than maxReplicas.",
						},
					},
					Properties: map[string]apiextensions.JSONSchemaProps{
						"minReplicas": {
							Type: "integer",
						},
						"maxReplicas": {
							Type: "integer",
						},
						"subRule": {
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "self.isTest == true",
									Message: "isTest should be true.",
								},
							},
							Properties: map[string]apiextensions.JSONSchemaProps{
								"isTest": {
									Type: "boolean",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for object of array",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					XValidations: apiextensions.ValidationRules{
						{
							Rule:    "size(self.nestedObj[0]) == 10",
							Message: "size of first element in nestedObj should be equal to 10",
						},
					},
					Properties: map[string]apiextensions.JSONSchemaProps{
						"nestedObj": {
							Type: "array",
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
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for array",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "array",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "size(self) > 0",
									Message: "scoped field should contain more than 0 element.",
								},
							},
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for array of object",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Items: &apiextensions.JSONSchemaPropsOrArray{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "array",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "self[0].nestedObj.val > 0",
									Message: "val should be greater than 0.",
								},
							},
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									Properties: map[string]apiextensions.JSONSchemaProps{
										"nestedObj": {
											Type: "object",
											Properties: map[string]apiextensions.JSONSchemaProps{
												"val": {
													Type: "integer",
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
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid x-kubernetes-validations for escaping",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					XValidations: apiextensions.ValidationRules{
						{
							Rule: "self.__if__ > 0",
						},
						{
							Rule: "self.__namespace__ > 0",
						},
						{
							Rule: "self.self > 0",
						},
						{
							Rule: "self.int > 0",
						},
					},
					Properties: map[string]apiextensions.JSONSchemaProps{
						"if": {
							Type: "integer",
						},
						"namespace": {
							Type: "integer",
						},
						"self": {
							Type: "integer",
						},
						"int": {
							Type: "integer",
						},
					},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "invalid x-kubernetes-validations for escaping",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					XValidations: apiextensions.ValidationRules{
						{
							Rule: "self.if > 0",
						},
						{
							Rule: "self.namespace > 0",
						},
						{
							Rule: "self.unknownProp > 0",
						},
					},
					Properties: map[string]apiextensions.JSONSchemaProps{
						"if": {
							Type: "integer",
						},
						"namespace": {
							Type: "integer",
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-validations[0].rule"),
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-validations[1].rule"),
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-validations[2].rule"),
			},
			opts: validationOptions{
				requireStructuralSchema: true,
			},
		},
		{
			name: "valid default with x-kubernetes-validations",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"embedded": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"metadata": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {
											Type: "string",
											XValidations: apiextensions.ValidationRules{
												{
													Rule: "self == 'singleton'",
												},
											},
											Default: jsonPtr("singleton"),
										},
									},
								},
							},
						},
						"value": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.startsWith('kube')",
								},
							},
							Default: jsonPtr("kube-everything"),
						},
						"object": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "integer",
								},
								"field2": {
									Type: "integer",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.field1 < self.field2",
								},
							},
							Default: jsonPtr(map[string]interface{}{"field1": 1, "field2": 2}),
						},
					},
				},
			},
			opts: validationOptions{
				requireStructuralSchema: true,
				allowDefaults:           true,
			},
		},
		{
			name: "invalid default with x-kubernetes-validations",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"embedded": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"metadata": {
									Type:              "object",
									XEmbeddedResource: true,
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {
											Type: "string",
											XValidations: apiextensions.ValidationRules{
												{
													Rule: "self == 'singleton'",
												},
											},
											Default: jsonPtr("nope"),
										},
									},
								},
							},
						},
						"value": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.startsWith('kube')",
								},
							},
							Default: jsonPtr("nope"),
						},
						"object": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field1": {
									Type: "integer",
								},
								"field2": {
									Type: "integer",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.field1 < self.field2",
								},
							},
							Default: jsonPtr(map[string]interface{}{"field1": 2, "field2": 1}),
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[embedded].properties[metadata].properties[name].default"),
				invalid("spec.validation.openAPIV3Schema.properties[value].default"),
				invalid("spec.validation.openAPIV3Schema.properties[object].default"),
			},
			opts: validationOptions{
				requireStructuralSchema: true,
				allowDefaults:           true,
			},
		},
		{
			name: "rule is empty or not specified",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "integer",
							XValidations: apiextensions.ValidationRules{
								{
									Message: "something",
								},
								{
									Rule:    "   ",
									Message: "something",
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].rule"),
				required("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[1].rule"),
			},
		},
		{
			name: "multiline rule with message",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "integer",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:    "self >= 0 &&\nself <= 100",
									Message: "value must be between 0 and 100 (inclusive)",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "invalid and required messages",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "integer",
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self >= 0 &&\nself <= 100",
								},
								{
									Rule:    "self == 50",
									Message: "value requirements:\nmust be >= 0\nmust be <= 100 ",
								},
								{
									Rule:    "self == 50",
									Message: " ",
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				// message must be specified if rule contains line breaks
				required("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].message"),
				// message must not contain line breaks
				invalid("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[1].message"),
				// message must be non-empty if specified
				invalid("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[2].message"),
			},
		},
		{
			name: "forbid transition rule on element of list of type atomic",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "array",
							XListType: strPtr("atomic"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "forbid transition rule on element of list defaulting to type atomic",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "allow transition rule on list of type atomic",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "array",
							MaxItems:  int64ptr(10),
							XListType: strPtr("atomic"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on list defaulting to type atomic",
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "array",
							MaxItems: int64ptr(10),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "forbid transition rule on element of list of type set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "array",
							MaxItems:  int64ptr(10),
							XListType: strPtr("set"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "allow transition rule on list of type set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "array",
							MaxItems:  int64ptr(10),
							XListType: strPtr("set"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on element of list of type map",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:         "array",
							XListType:    strPtr("map"),
							XListMapKeys: []string{"name"},
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
									Required: []string{"name"},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {Type: "string", MaxLength: int64ptr(5)},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on list of type map",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:         "array",
							MaxItems:     int64ptr(10),
							XListType:    strPtr("map"),
							XListMapKeys: []string{"name"},
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:     "object",
									Required: []string{"name"},
									Properties: map[string]apiextensions.JSONSchemaProps{
										"name": {Type: "string", MaxLength: int64ptr(5)},
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on element of map of type granular",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "object",
							XMapType: strPtr("granular"),
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subfield": {
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "forbid transition rule on element of map of unrecognized type",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "object",
							XMapType: strPtr("future"),
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subfield": {
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].properties[subfield].x-kubernetes-validations[0].rule"),
				unsupported("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-map-type"),
			},
		},
		{
			name: "allow transition rule on element of map defaulting to type granular",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subfield": {
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on map of type granular",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "object",
							XMapType: strPtr("granular"),
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on map defaulting to type granular",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "object",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on element of map of type atomic",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "object",
							XMapType: strPtr("atomic"),
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subfield": {
									Type: "object",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == oldSelf"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "allow transition rule on map of type atomic",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "object",
							XMapType: strPtr("atomic"),
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == oldSelf"},
							},
						},
					},
				},
			},
		},
		{
			name: "forbid double-nested rule with no limit set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{
											Type:     "object",
											Required: []string{"key"},
											Properties: map[string]apiextensions.JSONSchemaProps{
												"key": {Type: "string"},
											},
										},
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self.all(x, x.all(y, x[y].key == x[y].key))"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				// exceeds per-rule limit and contributes to total limit being exceeded (1 error for each)
				forbidden("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].rule"),
				forbidden("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].rule"),
				// total limit is exceeded
				forbidden("spec.validation.openAPIV3Schema"),
			},
		},
		{
			name: "forbid double-nested rule with one limit set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{
											Type:     "object",
											Required: []string{"key"},
											Properties: map[string]apiextensions.JSONSchemaProps{
												"key": {Type: "string", MaxLength: int64ptr(10)},
											},
										},
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self.all(x, x.all(y, x[y].key == x[y].key))"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				// exceeds per-rule limit and contributes to total limit being exceeded (1 error for each)
				forbidden("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].rule"),
				forbidden("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].rule"),
				// total limit is exceeded
				forbidden("spec.validation.openAPIV3Schema"),
			},
		},
		{
			name: "allow double-nested rule with three limits set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "array",
							MaxItems: int64ptr(10),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:          "object",
									MaxProperties: int64ptr(10),
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{
											Type:     "object",
											Required: []string{"key"},
											Properties: map[string]apiextensions.JSONSchemaProps{
												"key": {Type: "string", MaxLength: int64ptr(10)},
											},
										},
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self.all(x, x.all(y, x[y].key == x[y].key))"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{},
		},
		{
			name: "allow double-nested rule with one limit set on outermost array",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "array",
							MaxItems: int64ptr(4),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "object",
									AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
										Schema: &apiextensions.JSONSchemaProps{
											Type:     "object",
											Required: []string{"key"},
											Properties: map[string]apiextensions.JSONSchemaProps{
												"key": {Type: "number"},
											},
										},
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self.all(x, x.all(y, x[y].key == x[y].key))"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{},
		},
		{
			name: "check for cardinality of 1 under root object",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "integer",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self < 1024"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{},
		},
		{
			name: "forbid validation rules where cost total exceeds total limit",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"list": {
							Type:     "array",
							MaxItems: int64Ptr(100000),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64Ptr(5000),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self.contains('keyword')"},
									},
								},
							},
						},
						"map": {
							Type:          "object",
							MaxProperties: int64Ptr(1000),
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Allows: true,
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64Ptr(5000),
									XValidations: apiextensions.ValidationRules{
										{Rule: "self.contains('keyword')"},
									},
								},
							},
						},
						"field": { // include a validation rule that does not contribute to total limit being exceeded (i.e. it is less than 1% of the limit)
							Type: "integer",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self > 50 && self < 100"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				// exceeds per-rule limit and contributes to total limit being exceeded (1 error for each)
				forbidden("spec.validation.openAPIV3Schema.properties[list].items.x-kubernetes-validations[0].rule"),
				forbidden("spec.validation.openAPIV3Schema.properties[list].items.x-kubernetes-validations[0].rule"),
				// contributes to total limit being exceeded, but does not exceed per-rule limit
				forbidden("spec.validation.openAPIV3Schema.properties[map].additionalProperties.x-kubernetes-validations[0].rule"),
				// total limit is exceeded
				forbidden("spec.validation.openAPIV3Schema"),
			},
		},
		{
			name: "skip CEL expression validation when OpenAPIv3 schema is an invalid structural schema",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					// illegal to have both Properties and AdditionalProperties
					Properties: map[string]apiextensions.JSONSchemaProps{
						"field": {
							Type: "integer",
						},
					},
					AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
						Schema: &apiextensions.JSONSchemaProps{
							Type: "string",
						},
					},
					XValidations: apiextensions.ValidationRules{
						{Rule: "self.invalidFieldName > 50"}, // invalid CEL rule
					},
				},
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.additionalProperties"), // illegal to have both properties and additional properties
				forbidden("spec.validation.openAPIV3Schema.additionalProperties"), // structural schema rule: illegal to have additional properties at root
				// Error for invalid CEL rule is NOT expected here because CEL rules are not checked when the schema is invalid
			},
		},
		{
			name: "skip CEL expression validation when OpenAPIv3 schema is an invalid structural schema at level below",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"field": {
							Type: "object",
							// illegal to have both Properties and AdditionalProperties
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "integer",
								},
							},
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
					XValidations: apiextensions.ValidationRules{
						{Rule: "self.invalidFieldName > 50"},
					},
				},
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.properties[field].additionalProperties"),
			},
		},
		{
			// So long at the schema information accessible to the CEL expression is valid, the expression should be validated.
			name: "do not skip when OpenAPIv3 schema is an invalid structural schema in a separate part of the schema tree",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {
							Type: "object",
							// illegal to have both Properties and AdditionalProperties
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "integer",
								},
							},
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
						"b": {
							Type: "object",
							Properties: map[string]apiextensions.JSONSchemaProps{
								"field": {
									Type: "integer",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "self.invalidFieldName > 50"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.properties[a].additionalProperties"),
				invalid("spec.validation.openAPIV3Schema.properties[b].x-kubernetes-validations[0].rule"),
			},
		},
		{
			// So long at the schema information accessible to the CEL expression is valid, the expression should be validated.
			name: "do not skip CEL expression validation when OpenAPIv3 schema is an invalid structural schema at level above",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {
							Type: "object",
							// illegal to have both Properties and AdditionalProperties
							Properties: map[string]apiextensions.JSONSchemaProps{
								"b": {
									Type: "integer",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == 'abc'"},
									},
								},
							},
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				forbidden("spec.validation.openAPIV3Schema.properties[a].additionalProperties"),
				invalid("spec.validation.openAPIV3Schema.properties[a].properties[b].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated for escaped property name",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f/2": {
							Type: "string",
						},
					},
					XValidations: apiextensions.ValidationRules{
						{Rule: "self.f__slash__2 == 1"}, // invalid comparison of string and int
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under array items",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == 1"}, // invalid comparison of string and int
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[a].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under array items, parent has rule",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == 1"}, // invalid comparison of string and int
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "1 == 1"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[a].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under additionalProperties",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {
							Type: "object",
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == 1"}, // invalid comparison of string and int
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[a].additionalProperties.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under additionalProperties, parent has rule",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"a": {
							Type: "object",
							AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
									XValidations: apiextensions.ValidationRules{
										{Rule: "self == 1"}, // invalid comparison of string and int
									},
								},
							},
							XValidations: apiextensions.ValidationRules{
								{Rule: "1 == 1"},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[a].additionalProperties.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under unescaped property name",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == 1"}, // invalid comparison of string and int
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under unescaped property name, parent has rule",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == 1"}, // invalid comparison of string and int
							},
						},
					},
					XValidations: apiextensions.ValidationRules{
						{Rule: "1 == 1"},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under escaped property name",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f/2": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == 1"}, // invalid comparison of string and int
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f/2].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under escaped property name, parent has rule",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f/2": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == 1"}, // invalid comparison of string and int
							},
						},
					},
					XValidations: apiextensions.ValidationRules{
						{Rule: "1 == 1"},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f/2].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under unescapable property name",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f@2": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == 1"}, // invalid comparison of string and int
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f@2].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule validated under unescapable property name, parent has rule",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f@2": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{Rule: "self == 1"}, // invalid comparison of string and int
							},
						},
					},
					XValidations: apiextensions.ValidationRules{
						{Rule: "1 == 1"},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f@2].x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "x-kubernetes-validations rule with messageExpression",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "self == \"string value\"",
									MessageExpression: `self + " should be \"string value\""`,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{},
		},
		{
			name: "x-kubernetes-validations rule allows both message and messageExpression",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "self == \"string value\"",
									Message:           `string should be set to "string value"`,
									MessageExpression: `self + " should be \"string value\""`,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{},
		},
		{
			name: "x-kubernetes-validations rule invalidated by messageExpression syntax error",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "self == \"string value\"",
									MessageExpression: `self + " `,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].messageExpression"),
			},
		},
		{
			name: "x-kubernetes-validations rule invalidated by messageExpression not returning a string",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "self == \"string value\"",
									MessageExpression: `256`,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].messageExpression"),
			},
		},
		{
			name: "x-kubernetes-validations rule invalidated by messageExpression exceeding per-expression estimated cost limit",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "true",
									MessageExpression: `self[0] + self[1] + self[2] + self[3] + self[4] + self[5] + self[6] + self[7]`,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				// forbidden due to messageExpression exceeding per-expression cost limit
				forbidden("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].messageExpression"),
			},
		},
		{
			name: "x-kubernetes-validations rule with lowerAscii check should be within estimated cost limit",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type:     "array",
							MaxItems: pointer.Int64(5),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: pointer.Int64(5),
								},
							},
							XValidations: apiextensions.ValidationRules{
								{
									Rule: "self.all(x, self.exists_one(y, x.lowerAscii() == y.lowerAscii()))",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "x-kubernetes-validations rule invalidated by messageExpression exceeding per-CRD estimated cost limit",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type: "string",
								},
							},
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "true",
									MessageExpression: `string(self[0]) + string(self[1]) + string(self[2])`,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				// forbidden due to per-CRD cost limit being exceeded
				forbidden("spec.validation.openAPIV3Schema"),
				// forbidden due to messageExpression exceeding per-expression cost limit
				forbidden("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].messageExpression"),
				// additional message indicated messageExpression's contribution to exceeding the per-CRD cost limit
				forbidden("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].messageExpression"),
			},
		},
		{
			name: "x-kubernetes-validations rule invalidated by messageExpression being only empty spaces",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"f": {
							Type: "string",
							XValidations: apiextensions.ValidationRules{
								{
									Rule:              "self == \"string value\"",
									MessageExpression: `     `,
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				required("spec.validation.openAPIV3Schema.properties[f].x-kubernetes-validations[0].messageExpression"),
			},
		},
		{
			name: "forbid transition rule on element of list of type atomic when optionalOldSelf is set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "array",
							XListType: strPtr("atomic"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: `self == oldSelf.orValue("")`, OptionalOldSelf: ptr.To(true)},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "forbid transition rule on element of list defaulting to type atomic when optionalOldSelf is set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type: "array",
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: `self == oldSelf.orValue("")`, OptionalOldSelf: ptr.To(true)},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "forbid transition rule on element of list of type set when optionalOldSelf is set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "array",
							MaxItems:  int64ptr(10),
							XListType: strPtr("set"),
							Items: &apiextensions.JSONSchemaPropsOrArray{
								Schema: &apiextensions.JSONSchemaProps{
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: `self == oldSelf.orValue("")`, OptionalOldSelf: ptr.To(true)},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].items.x-kubernetes-validations[0].rule"),
			},
		},
		{
			name: "forbid transition rule on element of map of unrecognized type when optionalOldSelf is set",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:     "object",
							XMapType: strPtr("future"),
							Properties: map[string]apiextensions.JSONSchemaProps{
								"subfield": {
									Type:      "string",
									MaxLength: int64ptr(10),
									XValidations: apiextensions.ValidationRules{
										{Rule: `self == oldSelf.orValue("")`, OptionalOldSelf: ptr.To(true)},
									},
								},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].properties[subfield].x-kubernetes-validations[0].rule"),
				unsupported("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-map-type"),
			},
		},
		{
			name: "forbid setting optionalOldSelf to true if oldSelf is not used",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "string",
							MaxLength: int64ptr(10),
							XValidations: apiextensions.ValidationRules{
								{Rule: `self == "foo"`, OptionalOldSelf: ptr.To(true)},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].optionalOldSelf"),
			},
		},
		{
			name: "forbid setting optionalOldSelf to false if oldSelf is not used",
			opts: validationOptions{requireStructuralSchema: true},
			input: apiextensions.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensions.JSONSchemaProps{
					Type: "object",
					Properties: map[string]apiextensions.JSONSchemaProps{
						"value": {
							Type:      "string",
							MaxLength: int64ptr(10),
							XValidations: apiextensions.ValidationRules{
								{Rule: `self == "foo"`, OptionalOldSelf: ptr.To(false)},
							},
						},
					},
				},
			},
			expectedErrors: []validationMatch{
				invalid("spec.validation.openAPIV3Schema.properties[value].x-kubernetes-validations[0].optionalOldSelf"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.TODO()
			if tt.opts.celEnvironmentSet == nil {
				tt.opts.celEnvironmentSet = environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true)
			}
			got := validateCustomResourceDefinitionValidation(ctx, &tt.input, tt.statusEnabled, tt.opts, field.NewPath("spec", "validation"))

			seenErrs := make([]bool, len(got))

			for _, expectedError := range tt.expectedErrors {
				found := false
				for i, err := range got {
					if expectedError.matches(err) && !seenErrs[i] {
						found = true
						seenErrs[i] = true
						break
					}
				}

				if !found {
					t.Errorf("expected %v at %v, got %v", expectedError.errorType, expectedError.path.String(), got)
				}
			}

			for i, seen := range seenErrs {
				if !seen {
					t.Errorf("unexpected error: %v", got[i])
				}
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

func TestValidateCustomResourceDefinitionStoredVersions(t *testing.T) {
	tests := []struct {
		name           string
		versions       []string
		storageVersion string
		storedVersions []string
		errors         []validationMatch
	}{
		{
			name:           "one version",
			versions:       []string{"v1"},
			storageVersion: "v1",
			storedVersions: []string{"v1"},
		},
		{
			name:           "no stored version",
			versions:       []string{"v1"},
			storageVersion: "v1",
			storedVersions: []string{},
			errors: []validationMatch{
				invalid("status", "storedVersions").contains("Invalid value: []string{}: must have at least one stored version"),
			},
		},
		{
			name:           "many versions",
			versions:       []string{"v1alpha", "v1beta1", "v1"},
			storageVersion: "v1",
			storedVersions: []string{"v1alpha", "v1"},
		},
		{
			name:           "missing stored versions",
			versions:       []string{"v1beta1", "v1"},
			storageVersion: "v1",
			storedVersions: []string{"v1alpha", "v1beta1", "v1"},
			errors: []validationMatch{
				invalidIndex(0, "status", "storedVersions").contains("Invalid value: \"v1alpha\": must appear in spec.versions"),
			},
		},
		{
			name:           "missing storage versions",
			versions:       []string{"v1alpha", "v1beta1", "v1"},
			storageVersion: "v1",
			storedVersions: []string{"v1alpha", "v1beta1"},
			errors: []validationMatch{
				invalid("status", "storedVersions").contains("Invalid value: []string{\"v1alpha\", \"v1beta1\"}: must have the storage version v1"),
			},
		},
	}

	for _, tc := range tests {
		crd := &apiextensions.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: "plural.group.com", ResourceVersion: "1"},
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Group: "group.com",
				Scope: "Cluster",
				Names: apiextensions.CustomResourceDefinitionNames{Plural: "plural", Singular: "singular", Kind: "Plural", ListKind: "PluralList"},
			},
			Status: apiextensions.CustomResourceDefinitionStatus{StoredVersions: tc.storedVersions},
		}
		for _, version := range tc.versions {
			v := apiextensions.CustomResourceDefinitionVersion{Name: version}
			if tc.storageVersion == version {
				v.Storage = true
			}
			crd.Spec.Versions = append(crd.Spec.Versions, v)
		}

		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateCustomResourceDefinitionStoredVersions(crd.Status.StoredVersions, crd.Spec.Versions, field.NewPath("status", "storedVersions"))
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

func BenchmarkSchemaHas(b *testing.B) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	if err := apiextensions.AddToScheme(scheme); err != nil {
		b.Fatal(err)
	}
	fuzzerFuncs := fuzzer.MergeFuzzerFuncs(apiextensionsfuzzer.Funcs)
	seed := int64(5577006791947779410)
	f := fuzzer.FuzzerFor(fuzzerFuncs, rand.NewSource(seed), codecs)
	// fuzz internal types
	schema := &apiextensions.JSONSchemaProps{}
	f.NilChance(0).NumElements(10, 10).MaxDepth(10).Fuzz(schema)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if SchemaHas(schema, func(_ *apiextensions.JSONSchemaProps) bool {
			return false
		}) {
			b.Errorf("Function returned true")
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

func Test_validateDeprecationWarning(t *testing.T) {
	tests := []struct {
		name string

		deprecated bool
		warning    *string

		want []string
	}{
		{
			name:       "not deprecated, nil warning",
			deprecated: false,
			warning:    nil,
			want:       nil,
		},

		{
			name:       "not deprecated, empty warning",
			deprecated: false,
			warning:    pointer.StringPtr(""),
			want:       []string{"can only be set for deprecated versions"},
		},
		{
			name:       "not deprecated, set warning",
			deprecated: false,
			warning:    pointer.StringPtr("foo"),
			want:       []string{"can only be set for deprecated versions"},
		},

		{
			name:       "utf-8",
			deprecated: true,
			warning:    pointer.StringPtr("Iñtërnâtiônàlizætiøn,💝🐹🌇⛔"),
			want:       nil,
		},
		{
			name:       "long warning",
			deprecated: true,
			warning:    pointer.StringPtr(strings.Repeat("x", 256)),
			want:       nil,
		},

		{
			name:       "too long warning",
			deprecated: true,
			warning:    pointer.StringPtr(strings.Repeat("x", 257)),
			want:       []string{"must be <= 256 characters long"},
		},
		{
			name:       "newline",
			deprecated: true,
			warning:    pointer.StringPtr("Test message\nfoo"),
			want:       []string{"must only contain printable UTF-8 characters; non-printable character found at index 12"},
		},
		{
			name:       "non-printable character",
			deprecated: true,
			warning:    pointer.StringPtr("Test message\u0008"),
			want:       []string{"must only contain printable UTF-8 characters; non-printable character found at index 12"},
		},
		{
			name:       "null character",
			deprecated: true,
			warning:    pointer.StringPtr("Test message\u0000"),
			want:       []string{"must only contain printable UTF-8 characters; non-printable character found at index 12"},
		},
		{
			name:       "non-utf-8",
			deprecated: true,
			warning:    pointer.StringPtr("Test message\xc5foo"),
			want:       []string{"must only contain printable UTF-8 characters"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := validateDeprecationWarning(tt.deprecated, tt.warning); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("validateDeprecationWarning() = %v, want %v", got, tt.want)
			}
		})
	}
}

func genMapSchema() *apiextensions.JSONSchemaProps {
	return &apiextensions.JSONSchemaProps{
		Type: "object",
		AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
			Schema: &apiextensions.JSONSchemaProps{
				Type: "string",
			},
		},
	}
}

func withMaxProperties(mapSchema *apiextensions.JSONSchemaProps, maxProps *int64) *apiextensions.JSONSchemaProps {
	mapSchema.MaxProperties = maxProps
	return mapSchema
}

func genArraySchema() *apiextensions.JSONSchemaProps {
	return &apiextensions.JSONSchemaProps{
		Type: "array",
	}
}

func withMaxItems(arraySchema *apiextensions.JSONSchemaProps, maxItems *int64) *apiextensions.JSONSchemaProps {
	arraySchema.MaxItems = maxItems
	return arraySchema
}

func genObjectSchema() *apiextensions.JSONSchemaProps {
	return &apiextensions.JSONSchemaProps{
		Type: "object",
	}
}

func TestCostInfo(t *testing.T) {
	tests := []struct {
		name                   string
		schema                 []*apiextensions.JSONSchemaProps
		expectedMaxCardinality *uint64
	}{
		{
			name: "object",
			schema: []*apiextensions.JSONSchemaProps{
				genObjectSchema(),
			},
			expectedMaxCardinality: uint64ptr(1),
		},
		{
			name: "array",
			schema: []*apiextensions.JSONSchemaProps{
				withMaxItems(genArraySchema(), int64ptr(5)),
			},
			expectedMaxCardinality: uint64ptr(5),
		},
		{
			name:                   "unbounded array",
			schema:                 []*apiextensions.JSONSchemaProps{genArraySchema()},
			expectedMaxCardinality: nil,
		},
		{
			name:                   "map",
			schema:                 []*apiextensions.JSONSchemaProps{withMaxProperties(genMapSchema(), int64ptr(10))},
			expectedMaxCardinality: uint64ptr(10),
		},
		{
			name: "unbounded map",
			schema: []*apiextensions.JSONSchemaProps{
				genMapSchema(),
			},
			expectedMaxCardinality: nil,
		},
		{
			name: "array inside map",
			schema: []*apiextensions.JSONSchemaProps{
				withMaxProperties(genMapSchema(), int64ptr(5)),
				withMaxItems(genArraySchema(), int64ptr(5)),
			},
			expectedMaxCardinality: uint64ptr(25),
		},
		{
			name: "unbounded array inside bounded map",
			schema: []*apiextensions.JSONSchemaProps{
				withMaxProperties(genMapSchema(), int64ptr(5)),
				genArraySchema(),
			},
			expectedMaxCardinality: nil,
		},
		{
			name: "object inside array",
			schema: []*apiextensions.JSONSchemaProps{
				withMaxItems(genArraySchema(), int64ptr(3)),
				genObjectSchema(),
			},
			expectedMaxCardinality: uint64ptr(3),
		},
		{
			name: "map inside object inside array",
			schema: []*apiextensions.JSONSchemaProps{
				withMaxItems(genArraySchema(), int64ptr(2)),
				genObjectSchema(),
				withMaxProperties(genMapSchema(), int64ptr(4)),
			},
			expectedMaxCardinality: uint64ptr(8),
		},
		{
			name: "integer overflow bounds check",
			schema: []*apiextensions.JSONSchemaProps{
				withMaxItems(genArraySchema(), int64ptr(math.MaxInt)),
				withMaxItems(genArraySchema(), int64ptr(100)),
			},
			expectedMaxCardinality: uint64ptr(math.MaxUint),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// simulate the recursive validation calls
			schemas := append(tt.schema, &apiextensions.JSONSchemaProps{Type: "string"}) // append a leaf type
			curCostInfo := RootCELContext(schemas[0])
			for i := 1; i < len(schemas); i++ {
				curCostInfo = curCostInfo.childContext(schemas[i], nil)
			}
			if tt.expectedMaxCardinality == nil && curCostInfo.MaxCardinality == nil {
				// unbounded cardinality case, test ran correctly
			} else if tt.expectedMaxCardinality == nil && curCostInfo.MaxCardinality != nil {
				t.Errorf("expected unbounded cardinality (got %d)", curCostInfo.MaxCardinality)
			} else if tt.expectedMaxCardinality != nil && curCostInfo.MaxCardinality == nil {
				t.Errorf("expected bounded cardinality of %d but got unbounded cardinality", tt.expectedMaxCardinality)
			} else if *tt.expectedMaxCardinality != *curCostInfo.MaxCardinality {
				t.Errorf("wrong cardinality (expected %d, got %d)", *tt.expectedMaxCardinality, curCostInfo.MaxCardinality)
			}
		})
	}
}

func TestCelContext(t *testing.T) {
	tests := []struct {
		name   string
		schema *apiextensions.JSONSchemaProps
	}{
		{
			name: "verify that schemas are converted only once and then reused",
			schema: &apiextensions.JSONSchemaProps{
				Type:         "object",
				XValidations: []apiextensions.ValidationRule{{Rule: "self.size() < 100"}},
				AdditionalProperties: &apiextensions.JSONSchemaPropsOrBool{
					Schema: &apiextensions.JSONSchemaProps{
						Type: "array",
						Items: &apiextensions.JSONSchemaPropsOrArray{
							Schema: &apiextensions.JSONSchemaProps{
								Type:         "object",
								XValidations: []apiextensions.ValidationRule{{Rule: "has(self.field)"}},
								Properties: map[string]apiextensions.JSONSchemaProps{
									"field": {
										XValidations: []apiextensions.ValidationRule{{Rule: "self.startsWith('abc')"}},
										Type:         "string",
									},
								},
							},
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// simulate the recursive validation calls
			conversionCount := 0
			converter := func(schema *apiextensions.JSONSchemaProps, isRoot bool) (*CELTypeInfo, error) {
				conversionCount++
				return defaultConverter(schema, isRoot)
			}
			celContext := RootCELContext(tt.schema)
			celContext.converter = converter
			opts := validationOptions{
				celEnvironmentSet: environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true),
			}
			openAPIV3Schema := &specStandardValidatorV3{
				allowDefaults:            opts.allowDefaults,
				disallowDefaultsReason:   opts.disallowDefaultsReason,
				requireValidPropertyType: opts.requireValidPropertyType,
			}
			errors := ValidateCustomResourceDefinitionOpenAPISchema(tt.schema, field.NewPath("openAPIV3Schema"), openAPIV3Schema, true, &opts, celContext).AllErrors()
			if len(errors) != 0 {
				t.Errorf("Expected no validate errors but got %v", errors)
			}
			if conversionCount != 1 {
				t.Errorf("Expected 1 conversion to be performed by cel context during schema traversal but observed %d conversions", conversionCount)
			}
		})
	}
}

func TestPerCRDEstimatedCost(t *testing.T) {
	tests := []struct {
		name              string
		costs             []uint64
		expectedExpensive []uint64
		expectedTotal     uint64
	}{
		{
			name:              "no costs",
			costs:             []uint64{},
			expectedExpensive: []uint64{},
			expectedTotal:     uint64(0),
		},
		{
			name:              "one cost",
			costs:             []uint64{1000000},
			expectedExpensive: []uint64{1000000},
			expectedTotal:     uint64(1000000),
		},
		{
			name:              "one cost, ignored", // costs < 1% of the per-CRD cost limit are not considered expensive
			costs:             []uint64{900000},
			expectedExpensive: []uint64{},
			expectedTotal:     uint64(900000),
		},
		{
			name:              "2 costs",
			costs:             []uint64{5000000, 25000000},
			expectedExpensive: []uint64{25000000, 5000000},
			expectedTotal:     uint64(30000000),
		},
		{
			name:              "3 costs, one ignored",
			costs:             []uint64{5000000, 25000000, 900000},
			expectedExpensive: []uint64{25000000, 5000000},
			expectedTotal:     uint64(30900000),
		},
		{
			name:              "4 costs",
			costs:             []uint64{16000000, 50000000, 34000000, 50000000},
			expectedExpensive: []uint64{50000000, 50000000, 34000000, 16000000},
			expectedTotal:     uint64(150000000),
		},
		{
			name:              "5 costs, one trimmed, one ignored", // only the top 4 most expensive are tracked
			costs:             []uint64{16000000, 50000000, 900000, 34000000, 50000000, 50000001},
			expectedExpensive: []uint64{50000001, 50000000, 50000000, 34000000},
			expectedTotal:     uint64(200900001),
		},
		{
			name:              "costs do not overflow",
			costs:             []uint64{math.MaxUint64 / 2, math.MaxUint64 / 2, 1, 10, 100, 1000},
			expectedExpensive: []uint64{math.MaxUint64 / 2, math.MaxUint64 / 2},
			expectedTotal:     uint64(math.MaxUint64),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			crdCost := TotalCost{}
			for _, cost := range tt.costs {
				crdCost.ObserveExpressionCost(nil, cost)
			}
			if len(crdCost.MostExpensive) != len(tt.expectedExpensive) {
				t.Fatalf("expected %d largest costs but got %d: %v", len(tt.expectedExpensive), len(crdCost.MostExpensive), crdCost.MostExpensive)
			}
			for i, expensive := range crdCost.MostExpensive {
				if tt.expectedExpensive[i] != expensive.Cost {
					t.Errorf("expected largest cost of %d at index %d but got %d", tt.expectedExpensive[i], i, expensive.Cost)
				}
			}
			if tt.expectedTotal != crdCost.Total {
				t.Errorf("expected total cost of %d but got %d", tt.expectedTotal, crdCost.Total)
			}
		})
	}
}

func int64ptr(i int64) *int64 {
	return &i
}
