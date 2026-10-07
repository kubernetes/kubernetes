/*
Copyright The Kubernetes Authors.

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

package v1beta1

import (
	v1beta1 "k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConditionStatus instead.
//
//go:fix inline
type ConditionStatus = v1beta1.ConditionStatus

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConversionRequest instead.
//
//go:fix inline
type ConversionRequest = v1beta1.ConversionRequest

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConversionResponse instead.
//
//go:fix inline
type ConversionResponse = v1beta1.ConversionResponse

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConversionReview instead.
//
//go:fix inline
type ConversionReview = v1beta1.ConversionReview

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConversionStrategyType instead.
//
//go:fix inline
type ConversionStrategyType = v1beta1.ConversionStrategyType

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceColumnDefinition instead.
//
//go:fix inline
type CustomResourceColumnDefinition = v1beta1.CustomResourceColumnDefinition

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceConversion instead.
//
//go:fix inline
type CustomResourceConversion = v1beta1.CustomResourceConversion

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinition instead.
//
//go:fix inline
type CustomResourceDefinition = v1beta1.CustomResourceDefinition

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionCondition instead.
//
//go:fix inline
type CustomResourceDefinitionCondition = v1beta1.CustomResourceDefinitionCondition

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionConditionType instead.
//
//go:fix inline
type CustomResourceDefinitionConditionType = v1beta1.CustomResourceDefinitionConditionType

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionList instead.
//
//go:fix inline
type CustomResourceDefinitionList = v1beta1.CustomResourceDefinitionList

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionNames instead.
//
//go:fix inline
type CustomResourceDefinitionNames = v1beta1.CustomResourceDefinitionNames

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionSpec instead.
//
//go:fix inline
type CustomResourceDefinitionSpec = v1beta1.CustomResourceDefinitionSpec

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionStatus instead.
//
//go:fix inline
type CustomResourceDefinitionStatus = v1beta1.CustomResourceDefinitionStatus

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceDefinitionVersion instead.
//
//go:fix inline
type CustomResourceDefinitionVersion = v1beta1.CustomResourceDefinitionVersion

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceSubresourceScale instead.
//
//go:fix inline
type CustomResourceSubresourceScale = v1beta1.CustomResourceSubresourceScale

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceSubresourceStatus instead.
//
//go:fix inline
type CustomResourceSubresourceStatus = v1beta1.CustomResourceSubresourceStatus

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceSubresources instead.
//
//go:fix inline
type CustomResourceSubresources = v1beta1.CustomResourceSubresources

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceValidation instead.
//
//go:fix inline
type CustomResourceValidation = v1beta1.CustomResourceValidation

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ExternalDocumentation instead.
//
//go:fix inline
type ExternalDocumentation = v1beta1.ExternalDocumentation

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.FieldValueErrorReason instead.
//
//go:fix inline
type FieldValueErrorReason = v1beta1.FieldValueErrorReason

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSON instead.
//
//go:fix inline
type JSON = v1beta1.JSON

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaDefinitions instead.
//
//go:fix inline
type JSONSchemaDefinitions = v1beta1.JSONSchemaDefinitions

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaDependencies instead.
//
//go:fix inline
type JSONSchemaDependencies = v1beta1.JSONSchemaDependencies

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaProps instead.
//
//go:fix inline
type JSONSchemaProps = v1beta1.JSONSchemaProps

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaPropsOrArray instead.
//
//go:fix inline
type JSONSchemaPropsOrArray = v1beta1.JSONSchemaPropsOrArray

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaPropsOrBool instead.
//
//go:fix inline
type JSONSchemaPropsOrBool = v1beta1.JSONSchemaPropsOrBool

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaPropsOrStringArray instead.
//
//go:fix inline
type JSONSchemaPropsOrStringArray = v1beta1.JSONSchemaPropsOrStringArray

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.JSONSchemaURL instead.
//
//go:fix inline
type JSONSchemaURL = v1beta1.JSONSchemaURL

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ResourceScope instead.
//
//go:fix inline
type ResourceScope = v1beta1.ResourceScope

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SelectableField instead.
//
//go:fix inline
type SelectableField = v1beta1.SelectableField

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ServiceReference instead.
//
//go:fix inline
type ServiceReference = v1beta1.ServiceReference

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ValidationRule instead.
//
//go:fix inline
type ValidationRule = v1beta1.ValidationRule

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ValidationRules instead.
//
//go:fix inline
type ValidationRules = v1beta1.ValidationRules

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.WebhookClientConfig instead.
//
//go:fix inline
type WebhookClientConfig = v1beta1.WebhookClientConfig

const (
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ClusterScoped instead.
	//
	//go:fix inline
	ClusterScoped = v1beta1.ClusterScoped
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConditionFalse instead.
	//
	//go:fix inline
	ConditionFalse = v1beta1.ConditionFalse
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConditionTrue instead.
	//
	//go:fix inline
	ConditionTrue = v1beta1.ConditionTrue
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ConditionUnknown instead.
	//
	//go:fix inline
	ConditionUnknown = v1beta1.ConditionUnknown
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.CustomResourceCleanupFinalizer instead.
	//
	//go:fix inline
	CustomResourceCleanupFinalizer = v1beta1.CustomResourceCleanupFinalizer
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.Established instead.
	//
	//go:fix inline
	Established = v1beta1.Established
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.FieldValueDuplicate instead.
	//
	//go:fix inline
	FieldValueDuplicate = v1beta1.FieldValueDuplicate
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.FieldValueForbidden instead.
	//
	//go:fix inline
	FieldValueForbidden = v1beta1.FieldValueForbidden
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.FieldValueInvalid instead.
	//
	//go:fix inline
	FieldValueInvalid = v1beta1.FieldValueInvalid
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.FieldValueRequired instead.
	//
	//go:fix inline
	FieldValueRequired = v1beta1.FieldValueRequired
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.GroupName instead.
	//
	//go:fix inline
	GroupName = v1beta1.GroupName
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.KubeAPIApprovedAnnotation instead.
	//
	//go:fix inline
	KubeAPIApprovedAnnotation = v1beta1.KubeAPIApprovedAnnotation
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.KubernetesAPIApprovalPolicyConformant instead.
	//
	//go:fix inline
	KubernetesAPIApprovalPolicyConformant = v1beta1.KubernetesAPIApprovalPolicyConformant
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.NamesAccepted instead.
	//
	//go:fix inline
	NamesAccepted = v1beta1.NamesAccepted
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.NamespaceScoped instead.
	//
	//go:fix inline
	NamespaceScoped = v1beta1.NamespaceScoped
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.NonStructuralSchema instead.
	//
	//go:fix inline
	NonStructuralSchema = v1beta1.NonStructuralSchema
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.NoneConverter instead.
	//
	//go:fix inline
	NoneConverter = v1beta1.NoneConverter
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.Terminating instead.
	//
	//go:fix inline
	Terminating = v1beta1.Terminating
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.WebhookConverter instead.
	//
	//go:fix inline
	WebhookConverter = v1beta1.WebhookConverter
)

var (
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ErrIntOverflowGenerated instead.
	ErrIntOverflowGenerated = v1beta1.ErrIntOverflowGenerated
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ErrInvalidLengthGenerated instead.
	ErrInvalidLengthGenerated = v1beta1.ErrInvalidLengthGenerated
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.ErrUnexpectedEndOfGroupGenerated instead.
	ErrUnexpectedEndOfGroupGenerated = v1beta1.ErrUnexpectedEndOfGroupGenerated
	// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SchemeGroupVersion instead.
	SchemeGroupVersion = v1beta1.SchemeGroupVersion
)

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.Kind instead.
//
//go:fix inline
func Kind(kind string) schema.GroupKind {
	return v1beta1.Kind(kind)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.RegisterDefaults instead.
//
//go:fix inline
func RegisterDefaults(scheme *runtime.Scheme) error {
	return v1beta1.RegisterDefaults(scheme)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.Resource instead.
//
//go:fix inline
func Resource(resource string) schema.GroupResource {
	return v1beta1.Resource(resource)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SetDefaults_CustomResourceDefinition instead.
//
//go:fix inline
func SetDefaults_CustomResourceDefinition(obj *v1beta1.CustomResourceDefinition) {
	v1beta1.SetDefaults_CustomResourceDefinition(obj)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SetDefaults_CustomResourceDefinitionSpec instead.
//
//go:fix inline
func SetDefaults_CustomResourceDefinitionSpec(obj *v1beta1.CustomResourceDefinitionSpec) {
	v1beta1.SetDefaults_CustomResourceDefinitionSpec(obj)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SetDefaults_ServiceReference instead.
//
//go:fix inline
func SetDefaults_ServiceReference(obj *v1beta1.ServiceReference) {
	v1beta1.SetDefaults_ServiceReference(obj)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SetObjectDefaults_CustomResourceDefinition instead.
//
//go:fix inline
func SetObjectDefaults_CustomResourceDefinition(in *v1beta1.CustomResourceDefinition) {
	v1beta1.SetObjectDefaults_CustomResourceDefinition(in)
}

// Deprecated: Use k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1.SetObjectDefaults_CustomResourceDefinitionList instead.
//
//go:fix inline
func SetObjectDefaults_CustomResourceDefinitionList(in *v1beta1.CustomResourceDefinitionList) {
	v1beta1.SetObjectDefaults_CustomResourceDefinitionList(in)
}
