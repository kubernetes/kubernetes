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
	apiextensionsv1beta1 "k8s.io/apiextensions/pkg/apis/apiextensions/v1beta1"
	v1beta1 "k8s.io/apiextensions/pkg/client/applyconfiguration/apiextensions/v1beta1"
)

//go:fix inline
type CustomResourceColumnDefinitionApplyConfiguration = v1beta1.CustomResourceColumnDefinitionApplyConfiguration

//go:fix inline
type CustomResourceConversionApplyConfiguration = v1beta1.CustomResourceConversionApplyConfiguration

//go:fix inline
type CustomResourceDefinitionApplyConfiguration = v1beta1.CustomResourceDefinitionApplyConfiguration

//go:fix inline
type CustomResourceDefinitionConditionApplyConfiguration = v1beta1.CustomResourceDefinitionConditionApplyConfiguration

//go:fix inline
type CustomResourceDefinitionNamesApplyConfiguration = v1beta1.CustomResourceDefinitionNamesApplyConfiguration

//go:fix inline
type CustomResourceDefinitionSpecApplyConfiguration = v1beta1.CustomResourceDefinitionSpecApplyConfiguration

//go:fix inline
type CustomResourceDefinitionStatusApplyConfiguration = v1beta1.CustomResourceDefinitionStatusApplyConfiguration

//go:fix inline
type CustomResourceDefinitionVersionApplyConfiguration = v1beta1.CustomResourceDefinitionVersionApplyConfiguration

//go:fix inline
type CustomResourceSubresourceScaleApplyConfiguration = v1beta1.CustomResourceSubresourceScaleApplyConfiguration

//go:fix inline
type CustomResourceSubresourcesApplyConfiguration = v1beta1.CustomResourceSubresourcesApplyConfiguration

//go:fix inline
type CustomResourceValidationApplyConfiguration = v1beta1.CustomResourceValidationApplyConfiguration

//go:fix inline
type ExternalDocumentationApplyConfiguration = v1beta1.ExternalDocumentationApplyConfiguration

//go:fix inline
type JSONSchemaPropsApplyConfiguration = v1beta1.JSONSchemaPropsApplyConfiguration

//go:fix inline
type SelectableFieldApplyConfiguration = v1beta1.SelectableFieldApplyConfiguration

//go:fix inline
type ServiceReferenceApplyConfiguration = v1beta1.ServiceReferenceApplyConfiguration

//go:fix inline
type ValidationRuleApplyConfiguration = v1beta1.ValidationRuleApplyConfiguration

//go:fix inline
type WebhookClientConfigApplyConfiguration = v1beta1.WebhookClientConfigApplyConfiguration

//go:fix inline
func CustomResourceColumnDefinition() *v1beta1.CustomResourceColumnDefinitionApplyConfiguration {
	return v1beta1.CustomResourceColumnDefinition()
}

//go:fix inline
func CustomResourceConversion() *v1beta1.CustomResourceConversionApplyConfiguration {
	return v1beta1.CustomResourceConversion()
}

//go:fix inline
func CustomResourceDefinition(name string) *v1beta1.CustomResourceDefinitionApplyConfiguration {
	return v1beta1.CustomResourceDefinition(name)
}

//go:fix inline
func CustomResourceDefinitionCondition() *v1beta1.CustomResourceDefinitionConditionApplyConfiguration {
	return v1beta1.CustomResourceDefinitionCondition()
}

//go:fix inline
func CustomResourceDefinitionNames() *v1beta1.CustomResourceDefinitionNamesApplyConfiguration {
	return v1beta1.CustomResourceDefinitionNames()
}

//go:fix inline
func CustomResourceDefinitionSpec() *v1beta1.CustomResourceDefinitionSpecApplyConfiguration {
	return v1beta1.CustomResourceDefinitionSpec()
}

//go:fix inline
func CustomResourceDefinitionStatus() *v1beta1.CustomResourceDefinitionStatusApplyConfiguration {
	return v1beta1.CustomResourceDefinitionStatus()
}

//go:fix inline
func CustomResourceDefinitionVersion() *v1beta1.CustomResourceDefinitionVersionApplyConfiguration {
	return v1beta1.CustomResourceDefinitionVersion()
}

//go:fix inline
func CustomResourceSubresourceScale() *v1beta1.CustomResourceSubresourceScaleApplyConfiguration {
	return v1beta1.CustomResourceSubresourceScale()
}

//go:fix inline
func CustomResourceSubresources() *v1beta1.CustomResourceSubresourcesApplyConfiguration {
	return v1beta1.CustomResourceSubresources()
}

//go:fix inline
func CustomResourceValidation() *v1beta1.CustomResourceValidationApplyConfiguration {
	return v1beta1.CustomResourceValidation()
}

//go:fix inline
func ExternalDocumentation() *v1beta1.ExternalDocumentationApplyConfiguration {
	return v1beta1.ExternalDocumentation()
}

//go:fix inline
func ExtractCustomResourceDefinition(customResourceDefinition *apiextensionsv1beta1.CustomResourceDefinition, fieldManager string) (*v1beta1.CustomResourceDefinitionApplyConfiguration, error) {
	return v1beta1.ExtractCustomResourceDefinition(customResourceDefinition, fieldManager)
}

//go:fix inline
func ExtractCustomResourceDefinitionFrom(customResourceDefinition *apiextensionsv1beta1.CustomResourceDefinition, fieldManager string, subresource string) (*v1beta1.CustomResourceDefinitionApplyConfiguration, error) {
	return v1beta1.ExtractCustomResourceDefinitionFrom(customResourceDefinition, fieldManager, subresource)
}

//go:fix inline
func ExtractCustomResourceDefinitionStatus(customResourceDefinition *apiextensionsv1beta1.CustomResourceDefinition, fieldManager string) (*v1beta1.CustomResourceDefinitionApplyConfiguration, error) {
	return v1beta1.ExtractCustomResourceDefinitionStatus(customResourceDefinition, fieldManager)
}

//go:fix inline
func JSONSchemaProps() *v1beta1.JSONSchemaPropsApplyConfiguration {
	return v1beta1.JSONSchemaProps()
}

//go:fix inline
func SelectableField() *v1beta1.SelectableFieldApplyConfiguration {
	return v1beta1.SelectableField()
}

//go:fix inline
func ServiceReference() *v1beta1.ServiceReferenceApplyConfiguration {
	return v1beta1.ServiceReference()
}

//go:fix inline
func ValidationRule() *v1beta1.ValidationRuleApplyConfiguration {
	return v1beta1.ValidationRule()
}

//go:fix inline
func WebhookClientConfig() *v1beta1.WebhookClientConfigApplyConfiguration {
	return v1beta1.WebhookClientConfig()
}
