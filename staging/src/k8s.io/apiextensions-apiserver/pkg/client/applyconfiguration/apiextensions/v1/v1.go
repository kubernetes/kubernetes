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

package v1

import (
	apiextensionsv1 "k8s.io/apiextensions/pkg/apis/apiextensions/v1"
	v1 "k8s.io/apiextensions/pkg/client/applyconfiguration/apiextensions/v1"
)

//go:fix inline
type CustomResourceColumnDefinitionApplyConfiguration = v1.CustomResourceColumnDefinitionApplyConfiguration

//go:fix inline
type CustomResourceConversionApplyConfiguration = v1.CustomResourceConversionApplyConfiguration

//go:fix inline
type CustomResourceDefinitionApplyConfiguration = v1.CustomResourceDefinitionApplyConfiguration

//go:fix inline
type CustomResourceDefinitionConditionApplyConfiguration = v1.CustomResourceDefinitionConditionApplyConfiguration

//go:fix inline
type CustomResourceDefinitionNamesApplyConfiguration = v1.CustomResourceDefinitionNamesApplyConfiguration

//go:fix inline
type CustomResourceDefinitionSpecApplyConfiguration = v1.CustomResourceDefinitionSpecApplyConfiguration

//go:fix inline
type CustomResourceDefinitionStatusApplyConfiguration = v1.CustomResourceDefinitionStatusApplyConfiguration

//go:fix inline
type CustomResourceDefinitionVersionApplyConfiguration = v1.CustomResourceDefinitionVersionApplyConfiguration

//go:fix inline
type CustomResourceSubresourceScaleApplyConfiguration = v1.CustomResourceSubresourceScaleApplyConfiguration

//go:fix inline
type CustomResourceSubresourcesApplyConfiguration = v1.CustomResourceSubresourcesApplyConfiguration

//go:fix inline
type CustomResourceValidationApplyConfiguration = v1.CustomResourceValidationApplyConfiguration

//go:fix inline
type ExternalDocumentationApplyConfiguration = v1.ExternalDocumentationApplyConfiguration

//go:fix inline
type JSONSchemaPropsApplyConfiguration = v1.JSONSchemaPropsApplyConfiguration

//go:fix inline
type SelectableFieldApplyConfiguration = v1.SelectableFieldApplyConfiguration

//go:fix inline
type ServiceReferenceApplyConfiguration = v1.ServiceReferenceApplyConfiguration

//go:fix inline
type ValidationRuleApplyConfiguration = v1.ValidationRuleApplyConfiguration

//go:fix inline
type WebhookClientConfigApplyConfiguration = v1.WebhookClientConfigApplyConfiguration

//go:fix inline
type WebhookConversionApplyConfiguration = v1.WebhookConversionApplyConfiguration

//go:fix inline
func CustomResourceColumnDefinition() *v1.CustomResourceColumnDefinitionApplyConfiguration {
	return v1.CustomResourceColumnDefinition()
}

//go:fix inline
func CustomResourceConversion() *v1.CustomResourceConversionApplyConfiguration {
	return v1.CustomResourceConversion()
}

//go:fix inline
func CustomResourceDefinition(name string) *v1.CustomResourceDefinitionApplyConfiguration {
	return v1.CustomResourceDefinition(name)
}

//go:fix inline
func CustomResourceDefinitionCondition() *v1.CustomResourceDefinitionConditionApplyConfiguration {
	return v1.CustomResourceDefinitionCondition()
}

//go:fix inline
func CustomResourceDefinitionNames() *v1.CustomResourceDefinitionNamesApplyConfiguration {
	return v1.CustomResourceDefinitionNames()
}

//go:fix inline
func CustomResourceDefinitionSpec() *v1.CustomResourceDefinitionSpecApplyConfiguration {
	return v1.CustomResourceDefinitionSpec()
}

//go:fix inline
func CustomResourceDefinitionStatus() *v1.CustomResourceDefinitionStatusApplyConfiguration {
	return v1.CustomResourceDefinitionStatus()
}

//go:fix inline
func CustomResourceDefinitionVersion() *v1.CustomResourceDefinitionVersionApplyConfiguration {
	return v1.CustomResourceDefinitionVersion()
}

//go:fix inline
func CustomResourceSubresourceScale() *v1.CustomResourceSubresourceScaleApplyConfiguration {
	return v1.CustomResourceSubresourceScale()
}

//go:fix inline
func CustomResourceSubresources() *v1.CustomResourceSubresourcesApplyConfiguration {
	return v1.CustomResourceSubresources()
}

//go:fix inline
func CustomResourceValidation() *v1.CustomResourceValidationApplyConfiguration {
	return v1.CustomResourceValidation()
}

//go:fix inline
func ExternalDocumentation() *v1.ExternalDocumentationApplyConfiguration {
	return v1.ExternalDocumentation()
}

//go:fix inline
func ExtractCustomResourceDefinition(customResourceDefinition *apiextensionsv1.CustomResourceDefinition, fieldManager string) (*v1.CustomResourceDefinitionApplyConfiguration, error) {
	return v1.ExtractCustomResourceDefinition(customResourceDefinition, fieldManager)
}

//go:fix inline
func ExtractCustomResourceDefinitionFrom(customResourceDefinition *apiextensionsv1.CustomResourceDefinition, fieldManager string, subresource string) (*v1.CustomResourceDefinitionApplyConfiguration, error) {
	return v1.ExtractCustomResourceDefinitionFrom(customResourceDefinition, fieldManager, subresource)
}

//go:fix inline
func ExtractCustomResourceDefinitionStatus(customResourceDefinition *apiextensionsv1.CustomResourceDefinition, fieldManager string) (*v1.CustomResourceDefinitionApplyConfiguration, error) {
	return v1.ExtractCustomResourceDefinitionStatus(customResourceDefinition, fieldManager)
}

//go:fix inline
func JSONSchemaProps() *v1.JSONSchemaPropsApplyConfiguration {
	return v1.JSONSchemaProps()
}

//go:fix inline
func SelectableField() *v1.SelectableFieldApplyConfiguration {
	return v1.SelectableField()
}

//go:fix inline
func ServiceReference() *v1.ServiceReferenceApplyConfiguration {
	return v1.ServiceReference()
}

//go:fix inline
func ValidationRule() *v1.ValidationRuleApplyConfiguration {
	return v1.ValidationRule()
}

//go:fix inline
func WebhookClientConfig() *v1.WebhookClientConfigApplyConfiguration {
	return v1.WebhookClientConfig()
}

//go:fix inline
func WebhookConversion() *v1.WebhookConversionApplyConfiguration {
	return v1.WebhookConversion()
}
