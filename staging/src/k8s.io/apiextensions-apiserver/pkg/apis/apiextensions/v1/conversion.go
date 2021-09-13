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

package v1

import (
	"bytes"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/util/json"
)

func Convert_apiextensions_JSONSchemaProps_To_v1_JSONSchemaProps(in *apiextensions.JSONSchemaProps, out *JSONSchemaProps, s conversion.Scope) error {
	if err := autoConvert_apiextensions_JSONSchemaProps_To_v1_JSONSchemaProps(in, out, s); err != nil {
		return err
	}
	if in.Default != nil && *(in.Default) == nil {
		out.Default = nil
	}
	if in.Example != nil && *(in.Example) == nil {
		out.Example = nil
	}
	return nil
}

var nullLiteral = []byte(`null`)

func Convert_apiextensions_JSON_To_v1_JSON(in *apiextensions.JSON, out *JSON, s conversion.Scope) error {
	raw, err := json.Marshal(*in)
	if err != nil {
		return err
	}
	if len(raw) == 0 || bytes.Equal(raw, nullLiteral) {
		// match JSON#UnmarshalJSON treatment of literal nulls
		out.Raw = nil
	} else {
		out.Raw = raw
	}
	return nil
}

func Convert_v1_JSON_To_apiextensions_JSON(in *JSON, out *apiextensions.JSON, s conversion.Scope) error {
	if in != nil {
		var i interface{}
		if len(in.Raw) > 0 && !bytes.Equal(in.Raw, nullLiteral) {
			if err := json.Unmarshal(in.Raw, &i); err != nil {
				return err
			}
		}
		*out = i
	} else {
		out = nil
	}
	return nil
}

func Convert_apiextensions_CustomResourceDefinitionSpec_To_v1_CustomResourceDefinitionSpec(in *apiextensions.CustomResourceDefinitionSpec, out *CustomResourceDefinitionSpec, s conversion.Scope) error {
	if err := autoConvert_apiextensions_CustomResourceDefinitionSpec_To_v1_CustomResourceDefinitionSpec(in, out, s); err != nil {
		return err
	}

	if len(out.Versions) == 0 && len(in.Version) > 0 {
		// no versions were specified, and a version name was specified
		out.Versions = []CustomResourceDefinitionVersion{{Name: in.Version, Served: true, Storage: true}}
	}

	// If spec.{subresources,validation,additionalPrinterColumns} exists, move to versions
	if in.Subresources != nil {
		subresources := &CustomResourceSubresources{}
		if err := Convert_apiextensions_CustomResourceSubresources_To_v1_CustomResourceSubresources(in.Subresources, subresources, s); err != nil {
			return err
		}
		for i := range out.Versions {
			out.Versions[i].Subresources = subresources
		}
	}
	if in.Validation != nil {
		schema := &CustomResourceValidation{}
		if err := Convert_apiextensions_CustomResourceValidation_To_v1_CustomResourceValidation(in.Validation, schema, s); err != nil {
			return err
		}
		for i := range out.Versions {
			out.Versions[i].Schema = schema
		}
	}
	if in.AdditionalPrinterColumns != nil {
		additionalPrinterColumns := make([]CustomResourceColumnDefinition, len(in.AdditionalPrinterColumns))
		for i := range in.AdditionalPrinterColumns {
			if err := Convert_apiextensions_CustomResourceColumnDefinition_To_v1_CustomResourceColumnDefinition(&in.AdditionalPrinterColumns[i], &additionalPrinterColumns[i], s); err != nil {
				return err
			}
		}
		for i := range out.Versions {
			out.Versions[i].AdditionalPrinterColumns = additionalPrinterColumns
		}
	}
	return nil
}

func Convert_v1_CustomResourceDefinitionSpec_To_apiextensions_CustomResourceDefinitionSpec(in *CustomResourceDefinitionSpec, out *apiextensions.CustomResourceDefinitionSpec, s conversion.Scope) error {
	if err := autoConvert_v1_CustomResourceDefinitionSpec_To_apiextensions_CustomResourceDefinitionSpec(in, out, s); err != nil {
		return err
	}

	if len(out.Versions) == 0 {
		return nil
	}

	// Copy versions[0] to version
	out.Version = out.Versions[0].Name

	// If versions[*].{subresources,schema,additionalPrinterColumns} are identical, move to spec
	subresources := out.Versions[0].Subresources
	subresourcesIdentical := true
	validation := out.Versions[0].Schema
	validationIdentical := true
	additionalPrinterColumns := out.Versions[0].AdditionalPrinterColumns
	additionalPrinterColumnsIdentical := true

	// Detect if per-version fields are identical
	for _, v := range out.Versions {
		if subresourcesIdentical && !apiequality.Semantic.DeepEqual(v.Subresources, subresources) {
			subresourcesIdentical = false
		}
		if validationIdentical && !apiequality.Semantic.DeepEqual(v.Schema, validation) {
			validationIdentical = false
		}
		if additionalPrinterColumnsIdentical && !apiequality.Semantic.DeepEqual(v.AdditionalPrinterColumns, additionalPrinterColumns) {
			additionalPrinterColumnsIdentical = false
		}
	}

	// If they are, set the top-level fields and clear the per-version fields
	if subresourcesIdentical {
		out.Subresources = subresources
	}
	if validationIdentical {
		out.Validation = validation
	}
	if additionalPrinterColumnsIdentical {
		out.AdditionalPrinterColumns = additionalPrinterColumns
	}
	for i := range out.Versions {
		if subresourcesIdentical {
			out.Versions[i].Subresources = nil
		}
		if validationIdentical {
			out.Versions[i].Schema = nil
		}
		if additionalPrinterColumnsIdentical {
			out.Versions[i].AdditionalPrinterColumns = nil
		}
	}

	return nil
}

func Convert_v1_CustomResourceConversion_To_apiextensions_CustomResourceConversion(in *CustomResourceConversion, out *apiextensions.CustomResourceConversion, s conversion.Scope) error {
	if err := autoConvert_v1_CustomResourceConversion_To_apiextensions_CustomResourceConversion(in, out, s); err != nil {
		return err
	}

	out.WebhookClientConfig = nil
	out.ConversionReviewVersions = nil
	if in.Webhook != nil {
		out.ConversionReviewVersions = in.Webhook.ConversionReviewVersions
		if in.Webhook.ClientConfig != nil {
			out.WebhookClientConfig = &apiextensions.WebhookClientConfig{}
			if err := Convert_v1_WebhookClientConfig_To_apiextensions_WebhookClientConfig(in.Webhook.ClientConfig, out.WebhookClientConfig, s); err != nil {
				return err
			}
		}
	}
	return nil
}

func Convert_apiextensions_CustomResourceConversion_To_v1_CustomResourceConversion(in *apiextensions.CustomResourceConversion, out *CustomResourceConversion, s conversion.Scope) error {
	if err := autoConvert_apiextensions_CustomResourceConversion_To_v1_CustomResourceConversion(in, out, s); err != nil {
		return err
	}

	out.Webhook = nil
	if in.WebhookClientConfig != nil || in.ConversionReviewVersions != nil {
		out.Webhook = &WebhookConversion{}
		out.Webhook.ConversionReviewVersions = in.ConversionReviewVersions
		if in.WebhookClientConfig != nil {
			out.Webhook.ClientConfig = &WebhookClientConfig{}
			if err := Convert_apiextensions_WebhookClientConfig_To_v1_WebhookClientConfig(in.WebhookClientConfig, out.Webhook.ClientConfig, s); err != nil {
				return err
			}
		}
	}
	return nil
}
