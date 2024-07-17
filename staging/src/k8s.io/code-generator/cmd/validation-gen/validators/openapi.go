/*
Copyright 2024 The Kubernetes Authors.

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

package validators

import (
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/types"
	"k8s.io/kube-openapi/pkg/generators"
)

func init() {
	AddToRegistry(InitOpenAPIDeclarativeValidator)
}

func InitOpenAPIDeclarativeValidator(c *generator.Context) DeclarativeValidator {
	return &openAPIDeclarativeValidator{}
}

type openAPIDeclarativeValidator struct{}

const (
	markerPrefix      = "+k8s:validation:"
	utilValidationPkg = "k8s.io/apimachinery/pkg/util/validation"

	formatTagName    = "k8s:validation:format"
	maxLengthTagName = "k8s:validation:maxLength"
)

var (
	isValidIPValidator = types.Name{Package: utilValidationPkg, Name: "IsValidIP"}
	maxLengthValidator = types.Name{Package: utilValidationPkg, Name: "ValidateMaxLength"}
)

func (openAPIDeclarativeValidator) ExtractValidations(t *types.Type, comments []string) ([]FunctionGen, error) {
	var v []FunctionGen

	// Leverage the kube-openapi parser for 'k8s:validation:' validations.
	schema, err := generators.ParseCommentTags(t, comments, markerPrefix)
	if err != nil {
		return nil, err
	}
	if schema.MaxLength != nil {
		v = append(v, Function(formatTagName, maxLengthValidator, *schema.MaxLength))
	}
	if len(schema.Format) > 0 {
		formatFunction := FormatValidationFunction(schema.Format)
		if formatFunction != nil {
			v = append(v, formatFunction)
		}
	}

	return v, nil
}

func FormatValidationFunction(format string) FunctionGen {
	if format == "ip" {
		return Function(maxLengthTagName, isValidIPValidator)
	}
	// TODO: Flesh out the list of validation functions

	return nil // TODO: ignore unsupported formats?
}
