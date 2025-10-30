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
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	formatTagName = "k8s:format"
)

func init() {
	RegisterTagValidator(formatTagValidator{})
}

type formatTagValidator struct{}

func (formatTagValidator) Init(_ Config) {}

func (formatTagValidator) TagName() string {
	return formatTagName
}

var formatTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (formatTagValidator) ValidScopes() sets.Set[Scope] {
	return formatTagValidScopes
}

var (
	// Keep this list alphabetized.
	// TODO: uncomment the following when we've done the homework
	// to be sure it works the current state of IP manual-ratcheting
	// ipSloppyValidator         = types.Name{Package: libValidationPkg, Name: "IPSloppy"}
	extendedResourceNameValidator       = types.Name{Package: libValidationPkg, Name: "ExtendedResourceName"}
	labelKeyValidator                   = types.Name{Package: libValidationPkg, Name: "LabelKey"}
	labelValueValidator                 = types.Name{Package: libValidationPkg, Name: "LabelValue"}
	longNameCaselessValidator           = types.Name{Package: libValidationPkg, Name: "LongNameCaseless"}
	longNameValidator                   = types.Name{Package: libValidationPkg, Name: "LongName"}
	resourceFullyQualifiedNameValidator = types.Name{Package: libValidationPkg, Name: "ResourceFullyQualifiedName"}
	resourcePoolNameValidator           = types.Name{Package: libValidationPkg, Name: "ResourcePoolName"}
	shortNameValidator                  = types.Name{Package: libValidationPkg, Name: "ShortName"}
	uuidValidator                       = types.Name{Package: libValidationPkg, Name: "UUID"}
)

func (formatTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return Validations{}, fmt.Errorf("can only be used on string types (%s)", rootTypeString(context.Type, t))
	}

	var result Validations
	if formatFunction, err := getFormatValidationFunction(tag.Value); err != nil {
		return result, err
	} else {
		result.AddFunction(formatFunction)
	}
	return result, nil
}

func getFormatValidationFunction(format string) (FunctionGen, error) {
	// The naming convention for these formats follows the JSON schema style:
	// all lower-case, dashes between words. See
	// https://json-schema.org/draft/2020-12/json-schema-validation#name-defined-formats
	// for more examples.

	switch format {
	// Keep this sequence alphabetized.
	case "k8s-extended-resource-name":
		return Function(formatTagName, DefaultFlags, extendedResourceNameValidator), nil
	// TODO: uncomment the following when we've done the homework
	// to be sure it works the current state of IP manual-ratcheting
	/*
		case "k8s-ip":
			return Function(formatTagName, DefaultFlags, ipSloppyValidator), nil
	*/
	case "k8s-label-key":
		return Function(formatTagName, DefaultFlags, labelKeyValidator), nil
	case "k8s-label-value":
		return Function(formatTagName, DefaultFlags, labelValueValidator), nil
	case "k8s-long-name":
		return Function(formatTagName, DefaultFlags, longNameValidator), nil
	case "k8s-long-name-caseless":
		return Function(formatTagName, DefaultFlags, longNameCaselessValidator), nil
	case "k8s-resource-fully-qualified-name":
		return Function(formatTagName, DefaultFlags, resourceFullyQualifiedNameValidator), nil
	case "k8s-resource-pool-name":
		return Function(formatTagName, DefaultFlags, resourcePoolNameValidator), nil
	case "k8s-short-name":
		return Function(formatTagName, DefaultFlags, shortNameValidator), nil
	case "k8s-uuid":
		return Function(formatTagName, DefaultFlags, uuidValidator), nil
	}
	// TODO: Flesh out the list of validation functions

	return FunctionGen{}, fmt.Errorf("unsupported validation format %q", format)
}

func (ftv formatTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            ftv.TagName(),
		StabilityLevel: Beta,
		Scopes:         ftv.ValidScopes().UnsortedList(),
		Description:    "Indicates that a string field has a particular format.",
		Payloads: []TagPayloadDoc{{ // Keep this list alphabetized.
			Description: "k8s-extended-resource-name",
			Docs:        "This field holds a Kubernetes extended resource name. This is a domain-prefixed name that must not have a `kubernetes.io` or `requests.` prefix. When `requests.` is prepended, the result must be a valid label key, as used by quota.",
		}, {
			Description: "k8s-ip",
			Docs:        "This field holds an IPv4 or IPv6 address value. IPv4 octets may have leading zeros.",
		}, {
			Description: "k8s-label-key",
			Docs:        "This field holds a Kubernetes label key.",
		}, {
			Description: "k8s-label-value",
			Docs:        "This field holds a Kubernetes label value.",
		}, {
			Description: "k8s-long-name",
			Docs:        "This field holds a Kubernetes \"long name\", aka a \"DNS subdomain\" value.",
		}, {
			Description: "k8s-long-name-caseless",
			Docs:        "Deprecated: This field holds a case-insensitive Kubernetes \"long name\", aka a \"DNS subdomain\" value.",
		}, {
			Description: "k8s-resource-fully-qualified-name",
			Docs:        "This field holds a Kubernetes resource \"fully qualified name\" value. A fully qualified name must not be empty and must be composed of a prefix and a name, separated by a slash (e.g., \"prefix/name\"). The prefix must be a DNS subdomain, and the name part must be a C identifier with no more than 32 characters.",
		}, {
			Description: "k8s-resource-pool-name",
			Docs:        "This field holds value with one or more Kubernetes \"long name\" parts separated by `/` and no longer than 253 characters.",
		}, {
			Description: "k8s-short-name",
			Docs:        "This field holds a Kubernetes \"short name\", aka a \"DNS label\" value.",
		}, {
			Description: "k8s-uuid",
			Docs:        "This field holds a Kubernetes UUID, which conforms to RFC 4122.",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
}
