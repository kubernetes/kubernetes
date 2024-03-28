package library

import (
	"net/url"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/util/validation"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

// Format provides a CEL library exposing common named Kubernetes string
// validations. Can be used in CRD ValidationRules messageExpression. Example:
//
//			     rule: format.dns1123label.validate(object.metadata.name).hasValue()
//	messageExpression: format.dns1123label.validate(object.metadata.name).value().join("\n")
//
// format.named(name: string) -> ?Format
//
//		Returns the Format with the given name, if it exists. Otherwise, optional.none
//		Allowed names are:
//		 - `dns1123Label`
//		 - `dns1123Subdomain`
//		 - `dns1035Label`
//		 - `qualifiedName`
//		 - `dns1123LabelPrefix`
//		 - `dns1123SubdomainPrefix`
//		 - `dns1035LabelPrefix`
//		 - `labelValue`
//	     - `uri`
//	     - `uuid`
//	     - `byte`
//	     - `date`
//	     - `datetime`
//
// format.<formatName> -> Format
//
//		Constant values for the named formats are also available:
//			E.g., format.dns1123Label
//				  format.dns1123Subdomain
//				  etc...
//
//		Examples:
//			format.dns1123Label.validate("my-label-name")
//			format.dns1123Subdomain.validate("apiextensions.k8s.io")
//	     	format.dns1035Label.validate("my-label-name")
//			format.qualifiedName.validate("apiextensions.k8s.io/v1beta1")
//			format.dns1123LabelPrefix.validate("my-label-prefix-")
//			format.dns1123SubdomainPrefix.validate("mysubdomain.prefix.-")
//			format.dns1035LabelPrefix.validate("my-label-prefix-")
//			format.uri.validate("http://example.com")
//			format.uuid.validate("123e4567-e89b-12d3-a456-426614174000")
//			format.byte.validate("aGVsbG8=")
//			format.date.validate("2021-01-01")
//			format.datetime.validate("2021-01-01T00:00:00Z")
//
// <Format>.validate(str: string) -> ?list<string>
//
//	Validates the given string against the given format. Returns optional.none
//	if the string is valid, otherwise a list of validation error strings.
//
// <Format>.validatePrefix(str: string) -> ?list<string>
//
//	Same as validate, but assumes the input string will be used as a prefix for
//	a kubernetes generated name.
func Format() cel.EnvOption {
	return cel.Lib(formatLib)
}

var formatLib = &format{}

type format struct{}

func (*format) LibraryName() string {
	return "format"
}

func (*format) CompileOptions() []cel.EnvOption {
	options := make([]cel.EnvOption, 0, len(formatLibraryDecls))
	for name, overloads := range formatLibraryDecls {
		options = append(options, cel.Function(name, overloads...))
	}
	for name, constantValue := range ConstantFormats {
		options = append(options, cel.Constant("format."+name, apiservercel.FormatType, constantValue))
	}
	return options
}

func (*format) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

var ConstantFormats map[string]*apiservercel.Format = map[string]*apiservercel.Format{
	"dns1123Label": {
		Name:         "DNS1123Label",
		ValidateFunc: func(s string) []string { return apimachineryvalidation.NameIsDNSLabel(s, false) },
	},
	"dns1123Subdomain": {
		Name:         "DNS1123Subdomain",
		ValidateFunc: func(s string) []string { return apimachineryvalidation.NameIsDNSSubdomain(s, false) },
	},
	"dns1035Label": {
		Name:         "DNS1035Label",
		ValidateFunc: func(s string) []string { return apimachineryvalidation.NameIsDNS1035Label(s, false) },
	},
	"qualifiedName": {
		Name:         "QualifiedName",
		ValidateFunc: validation.IsQualifiedName,
	},

	"dns1123LabelPrefix": {
		Name:         "DNS1123LabelPrefix",
		ValidateFunc: func(s string) []string { return apimachineryvalidation.NameIsDNSLabel(s, true) },
	},
	"dns1123SubdomainPrefix": {
		Name:         "DNS1123SubdomainPrefix",
		ValidateFunc: func(s string) []string { return apimachineryvalidation.NameIsDNSSubdomain(s, true) },
	},
	"dns1035LabelPrefix": {
		Name:         "DNS1035LabelPrefix",
		ValidateFunc: func(s string) []string { return apimachineryvalidation.NameIsDNS1035Label(s, true) },
	},
	"labelValue": {
		Name:         "LabelValue",
		ValidateFunc: validation.IsValidLabelValue,
	},

	// CRD formats
	// Implementations sourced from strfmt, which kube-openapi uses as its
	// format library. There are other CRD formats supported, but they are
	// covered by other portions of the CEL library (like IP/CIDR), or their
	// use is discouraged (like bsonobjectid, email, etc)
	"uri": {
		Name: "URI",
		ValidateFunc: func(s string) []string {
			// Directly call ParseRequestURI since we can get a better error message
			_, err := url.ParseRequestURI(s)
			if err != nil {
				return []string{err.Error()}
			}
			return nil
		},
	},
	"uuid": {
		Name: "uuid",
		ValidateFunc: func(s string) []string {
			if !strfmt.Default.Validates("uuid", s) {
				return []string{"does not match the UUID format"}
			}
			return nil
		},
	},
	"byte": {
		Name: "byte",
		ValidateFunc: func(s string) []string {
			if !strfmt.Default.Validates("byte", s) {
				return []string{"invalid base64"}
			}
			return nil
		},
	},
	"date": {
		Name: "date",
		ValidateFunc: func(s string) []string {
			if !strfmt.Default.Validates("date", s) {
				return []string{"invalid date"}
			}
			return nil
		},
	},
	"datetime": {
		Name: "datetime",
		ValidateFunc: func(s string) []string {
			if !strfmt.Default.Validates("datetime", s) {
				return []string{"invalid datetime"}
			}
			return nil
		},
	},
}

var formatLibraryDecls = map[string][]cel.FunctionOpt{
	"validate": {
		cel.MemberOverload("format-validate", []*cel.Type{apiservercel.FormatType, cel.StringType}, cel.OptionalType(cel.ListType(cel.StringType)), cel.BinaryBinding(formatValidate)),
	},
	"format.named": {
		cel.Overload("format-named", []*cel.Type{cel.StringType}, cel.OptionalType(apiservercel.FormatType), cel.UnaryBinding(func(name ref.Val) ref.Val {
			nameString, ok := name.Value().(string)
			if !ok {
				return types.MaybeNoSuchOverloadErr(name)
			}

			f, ok := ConstantFormats[nameString]
			if !ok {
				return types.OptionalNone
			}
			return types.OptionalOf(f)
		})),
	},
}

func formatValidate(arg1, arg2 ref.Val) ref.Val {
	f, ok := arg1.Value().(*apiservercel.Format)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg1)
	}

	str, ok := arg2.Value().(string)
	if !ok {
		return types.MaybeNoSuchOverloadErr(arg2)
	}

	res := f.ValidateFunc(str)
	if len(res) == 0 {
		return types.OptionalNone
	}
	return types.OptionalOf(types.NewStringList(types.DefaultTypeAdapter, res))
}
