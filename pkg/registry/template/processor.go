/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package template

import (
	"fmt"
	"strings"

	"encoding/json"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// TemplateProcessor implements parameter substitution logic for templates
type TemplateProcessor interface {
	Process(ctx api.Context, params *extensions.TemplateParameters) (*api.List, error)
}

type templateProcessor struct {
	Registry
}

func NewTemplateProcessor(registry Registry) TemplateProcessor {
	return &templateProcessor{registry}
}

var _ TemplateProcessor = &templateProcessor{}

func (tp *templateProcessor) Process(ctx api.Context, params *extensions.TemplateParameters) (*api.List, error) {
	templateErrors := field.ErrorList{}
	result := &api.List{}

	// Lookup the Template
	template, err := tp.GetTemplate(ctx, params.Name)
	if err != nil {
		return nil, errors.NewBadRequest(err.Error())
	}

	// Validate the TemplateParameters are good for the Template
	if err := ValidateProcessTemplate(params, template); err != nil {
		return nil, errors.NewBadRequest(err.Error())
	}

	// Get an object to replace the string values
	replacer := GetParameterReplacer(params, template)

	// Perform the Substitutions on each Object in the Template
	for i, obj := range template.Spec.Objects {
		path := field.NewPath("spec", "objects").Index(i)
		finishedObj, err := SubstituteParams(path, obj, replacer)
		if err != nil {
			templateErrors = append(templateErrors, err)
			continue
		}

		result.Items = append(result.Items, finishedObj)
	}
	if len(templateErrors) > 0 {
		return nil, errors.NewInvalid(params.GroupVersionKind().GroupKind(), params.Name, templateErrors)
	}

	return result, nil
}

// GetParameterReplacer Returns an object to perform string substitutions
func GetParameterReplacer(params *extensions.TemplateParameters, template *extensions.Template) *strings.Replacer {
	replaceWith := []string{}
	substitutionMap := map[string]extensions.Parameter{}
	for _, p := range template.Spec.Parameters {
		substitutionMap[p.Name] = p
	}

	// Copy the TemplateParameters values into the Template.Parameters structures.
	// This will override default values when specified.
	for name, value := range params.ParameterValues {
		// Doesn't allow updating this field in the struct directly from the map
		// so we have to copy it out, update, and then re-insert
		p := substitutionMap[name]
		p.Value = value
		substitutionMap[name] = p
	}

	// Create the Replacer substitution mappings for the Parameters.
	for k, v := range substitutionMap {
		// TODO: Figure out if we need to support $((NAME)) syntax since
		// we are doing this server side and the quoting has been stripped
		replaceWith = append(replaceWith, fmt.Sprintf("$(%s)", k), v.Value)
	}
	return strings.NewReplacer(replaceWith...)
}

// SubstituteParams replaces the TemplateParameters specified in obj with their expansion values.
func SubstituteParams(path *field.Path, obj runtime.RawExtension, replacer *strings.Replacer) (runtime.Object, *field.Error) {
	unstructObj, err := DecodeRawToUnstructured(path, obj)
	if err != nil {
		return nil, err
	}

	// Do template Parameter substitutions
	VisitObjectStrings(unstructObj.Object, func(in string) string {
		return replacer.Replace(in)
	})

	// Encode the runtime.Unstructured object into a runtime.Object
	return EncodeUnstructuredToObject(path, unstructObj)
}

// DecodeRawToUnstructured decodes obj into a runtime.Unstructured
func DecodeRawToUnstructured(path *field.Path, obj runtime.RawExtension) (*runtime.Unstructured, *field.Error) {
	uObj, err := runtime.Decode(runtime.UnstructuredJSONScheme, obj.Raw)
	if err != nil {
		return nil, field.Invalid(path, obj, fmt.Sprintf(
			"unable to handle object, failed to Decode bytes %s: %v", obj.Raw, err))
	}
	unstructObj, ok := uObj.(*runtime.Unstructured)
	if !ok {
		return nil, field.Invalid(path, obj, fmt.Sprintf(
			"unable to handle object, Decoded bytes expected type runtime.Unstructured, but was %T: %v", uObj, err))
	}
	return unstructObj, nil
}

// EncodeUnstructuredToObject encodes obj into a runtime.Object
func EncodeUnstructuredToObject(path *field.Path, obj *runtime.Unstructured) (runtime.Object, *field.Error) {
	byteObj, err := json.Marshal(obj.Object)
	if err != nil {
		return nil, field.Invalid(path, obj, fmt.Sprintf(
			"unable to handle object, failed to Marshall map to Json %v: %v", obj.Object, err))
	}
	runtimeObj, err := runtime.Decode(api.Codecs.UniversalDeserializer(), byteObj)
	if err != nil {
		return nil, field.Invalid(path, obj,
			fmt.Sprintf("unable to handle object, failed to deserialize Json to an Object %s: %v", byteObj, err))
	}
	return runtimeObj, nil
}

// ValidateProcessTemplate validates that templateParameters can be used to expand template
func ValidateProcessTemplate(templateParameters *extensions.TemplateParameters, template *extensions.Template) error {
	allErrs := []error{}

	// Index the Template.Parameters by name
	vals := map[string]extensions.Parameter{}
	for _, p := range template.Spec.Parameters {
		vals[p.Name] = p
	}

	// Compare the provided TemplateParameters to the Template.Parameters to
	// make sure they are acceptable.
	for name, value := range templateParameters.ParameterValues {
		// Verify all provided parameters are specified in the original Template Parameter list
		if _, ok := vals[name]; !ok {
			allErrs = append(allErrs, fmt.Errorf("Parameter %s not defined in the Template.", name))
			continue
		}
		// Doesn't allow updating this field in the struct directly from the map
		// so we have to copy it out, update, and then re-insert
		p := vals[name]
		p.Value = value
		vals[name] = p

		// Verify the new TemplateParameter value matches the Template.Parameter type
		if err := validation.ValidateTemplateParamType(p); err != nil {
			allErrs = append(allErrs, err)
			continue
		}
	}

	// Validate the merged Template.Parameters
	for k, v := range vals {
		// Verify all Required Parameters have a Value specified
		if v.Required && len(v.Value) <= 0 {
			allErrs = append(allErrs, fmt.Errorf("Parameter %s has no Value but is required.", k))
		}
	}
	return utilerrors.NewAggregate(allErrs)
}
