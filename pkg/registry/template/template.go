package template

import (
	"fmt"
	"regexp"
	"strings"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"

	"github.com/golang/glog"
	// TODO: Fix this import
	"k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/pkg/util/stringreplace"
)

var parameterExp = regexp.MustCompile(`\$\{([a-zA-Z0-9\_]+)\}`)

// Processor process the Template into the List with substituted parameters
type Processor struct{}

// Process transforms Template object into List object.
func (p *Processor) Process(template *extensions.Template) field.ErrorList {
	templateErrors := field.ErrorList{}

	itemPath := field.NewPath("item")
	for i, item := range template.Objects {
		idxPath := itemPath.Index(i)
		if obj, ok := item.(*runtime.Unknown); ok {
			// TODO: use runtime.DecodeList when it returns ValidationErrorList
			decodedObj, err := runtime.Decode(runtime.UnstructuredJSONScheme, obj.Raw)
			if err != nil {
				templateErrors = append(templateErrors, field.Invalid(idxPath.Child("objects"), obj, fmt.Sprintf("unable to handle object: %v", err)))
				continue
			}
			item = decodedObj
		}
		newItem, err := p.SubstituteParameters(template.Parameters, item)
		if err != nil {
			templateErrors = append(templateErrors, field.Invalid(idxPath.Child("parameters"), template.Parameters, err.Error()))
		}
		// If an object definition's metadata includes a namespace field, the field will be stripped out of
		// the definition during template instantiation.  This is necessary because all objects created during
		// instantiation are placed into the target namespace, so it would be invalid for the object to declare
		//a different namespace.
		stripNamespace(newItem)
		if err := labels.AddObjectLabels(newItem, template.ObjectLabels); err != nil {
			templateErrors = append(templateErrors, field.Invalid(idxPath.Child("labels"), err, "label could not be applied"))
		}
		template.Objects[i] = newItem
	}

	return templateErrors
}

func stripNamespace(obj runtime.Object) {
	// Remove namespace from the item
	if itemMeta, err := meta.Accessor(obj); err == nil {
		itemMeta.SetNamespace("")
		return
	}
	// TODO: allow meta.Accessor to handle runtime.Unstructured
	if unstruct, ok := obj.(*runtime.Unstructured); ok && unstruct.Object != nil {
		if obj, ok := unstruct.Object["metadata"]; ok {
			if m, ok := obj.(map[string]interface{}); ok {
				if _, ok := m["namespace"]; ok {
					m["namespace"] = ""
				}
			}
			return
		}
		if _, ok := unstruct.Object["namespace"]; ok {
			unstruct.Object["namespace"] = ""
			return
		}
	}
}

// AddParameter adds new custom parameter to the Template. It overrides
// the existing parameter, if already defined.
func AddParameter(t *extensions.Template, param extensions.Parameter) {
	if existing := GetParameterByName(t, param.Name); existing != nil {
		*existing = param
	} else {
		t.Parameters = append(t.Parameters, param)
	}
}

// GetParameterByName searches for a Parameter in the Template
// based on its name.
func GetParameterByName(t *extensions.Template, name string) *extensions.Parameter {
	for i, param := range t.Parameters {
		if param.Name == name {
			return &(t.Parameters[i])
		}
	}
	return nil
}

// SubstituteParameters loops over all values defined in structured
// and unstructured types that are children of item.
//
// Example of Parameter expression:
//   - ${PARAMETER_NAME}
//
func (p *Processor) SubstituteParameters(params []extensions.Parameter, item runtime.Object) (runtime.Object, error) {
	// Make searching for given parameter name/value more effective
	paramMap := make(map[string]string, len(params))
	for _, param := range params {
		paramMap[param.Name] = param.Value
	}
	glog.Infof("List of parameters %+v", paramMap)

	stringreplace.VisitObjectStrings(item, func(in string) string {
		glog.Infof("Visitor for %+v: %+v", in, item)
		for _, match := range parameterExp.FindAllStringSubmatch(in, -1) {
			if len(match) > 1 {
				if paramValue, found := paramMap[match[1]]; found {
					glog.Infof("Visitor replacing %v -> %v", match[0], paramValue)
					in = strings.Replace(in, match[0], paramValue, 1)
				}
			}
		}
		return in
	})

	return item, nil
}
