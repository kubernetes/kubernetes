package template

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/validation/field"
	//"github.com/openshift/origin/pkg/template/api/validation"
)

// templateStrategy implements behavior for Templates
type templateStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Template
// objects via the REST API.
var Strategy = templateStrategy{api.Scheme, api.SimpleNameGenerator}

// NamespaceScoped is true for templates.
func (templateStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (templateStrategy) PrepareForUpdate(obj, old runtime.Object) {}

// Canonicalize normalizes the object after validation.
func (templateStrategy) Canonicalize(obj runtime.Object) {
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (templateStrategy) PrepareForCreate(obj runtime.Object) {
}

// Validate validates a new template.
func (templateStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	return field.ErrorList{} //validation.ValidateTemplate(obj.(*extensions.Template))
}

// AllowCreateOnUpdate is false for templates.
func (templateStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (templateStrategy) AllowUnconditionalUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (templateStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return field.ErrorList{} // validation.ValidateTemplateUpdate(obj.(*extensions.Template), old.(*extensions.Template))
}

// templateToSelectableFields returns a label set that represents the object
// changes to the returned keys require registering conversions for existing versions using Scheme.AddFieldLabelConversionFunc
func templateToSelectableFields(template *extensions.Template) fields.Set {
	return fields.Set{
		"metadata.name": template.Name,
	}
}

// Matcher returns a generic matcher for a given label and field selector.
func Matcher(label labels.Selector, field fields.Selector) generic.Matcher {
	return generic.MatcherFunc(func(obj runtime.Object) (bool, error) {
		o, ok := obj.(*extensions.Template)
		if !ok {
			return false, fmt.Errorf("not a pod")
		}
		return label.Matches(labels.Set(o.Labels)) && field.Matches(templateToSelectableFields(o)), nil
	})
}
