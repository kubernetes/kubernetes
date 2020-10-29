package customresourcevalidation

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
)

type ObjectValidator interface {
	ValidateCreate(obj runtime.Object) field.ErrorList
	ValidateUpdate(obj runtime.Object, oldObj runtime.Object) field.ErrorList
	ValidateStatusUpdate(obj runtime.Object, oldObj runtime.Object) field.ErrorList
}

// ValidateCustomResource is an implementation of admission.Interface.
// It looks at all new pods and overrides each container's image pull policy to Always.
type validateCustomResource struct {
	*admission.Handler

	resources  map[schema.GroupResource]bool
	validators map[schema.GroupVersionKind]ObjectValidator
}

func NewValidator(resources map[schema.GroupResource]bool, validators map[schema.GroupVersionKind]ObjectValidator) (admission.ValidationInterface, error) {
	return &validateCustomResource{
		Handler:    admission.NewHandler(admission.Create, admission.Update),
		resources:  resources,
		validators: validators,
	}, nil
}

var _ admission.ValidationInterface = &validateCustomResource{}

// Validate is an admission function that will validate a CRD in config.openshift.io.  uncastAttributes are attributes
// that are of type unstructured.
func (a *validateCustomResource) Validate(ctx context.Context, uncastAttributes admission.Attributes, _ admission.ObjectInterfaces) error {
	attributes := &unstructuredUnpackingAttributes{Attributes: uncastAttributes}
	if a.shouldIgnore(attributes) {
		return nil
	}
	validator, ok := a.validators[attributes.GetKind()]
	if !ok {
		return admission.NewForbidden(attributes, fmt.Errorf("unhandled kind: %v", attributes.GetKind()))
	}

	switch attributes.GetOperation() {
	case admission.Create:
		// creating subresources isn't something we understand, but we can be pretty sure we don't need to validate it
		if len(attributes.GetSubresource()) > 0 {
			return nil
		}
		errors := validator.ValidateCreate(attributes.GetObject())
		if len(errors) == 0 {
			return nil
		}
		return apierrors.NewInvalid(attributes.GetKind().GroupKind(), attributes.GetName(), errors)

	case admission.Update:
		switch attributes.GetSubresource() {
		case "":
			errors := validator.ValidateUpdate(attributes.GetObject(), attributes.GetOldObject())
			if len(errors) == 0 {
				return nil
			}
			return apierrors.NewInvalid(attributes.GetKind().GroupKind(), attributes.GetName(), errors)

		case "status":
			errors := validator.ValidateStatusUpdate(attributes.GetObject(), attributes.GetOldObject())
			if len(errors) == 0 {
				return nil
			}
			return apierrors.NewInvalid(attributes.GetKind().GroupKind(), attributes.GetName(), errors)

		default:
			return admission.NewForbidden(attributes, fmt.Errorf("unhandled subresource: %v", attributes.GetSubresource()))
		}

	default:
		return admission.NewForbidden(attributes, fmt.Errorf("unhandled operation: %v", attributes.GetOperation()))
	}
}

func (a *validateCustomResource) shouldIgnore(attributes admission.Attributes) bool {
	if !a.resources[attributes.GetResource().GroupResource()] {
		return true
	}
	// if a subresource is specified and it isn't status, skip it
	if len(attributes.GetSubresource()) > 0 && attributes.GetSubresource() != "status" {
		return true
	}

	return false
}
