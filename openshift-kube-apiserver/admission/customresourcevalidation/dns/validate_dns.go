package dns

import (
	"fmt"
	"io"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/validation"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"

	operatorv1 "github.com/openshift/api/operator/v1"
	crvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "operator.openshift.io/ValidateDNS"

// Register registers the DNS validation plugin.
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return crvalidation.NewValidator(
			map[schema.GroupResource]bool{
				operatorv1.GroupVersion.WithResource("dnses").GroupResource(): true,
			},
			map[schema.GroupVersionKind]crvalidation.ObjectValidator{
				operatorv1.GroupVersion.WithKind("DNS"): dnsV1{},
			})
	})
}

// toDNSV1 converts a runtime object to a versioned DNS.
func toDNSV1(uncastObj runtime.Object) (*operatorv1.DNS, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	obj, ok := uncastObj.(*operatorv1.DNS)
	if !ok {
		return nil, field.ErrorList{
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"DNS"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"operator.openshift.io/v1"}),
		}
	}

	return obj, nil
}

// dnsV1 is runtime object that is validated as a versioned DNS.
type dnsV1 struct{}

// ValidateCreate validates a DNS that is being created.
func (dnsV1) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	obj, errs := toDNSV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	errs = append(errs, validateDNSSpecCreate(obj.Spec)...)

	return errs
}

// ValidateUpdate validates a DNS that is being updated.
func (dnsV1) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toDNSV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toDNSV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, validateDNSSpecUpdate(obj.Spec, oldObj.Spec)...)

	return errs
}

// ValidateStatusUpdate validates a DNS status that is being updated.
func (dnsV1) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toDNSV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toDNSV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}

// validateDNSSpecCreate validates the spec of a DNS that is being created.
func validateDNSSpecCreate(spec operatorv1.DNSSpec) field.ErrorList {
	return validateDNSSpec(spec)
}

// validateDNSSpecUpdate validates the spec of a DNS that is being updated.
func validateDNSSpecUpdate(newspec, oldspec operatorv1.DNSSpec) field.ErrorList {
	return validateDNSSpec(newspec)
}

// validateDNSSpec validates the spec of a DNS.
func validateDNSSpec(spec operatorv1.DNSSpec) field.ErrorList {
	var errs field.ErrorList
	specField := field.NewPath("spec")
	errs = append(errs, validateDNSNodePlacement(spec.NodePlacement, specField.Child("nodePlacement"))...)
	return errs
}

// validateDNSSpec validates the spec.nodePlacement field of a DNS.
func validateDNSNodePlacement(nodePlacement operatorv1.DNSNodePlacement, fldPath *field.Path) field.ErrorList {
	var errs field.ErrorList
	if len(nodePlacement.NodeSelector) != 0 {
		errs = append(errs, unversionedvalidation.ValidateLabels(nodePlacement.NodeSelector, fldPath.Child("nodeSelector"))...)
	}
	if len(nodePlacement.Tolerations) != 0 {
		errs = append(errs, validateTolerations(nodePlacement.Tolerations, fldPath.Child("tolerations"))...)
	}
	return errs
}

// validateTolerations validates a slice of corev1.Toleration.
func validateTolerations(versionedTolerations []corev1.Toleration, fldPath *field.Path) field.ErrorList {
	allErrors := field.ErrorList{}
	unversionedTolerations := make([]api.Toleration, len(versionedTolerations))
	for i := range versionedTolerations {
		if err := k8s_api_v1.Convert_v1_Toleration_To_core_Toleration(&versionedTolerations[i], &unversionedTolerations[i], nil); err != nil {
			allErrors = append(allErrors, field.Invalid(fldPath.Index(i), unversionedTolerations[i], err.Error()))
		}
	}
	allErrors = append(allErrors, apivalidation.ValidateTolerations(unversionedTolerations, fldPath)...)
	return allErrors
}
