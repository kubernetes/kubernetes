package validation

import (
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core/validation"

	authorizationv1 "github.com/openshift/api/authorization/v1"
)

func ValidateRoleBindingRestriction(rbr *authorizationv1.RoleBindingRestriction) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&rbr.ObjectMeta, true,
		apimachineryvalidation.NameIsDNSSubdomain, field.NewPath("metadata"))

	allErrs = append(allErrs,
		ValidateRoleBindingRestrictionSpec(&rbr.Spec, field.NewPath("spec"))...)

	return allErrs
}

func ValidateRoleBindingRestrictionUpdate(rbr, old *authorizationv1.RoleBindingRestriction) field.ErrorList {
	allErrs := ValidateRoleBindingRestriction(rbr)

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&rbr.ObjectMeta,
		&old.ObjectMeta, field.NewPath("metadata"))...)

	return allErrs
}

func ValidateRoleBindingRestrictionSpec(spec *authorizationv1.RoleBindingRestrictionSpec, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	const invalidMsg = `must specify exactly one of userrestriction, grouprestriction, or serviceaccountrestriction`

	if spec.UserRestriction != nil {
		if spec.GroupRestriction != nil {
			allErrs = append(allErrs, field.Invalid(fld.Child("grouprestriction"),
				"both userrestriction and grouprestriction specified", invalidMsg))
		}
		if spec.ServiceAccountRestriction != nil {
			allErrs = append(allErrs,
				field.Invalid(fld.Child("serviceaccountrestriction"),
					"both userrestriction and serviceaccountrestriction specified", invalidMsg))
		}
	} else if spec.GroupRestriction != nil {
		if spec.ServiceAccountRestriction != nil {
			allErrs = append(allErrs,
				field.Invalid(fld.Child("serviceaccountrestriction"),
					"both grouprestriction and serviceaccountrestriction specified", invalidMsg))
		}
	} else if spec.ServiceAccountRestriction == nil {
		allErrs = append(allErrs, field.Required(fld.Child("userrestriction"),
			invalidMsg))
	}

	if spec.UserRestriction != nil {
		allErrs = append(allErrs, ValidateRoleBindingRestrictionUser(spec.UserRestriction, fld.Child("userrestriction"))...)
	}
	if spec.GroupRestriction != nil {
		allErrs = append(allErrs, ValidateRoleBindingRestrictionGroup(spec.GroupRestriction, fld.Child("grouprestriction"))...)
	}
	if spec.ServiceAccountRestriction != nil {
		allErrs = append(allErrs, ValidateRoleBindingRestrictionServiceAccount(spec.ServiceAccountRestriction, fld.Child("serviceaccountrestriction"))...)
	}

	return allErrs
}

func ValidateRoleBindingRestrictionUser(user *authorizationv1.UserRestriction, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	const invalidMsg = `must specify at least one user, group, or label selector`

	if !(len(user.Users) > 0 || len(user.Groups) > 0 || len(user.Selectors) > 0) {
		allErrs = append(allErrs, field.Required(fld.Child("users"), invalidMsg))
	}

	for i, selector := range user.Selectors {
		allErrs = append(allErrs,
			unversionedvalidation.ValidateLabelSelector(&selector,
				unversionedvalidation.LabelSelectorValidationOptions{},
				fld.Child("selector").Index(i))...)
	}

	return allErrs
}

func ValidateRoleBindingRestrictionGroup(group *authorizationv1.GroupRestriction, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	const invalidMsg = `must specify at least one group or label selector`

	if !(len(group.Groups) > 0 || len(group.Selectors) > 0) {
		allErrs = append(allErrs, field.Required(fld.Child("groups"), invalidMsg))
	}

	for i, selector := range group.Selectors {
		allErrs = append(allErrs,
			unversionedvalidation.ValidateLabelSelector(&selector,
				unversionedvalidation.LabelSelectorValidationOptions{},
				fld.Child("selector").Index(i))...)
	}

	return allErrs
}

func ValidateRoleBindingRestrictionServiceAccount(sa *authorizationv1.ServiceAccountRestriction, fld *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	const invalidMsg = `must specify at least one service account or namespace`

	if !(len(sa.ServiceAccounts) > 0 || len(sa.Namespaces) > 0) {
		allErrs = append(allErrs,
			field.Required(fld.Child("serviceaccounts"), invalidMsg))
	}

	return allErrs
}
