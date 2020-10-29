package rolebindingrestriction

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	authorizationv1 "github.com/openshift/api/authorization/v1"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
	rbrvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/rolebindingrestriction/validation"
)

const PluginName = "authorization.openshift.io/ValidateRoleBindingRestriction"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				{Group: authorizationv1.GroupName, Resource: "rolebindingrestrictions"}: true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				authorizationv1.GroupVersion.WithKind("RoleBindingRestriction"): roleBindingRestrictionV1{},
			})
	})
}

func toRoleBindingRestriction(uncastObj runtime.Object) (*authorizationv1.RoleBindingRestriction, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*authorizationv1.RoleBindingRestriction)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"RoleBindingRestriction"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{authorizationv1.GroupVersion.String()}))
	}

	return obj, nil
}

type roleBindingRestrictionV1 struct {
}

func (roleBindingRestrictionV1) ValidateCreate(obj runtime.Object) field.ErrorList {
	roleBindingRestrictionObj, errs := toRoleBindingRestriction(obj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&roleBindingRestrictionObj.ObjectMeta, true, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	errs = append(errs, rbrvalidation.ValidateRoleBindingRestriction(roleBindingRestrictionObj)...)

	return errs
}

func (roleBindingRestrictionV1) ValidateUpdate(obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	roleBindingRestrictionObj, errs := toRoleBindingRestriction(obj)
	if len(errs) > 0 {
		return errs
	}
	roleBindingRestrictionOldObj, errs := toRoleBindingRestriction(oldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&roleBindingRestrictionObj.ObjectMeta, true, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	errs = append(errs, rbrvalidation.ValidateRoleBindingRestrictionUpdate(roleBindingRestrictionObj, roleBindingRestrictionOldObj)...)

	return errs
}

func (r roleBindingRestrictionV1) ValidateStatusUpdate(obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	return r.ValidateUpdate(obj, oldObj)
}
