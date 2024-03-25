package kubecontrollermanager

import (
	"context"
	"fmt"
	"io"

	operatorv1 "github.com/openshift/api/operator/v1"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "operator.openshift.io/ValidateKubeControllerManager"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				operatorv1.Resource("kubecontrollermanagers"): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				operatorv1.GroupVersion.WithKind("KubeControllerManager"): kubeControllerManagerV1{},
			})
	})
}

func toKubeControllerManager(uncastObj runtime.Object) (*operatorv1.KubeControllerManager, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*operatorv1.KubeControllerManager)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"KubeControllerManager"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"operator.openshift.io/v1"}))
	}

	return obj, nil
}

type kubeControllerManagerV1 struct {
}

func validateKubeControllerManagerSpecCreate(spec operatorv1.KubeControllerManagerSpec) field.ErrorList {
	allErrs := field.ErrorList{}

	// on create, we allow anything
	return allErrs
}

func validateKubeControllerManagerSpecUpdate(spec, oldSpec operatorv1.KubeControllerManagerSpec) field.ErrorList {
	allErrs := field.ErrorList{}

	// on update, fail if we go from secure to insecure
	if oldSpec.UseMoreSecureServiceCA && !spec.UseMoreSecureServiceCA {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec.useMoreSecureServiceCA"), "once enabled, the more secure service-ca.crt cannot be disabled"))
	}

	return allErrs
}

func (kubeControllerManagerV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, allErrs := toKubeControllerManager(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateKubeControllerManagerSpecCreate(obj.Spec)...)

	return allErrs
}

func (kubeControllerManagerV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, allErrs := toKubeControllerManager(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}
	oldObj, allErrs := toKubeControllerManager(uncastOldObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateKubeControllerManagerSpecUpdate(obj.Spec, oldObj.Spec)...)

	return allErrs
}

func (kubeControllerManagerV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toKubeControllerManager(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toKubeControllerManager(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}
