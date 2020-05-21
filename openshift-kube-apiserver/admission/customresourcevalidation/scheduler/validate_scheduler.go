package scheduler

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "config.openshift.io/ValidateScheduler"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				configv1.Resource("schedulers"): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				configv1.GroupVersion.WithKind("Scheduler"): schedulerV1{},
			})
	})
}

func toSchedulerV1(uncastObj runtime.Object) (*configv1.Scheduler, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.Scheduler)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"Scheduler"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type schedulerV1 struct {
}

func validateSchedulerSpec(spec configv1.SchedulerSpec) field.ErrorList {
	allErrs := field.ErrorList{}

	if name := spec.Policy.Name; len(name) > 0 {
		for _, msg := range validation.NameIsDNSSubdomain(spec.Policy.Name, false) {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec.Policy.name"), name, msg))
		}
	}

	return allErrs
}

func (schedulerV1) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	obj, allErrs := toSchedulerV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateSchedulerSpec(obj.Spec)...)

	return allErrs
}

func (schedulerV1) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, allErrs := toSchedulerV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}
	oldObj, allErrs := toSchedulerV1(uncastOldObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateSchedulerSpec(obj.Spec)...)

	return allErrs
}

func (schedulerV1) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toSchedulerV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toSchedulerV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}
