package node

import (
	"context"
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

var rejectionScenarios = []struct {
	fromProfile configv1.WorkerLatencyProfileType
	toProfile   configv1.WorkerLatencyProfileType
}{
	{fromProfile: "", toProfile: configv1.LowUpdateSlowReaction},
	{fromProfile: configv1.LowUpdateSlowReaction, toProfile: ""},
	{fromProfile: configv1.DefaultUpdateDefaultReaction, toProfile: configv1.LowUpdateSlowReaction},
	{fromProfile: configv1.LowUpdateSlowReaction, toProfile: configv1.DefaultUpdateDefaultReaction},
}

const PluginName = "config.openshift.io/RestrictExtremeWorkerLatencyProfile"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				configv1.Resource("nodes"): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				configv1.GroupVersion.WithKind("Node"): configNodeV1{},
			})
	})
}

func toConfigNodeV1(uncastObj runtime.Object) (*configv1.Node, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.Node)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"Node"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type configNodeV1 struct{}

func validateConfigNodeForExtremeLatencyProfile(obj, oldObj *configv1.Node) *field.Error {
	fromProfile := oldObj.Spec.WorkerLatencyProfile
	toProfile := obj.Spec.WorkerLatencyProfile

	for _, rejectionScenario := range rejectionScenarios {
		if fromProfile == rejectionScenario.fromProfile && toProfile == rejectionScenario.toProfile {
			return field.Invalid(field.NewPath("spec", "workerLatencyProfile"), obj.Spec.WorkerLatencyProfile,
				fmt.Sprintf(
					"cannot update worker latency profile from %q to %q as extreme profile transition is unsupported, please select any other profile with supported transition such as %q",
					oldObj.Spec.WorkerLatencyProfile,
					obj.Spec.WorkerLatencyProfile,
					configv1.MediumUpdateAverageReaction,
				),
			)
		}
	}
	return nil
}

func (configNodeV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, allErrs := toConfigNodeV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)

	return allErrs
}

func (configNodeV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, allErrs := toConfigNodeV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}
	oldObj, allErrs := toConfigNodeV1(uncastOldObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	if err := validateConfigNodeForExtremeLatencyProfile(obj, oldObj); err != nil {
		allErrs = append(allErrs, err)
	}

	return allErrs
}

func (configNodeV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toConfigNodeV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toConfigNodeV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}
