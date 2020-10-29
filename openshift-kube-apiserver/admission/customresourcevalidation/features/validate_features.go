package features

import (
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/util/sets"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "config.openshift.io/ValidateFeatureGate"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				configv1.Resource("features"): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				configv1.GroupVersion.WithKind("FeatureGate"): featureGateV1{},
			})
	})
}

func toFeatureGateV1(uncastObj runtime.Object) (*configv1.FeatureGate, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.FeatureGate)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"FeatureGate"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type featureGateV1 struct {
}

var knownFeatureSets = sets.NewString("", string(configv1.TechPreviewNoUpgrade), string(configv1.CustomNoUpgrade))

func validateFeatureGateSpecCreate(spec configv1.FeatureGateSpec) field.ErrorList {
	allErrs := field.ErrorList{}

	// on create, we only allow values that we are aware of
	if !knownFeatureSets.Has(string(spec.FeatureSet)) {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("spec.featureSet"), spec.FeatureSet, knownFeatureSets.List()))
	}

	return allErrs
}

func validateFeatureGateSpecUpdate(spec, oldSpec configv1.FeatureGateSpec) field.ErrorList {
	allErrs := field.ErrorList{}

	// on update, we don't fail validation on a field we don't recognize as long as it is not changing
	if !knownFeatureSets.Has(string(spec.FeatureSet)) && oldSpec.FeatureSet != spec.FeatureSet {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("spec.featureSet"), spec.FeatureSet, knownFeatureSets.List()))
	}

	// we do not allow anyone to take back TechPreview
	if oldSpec.FeatureSet == configv1.TechPreviewNoUpgrade && spec.FeatureSet != configv1.TechPreviewNoUpgrade {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec.featureSet"), "once enabled, tech preview features may not be disabled"))
	}
	// we do not allow anyone to take back CustomNoUpgrade
	if oldSpec.FeatureSet == configv1.CustomNoUpgrade && spec.FeatureSet != configv1.CustomNoUpgrade {
		allErrs = append(allErrs, field.Forbidden(field.NewPath("spec.featureSet"), "once enabled, custom feature gates may not be disabled"))
	}

	return allErrs
}

func (featureGateV1) ValidateCreate(uncastObj runtime.Object) field.ErrorList {
	obj, allErrs := toFeatureGateV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateFeatureGateSpecCreate(obj.Spec)...)

	return allErrs
}

func (featureGateV1) ValidateUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, allErrs := toFeatureGateV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}
	oldObj, allErrs := toFeatureGateV1(uncastOldObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	allErrs = append(allErrs, validateFeatureGateSpecUpdate(obj.Spec, oldObj.Spec)...)

	return allErrs
}

func (featureGateV1) ValidateStatusUpdate(uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toFeatureGateV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toFeatureGateV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}
