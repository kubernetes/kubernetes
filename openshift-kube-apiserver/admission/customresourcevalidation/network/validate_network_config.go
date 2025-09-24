package network

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"

	configv1 "github.com/openshift/api/config/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "config.openshift.io/ValidateNetwork"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				configv1.Resource("networks"): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				configv1.GroupVersion.WithKind("Network"): networkV1{},
			})
	})
}

func toNetworkV1(uncastObj runtime.Object) (*configv1.Network, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.Network)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"Network"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type networkV1 struct {
}

func validateNetworkServiceNodePortRangeUpdate(obj, oldObj *configv1.Network) *field.Error {
	var err error
	defaultRange := kubeoptions.DefaultServiceNodePortRange
	oldRange := &defaultRange
	newRange := &defaultRange

	oldRangeStr := oldObj.Spec.ServiceNodePortRange
	if oldRangeStr != "" {
		if oldRange, err = utilnet.ParsePortRange(oldRangeStr); err != nil {
			return field.Invalid(field.NewPath("spec", "serviceNodePortRange"),
				oldRangeStr,
				fmt.Sprintf("failed to parse the old port range: %v", err))
		}
	}
	newRangeStr := obj.Spec.ServiceNodePortRange
	if newRangeStr != "" {
		if newRange, err = utilnet.ParsePortRange(newRangeStr); err != nil {
			return field.Invalid(field.NewPath("spec", "serviceNodePortRange"),
				newRangeStr,
				fmt.Sprintf("failed to parse the new port range: %v", err))
		}
	}
	if !newRange.Contains(oldRange.Base) || !newRange.Contains(oldRange.Base+oldRange.Size-1) {
		return field.Invalid(field.NewPath("spec", "serviceNodePortRange"),
			newRangeStr,
			fmt.Sprintf("new service node port range %s does not completely cover the previous range %s", newRange, oldRange))
	}
	return nil
}

func (networkV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, allErrs := toNetworkV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)

	return allErrs
}

func (networkV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, allErrs := toNetworkV1(uncastObj)
	if len(allErrs) > 0 {
		return allErrs
	}
	oldObj, allErrs := toNetworkV1(uncastOldObj)
	if len(allErrs) > 0 {
		return allErrs
	}

	allErrs = append(allErrs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	if err := validateNetworkServiceNodePortRangeUpdate(obj, oldObj); err != nil {
		allErrs = append(allErrs, err)
	}

	return allErrs
}

func (networkV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toNetworkV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toNetworkV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)

	return errs
}
