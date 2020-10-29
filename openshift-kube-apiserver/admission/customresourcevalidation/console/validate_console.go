package console

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

const PluginName = "config.openshift.io/ValidateConsole"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				configv1.GroupVersion.WithResource("consoles").GroupResource(): true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				configv1.GroupVersion.WithKind("Console"): consoleV1{},
			})
	})
}

func toConsoleV1(uncastObj runtime.Object) (*configv1.Console, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	errs := field.ErrorList{}

	obj, ok := uncastObj.(*configv1.Console)
	if !ok {
		return nil, append(errs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"Console"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"config.openshift.io/v1"}))
	}

	return obj, nil
}

type consoleV1 struct{}

func (consoleV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, errs := toConsoleV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, customresourcevalidation.RequireNameCluster, field.NewPath("metadata"))...)
	errs = append(errs, validateConsoleSpecCreate(obj.Spec)...)

	return errs
}

func (consoleV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toConsoleV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toConsoleV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, validateConsoleSpecUpdate(obj.Spec, oldObj.Spec)...)

	return errs
}

func (consoleV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toConsoleV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toConsoleV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}

	// TODO validate the obj.  remember that status validation should *never* fail on spec validation errors.
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	errs = append(errs, validateConsoleStatus(obj.Status)...)

	return errs
}

func validateConsoleSpecCreate(spec configv1.ConsoleSpec) field.ErrorList {
	errs := field.ErrorList{}

	// TODO

	return errs
}

func validateConsoleSpecUpdate(newSpec, oldSpec configv1.ConsoleSpec) field.ErrorList {
	errs := field.ErrorList{}

	// TODO

	return errs
}

func validateConsoleStatus(status configv1.ConsoleStatus) field.ErrorList {
	errs := field.ErrorList{}

	// TODO

	return errs
}
