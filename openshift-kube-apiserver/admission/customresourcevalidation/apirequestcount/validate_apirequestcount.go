package apirequestcount

import (
	"context"
	"fmt"
	"io"
	"strings"

	apiv1 "github.com/openshift/api/apiserver/v1"
	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
)

const PluginName = "config.openshift.io/ValidateAPIRequestCount"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return newValidateAPIRequestCount()
	})
}

func newValidateAPIRequestCount() (admission.Interface, error) {
	return customresourcevalidation.NewValidator(
		map[schema.GroupResource]bool{
			apiv1.GroupVersion.WithResource("apirequestcounts").GroupResource(): true,
		},
		map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
			apiv1.GroupVersion.WithKind("APIRequestCount"): apiRequestCountV1{},
		})
}

type apiRequestCountV1 struct {
}

func toAPIRequestCountV1(uncastObj runtime.Object) (*apiv1.APIRequestCount, field.ErrorList) {
	obj, ok := uncastObj.(*apiv1.APIRequestCount)
	if !ok {
		return nil, field.ErrorList{
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"APIRequestCount"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{"apiserver.openshift.io/v1"}),
		}
	}

	return obj, nil
}

func (a apiRequestCountV1) ValidateCreate(_ context.Context, uncastObj runtime.Object) field.ErrorList {
	obj, errs := toAPIRequestCountV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	errs = append(errs, validation.ValidateObjectMeta(&obj.ObjectMeta, false, requireNameGVR, field.NewPath("metadata"))...)
	return errs
}

// requireNameGVR is a name validation function that requires the name to be of the form 'resource.version.group'.
func requireNameGVR(name string, _ bool) []string {
	if _, err := NameToResource(name); err != nil {
		return []string{err.Error()}
	}
	return nil
}

// NameToResource parses a name of the form 'resource.version.group'.
func NameToResource(name string) (schema.GroupVersionResource, error) {
	segments := strings.SplitN(name, ".", 3)
	result := schema.GroupVersionResource{Resource: segments[0]}
	switch len(segments) {
	case 3:
		result.Group = segments[2]
		fallthrough
	case 2:
		result.Version = segments[1]
	default:
		return schema.GroupVersionResource{}, fmt.Errorf("apirequestcount %s: name must be of the form 'resource.version.group'", name)
	}
	return result, nil
}

func (a apiRequestCountV1) ValidateUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toAPIRequestCountV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toAPIRequestCountV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	return errs
}

func (a apiRequestCountV1) ValidateStatusUpdate(_ context.Context, uncastObj runtime.Object, uncastOldObj runtime.Object) field.ErrorList {
	obj, errs := toAPIRequestCountV1(uncastObj)
	if len(errs) > 0 {
		return errs
	}
	oldObj, errs := toAPIRequestCountV1(uncastOldObj)
	if len(errs) > 0 {
		return errs
	}
	errs = append(errs, validation.ValidateObjectMetaUpdate(&obj.ObjectMeta, &oldObj.ObjectMeta, field.NewPath("metadata"))...)
	return errs
}
