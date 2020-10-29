package securitycontextconstraints

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	securityv1 "github.com/openshift/api/security/v1"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
	sccvalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints/validation"
)

const PluginName = "security.openshift.io/ValidateSecurityContextConstraints"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				{Group: securityv1.GroupName, Resource: "securitycontextconstraints"}: true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				securityv1.GroupVersion.WithKind("SecurityContextConstraints"): securityContextConstraintsV1{},
			})
	})
}

func toSecurityContextConstraints(uncastObj runtime.Object) (*securityv1.SecurityContextConstraints, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	obj, ok := uncastObj.(*securityv1.SecurityContextConstraints)
	if !ok {
		return nil, field.ErrorList{
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"SecurityContextConstraints"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{securityv1.GroupVersion.String()}),
		}
	}

	return obj, nil
}

type securityContextConstraintsV1 struct {
}

func (securityContextConstraintsV1) ValidateCreate(_ context.Context, obj runtime.Object) field.ErrorList {
	securityContextConstraintsObj, errs := toSecurityContextConstraints(obj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, sccvalidation.ValidateSecurityContextConstraints(securityContextConstraintsObj)...)

	return errs
}

func (securityContextConstraintsV1) ValidateUpdate(_ context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	securityContextConstraintsObj, errs := toSecurityContextConstraints(obj)
	if len(errs) > 0 {
		return errs
	}
	securityContextConstraintsOldObj, errs := toSecurityContextConstraints(oldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, sccvalidation.ValidateSecurityContextConstraintsUpdate(securityContextConstraintsObj, securityContextConstraintsOldObj)...)

	return errs
}

func (c securityContextConstraintsV1) ValidateStatusUpdate(ctx context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	return c.ValidateUpdate(ctx, obj, oldObj)
}
