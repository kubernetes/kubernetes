package clusterresourcequota

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"

	quotav1 "github.com/openshift/api/quota/v1"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation"
	quotavalidation "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/clusterresourcequota/validation"
)

const PluginName = "quota.openshift.io/ValidateClusterResourceQuota"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return customresourcevalidation.NewValidator(
			map[schema.GroupResource]bool{
				{Group: quotav1.GroupName, Resource: "clusterresourcequotas"}: true,
			},
			map[schema.GroupVersionKind]customresourcevalidation.ObjectValidator{
				quotav1.GroupVersion.WithKind("ClusterResourceQuota"): clusterResourceQuotaV1{},
			})
	})
}

func toClusterResourceQuota(uncastObj runtime.Object) (*quotav1.ClusterResourceQuota, field.ErrorList) {
	if uncastObj == nil {
		return nil, nil
	}

	allErrs := field.ErrorList{}

	obj, ok := uncastObj.(*quotav1.ClusterResourceQuota)
	if !ok {
		return nil, append(allErrs,
			field.NotSupported(field.NewPath("kind"), fmt.Sprintf("%T", uncastObj), []string{"ClusterResourceQuota"}),
			field.NotSupported(field.NewPath("apiVersion"), fmt.Sprintf("%T", uncastObj), []string{quotav1.GroupVersion.String()}))
	}

	return obj, nil
}

type clusterResourceQuotaV1 struct {
}

func (clusterResourceQuotaV1) ValidateCreate(_ context.Context, obj runtime.Object) field.ErrorList {
	clusterResourceQuotaObj, errs := toClusterResourceQuota(obj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&clusterResourceQuotaObj.ObjectMeta, false, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	errs = append(errs, quotavalidation.ValidateClusterResourceQuota(clusterResourceQuotaObj)...)

	return errs
}

func (clusterResourceQuotaV1) ValidateUpdate(_ context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	clusterResourceQuotaObj, errs := toClusterResourceQuota(obj)
	if len(errs) > 0 {
		return errs
	}
	clusterResourceQuotaOldObj, errs := toClusterResourceQuota(oldObj)
	if len(errs) > 0 {
		return errs
	}

	errs = append(errs, validation.ValidateObjectMeta(&clusterResourceQuotaObj.ObjectMeta, false, validation.NameIsDNSSubdomain, field.NewPath("metadata"))...)
	errs = append(errs, quotavalidation.ValidateClusterResourceQuotaUpdate(clusterResourceQuotaObj, clusterResourceQuotaOldObj)...)

	return errs
}

func (c clusterResourceQuotaV1) ValidateStatusUpdate(ctx context.Context, obj runtime.Object, oldObj runtime.Object) field.ErrorList {
	return c.ValidateUpdate(ctx, obj, oldObj)
}
