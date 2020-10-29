package validation

import (
	"sort"

	unversionedvalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/validation"

	quotav1 "github.com/openshift/api/quota/v1"
)

func ValidateClusterResourceQuota(clusterquota *quotav1.ClusterResourceQuota) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&clusterquota.ObjectMeta, false, validation.ValidateResourceQuotaName, field.NewPath("metadata"))

	hasSelectionCriteria := (clusterquota.Spec.Selector.LabelSelector != nil && len(clusterquota.Spec.Selector.LabelSelector.MatchLabels)+len(clusterquota.Spec.Selector.LabelSelector.MatchExpressions) > 0) ||
		(len(clusterquota.Spec.Selector.AnnotationSelector) > 0)

	if !hasSelectionCriteria {
		allErrs = append(allErrs, field.Required(field.NewPath("spec", "selector"), "must restrict the selected projects"))
	}
	if clusterquota.Spec.Selector.LabelSelector != nil {
		allErrs = append(allErrs, unversionedvalidation.ValidateLabelSelector(clusterquota.Spec.Selector.LabelSelector, unversionedvalidation.LabelSelectorValidationOptions{}, field.NewPath("spec", "selector", "labels"))...)
		if len(clusterquota.Spec.Selector.LabelSelector.MatchLabels)+len(clusterquota.Spec.Selector.LabelSelector.MatchExpressions) == 0 {
			allErrs = append(allErrs, field.Invalid(field.NewPath("spec", "selector", "labels"), clusterquota.Spec.Selector.LabelSelector, "must restrict the selected projects"))
		}
	}
	if clusterquota.Spec.Selector.AnnotationSelector != nil {
		allErrs = append(allErrs, validation.ValidateAnnotations(clusterquota.Spec.Selector.AnnotationSelector, field.NewPath("spec", "selector", "annotations"))...)
	}

	internalQuota := &core.ResourceQuotaSpec{}
	if err := v1.Convert_v1_ResourceQuotaSpec_To_core_ResourceQuotaSpec(&clusterquota.Spec.Quota, internalQuota, nil); err != nil {
		panic(err)
	}
	internalStatus := &core.ResourceQuotaStatus{}
	if err := v1.Convert_v1_ResourceQuotaStatus_To_core_ResourceQuotaStatus(&clusterquota.Status.Total, internalStatus, nil); err != nil {
		panic(err)
	}

	allErrs = append(allErrs, validation.ValidateResourceQuotaSpec(internalQuota, field.NewPath("spec", "quota"))...)
	allErrs = append(allErrs, validation.ValidateResourceQuotaStatus(internalStatus, field.NewPath("status", "overall"))...)

	orderedNamespaces := clusterquota.Status.Namespaces.DeepCopy()
	sort.Slice(orderedNamespaces, func(i, j int) bool {
		return orderedNamespaces[i].Namespace < orderedNamespaces[j].Namespace
	})

	for _, namespace := range orderedNamespaces {
		fldPath := field.NewPath("status", "namespaces").Key(namespace.Namespace)
		for k, v := range namespace.Status.Used {
			resPath := fldPath.Key(string(k))
			allErrs = append(allErrs, validation.ValidateResourceQuotaResourceName(core.ResourceName(k), resPath)...)
			allErrs = append(allErrs, validation.ValidateResourceQuantityValue(core.ResourceName(k), v, resPath)...)
		}
	}

	return allErrs
}

func ValidateClusterResourceQuotaUpdate(clusterquota, oldClusterResourceQuota *quotav1.ClusterResourceQuota) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&clusterquota.ObjectMeta, &oldClusterResourceQuota.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateClusterResourceQuota(clusterquota)...)

	return allErrs
}
