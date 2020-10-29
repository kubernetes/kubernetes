package validation

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	corekubev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/validation"

	quotav1 "github.com/openshift/api/quota/v1"
)

func spec(scopes ...corev1.ResourceQuotaScope) corev1.ResourceQuotaSpec {
	return corev1.ResourceQuotaSpec{
		Hard: corev1.ResourceList{
			corev1.ResourceCPU:                    resource.MustParse("100"),
			corev1.ResourceMemory:                 resource.MustParse("10000"),
			corev1.ResourceRequestsCPU:            resource.MustParse("100"),
			corev1.ResourceRequestsMemory:         resource.MustParse("10000"),
			corev1.ResourceLimitsCPU:              resource.MustParse("100"),
			corev1.ResourceLimitsMemory:           resource.MustParse("10000"),
			corev1.ResourcePods:                   resource.MustParse("10"),
			corev1.ResourceServices:               resource.MustParse("0"),
			corev1.ResourceReplicationControllers: resource.MustParse("10"),
			corev1.ResourceQuotas:                 resource.MustParse("10"),
			corev1.ResourceConfigMaps:             resource.MustParse("10"),
			corev1.ResourceSecrets:                resource.MustParse("10"),
		},
		Scopes: scopes,
	}
}

func scopeableSpec(scopes ...corev1.ResourceQuotaScope) corev1.ResourceQuotaSpec {
	return corev1.ResourceQuotaSpec{
		Hard: corev1.ResourceList{
			corev1.ResourceCPU:            resource.MustParse("100"),
			corev1.ResourceMemory:         resource.MustParse("10000"),
			corev1.ResourceRequestsCPU:    resource.MustParse("100"),
			corev1.ResourceRequestsMemory: resource.MustParse("10000"),
			corev1.ResourceLimitsCPU:      resource.MustParse("100"),
			corev1.ResourceLimitsMemory:   resource.MustParse("10000"),
		},
		Scopes: scopes,
	}
}

func TestValidationClusterQuota(t *testing.T) {
	// storage is not yet supported as a quota tracked resource
	invalidQuotaResourceSpec := corev1.ResourceQuotaSpec{
		Hard: corev1.ResourceList{
			corev1.ResourceStorage: resource.MustParse("10"),
		},
	}
	validLabels := map[string]string{"a": "b"}

	errs := ValidateClusterResourceQuota(
		&quotav1.ClusterResourceQuota{
			ObjectMeta: metav1.ObjectMeta{Name: "good"},
			Spec: quotav1.ClusterResourceQuotaSpec{
				Selector: quotav1.ClusterResourceQuotaSelector{LabelSelector: &metav1.LabelSelector{MatchLabels: validLabels}},
				Quota:    spec(),
			},
		},
	)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string]struct {
		A quotav1.ClusterResourceQuota
		T field.ErrorType
		F string
	}{
		"non-zero-length namespace": {
			A: quotav1.ClusterResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Namespace: "bad", Name: "good"},
				Spec: quotav1.ClusterResourceQuotaSpec{
					Selector: quotav1.ClusterResourceQuotaSelector{LabelSelector: &metav1.LabelSelector{MatchLabels: validLabels}},
					Quota:    spec(),
				},
			},
			T: field.ErrorTypeForbidden,
			F: "metadata.namespace",
		},
		"missing label selector": {
			A: quotav1.ClusterResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "good"},
				Spec: quotav1.ClusterResourceQuotaSpec{
					Quota: spec(),
				},
			},
			T: field.ErrorTypeRequired,
			F: "spec.selector",
		},
		"ok scope": {
			A: quotav1.ClusterResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "good"},
				Spec: quotav1.ClusterResourceQuotaSpec{
					Quota: scopeableSpec(corev1.ResourceQuotaScopeNotTerminating),
				},
			},
			T: field.ErrorTypeRequired,
			F: "spec.selector",
		},
		"bad scope": {
			A: quotav1.ClusterResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "good"},
				Spec: quotav1.ClusterResourceQuotaSpec{
					Selector: quotav1.ClusterResourceQuotaSelector{LabelSelector: &metav1.LabelSelector{MatchLabels: validLabels}},
					Quota:    spec(corev1.ResourceQuotaScopeNotTerminating),
				},
			},
			T: field.ErrorTypeInvalid,
			F: "spec.quota.scopes",
		},
		"bad quota spec": {
			A: quotav1.ClusterResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Name: "good"},
				Spec: quotav1.ClusterResourceQuotaSpec{
					Selector: quotav1.ClusterResourceQuotaSelector{LabelSelector: &metav1.LabelSelector{MatchLabels: validLabels}},
					Quota:    invalidQuotaResourceSpec,
				},
			},
			T: field.ErrorTypeInvalid,
			F: "spec.quota.hard[storage]",
		},
	}
	for k, v := range errorCases {
		errs := ValidateClusterResourceQuota(&v.A)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.A)
			continue
		}
		for i := range errs {
			if errs[i].Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidationQuota(t *testing.T) {
	tests := map[string]struct {
		A corev1.ResourceQuota
		T field.ErrorType
		F string
	}{
		"scope": {
			A: corev1.ResourceQuota{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "good"},
				Spec:       scopeableSpec(corev1.ResourceQuotaScopeNotTerminating),
			},
		},
	}
	for k, v := range tests {
		internal := core.ResourceQuota{}
		if err := corekubev1.Convert_v1_ResourceQuota_To_core_ResourceQuota(&v.A, &internal, nil); err != nil {
			panic(err)
		}
		errs := validation.ValidateResourceQuota(&internal)
		if len(errs) != 0 {
			t.Errorf("%s: %v", k, errs)
			continue
		}
	}
}
