/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package validation

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateResourceQuota(t *testing.T) {
	spec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU:                    resource.MustParse("100"),
			core.ResourceMemory:                 resource.MustParse("10000"),
			core.ResourceRequestsCPU:            resource.MustParse("100"),
			core.ResourceRequestsMemory:         resource.MustParse("10000"),
			core.ResourceLimitsCPU:              resource.MustParse("100"),
			core.ResourceLimitsMemory:           resource.MustParse("10000"),
			core.ResourcePods:                   resource.MustParse("10"),
			core.ResourceServices:               resource.MustParse("0"),
			core.ResourceReplicationControllers: resource.MustParse("10"),
			core.ResourceQuotas:                 resource.MustParse("10"),
			core.ResourceConfigMaps:             resource.MustParse("10"),
			core.ResourceSecrets:                resource.MustParse("10"),
		},
	}

	terminatingSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU:       resource.MustParse("100"),
			core.ResourceLimitsCPU: resource.MustParse("200"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScopeTerminating},
	}

	nonTerminatingSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScopeNotTerminating},
	}

	bestEffortSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourcePods: resource.MustParse("100"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScopeBestEffort},
	}

	nonBestEffortSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScopeNotBestEffort},
	}

	// storage is not yet supported as a quota tracked resource
	invalidQuotaResourceSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceStorage: resource.MustParse("10"),
		},
	}

	negativeSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU:                    resource.MustParse("-100"),
			core.ResourceMemory:                 resource.MustParse("-10000"),
			core.ResourcePods:                   resource.MustParse("-10"),
			core.ResourceServices:               resource.MustParse("-10"),
			core.ResourceReplicationControllers: resource.MustParse("-10"),
			core.ResourceQuotas:                 resource.MustParse("-10"),
			core.ResourceConfigMaps:             resource.MustParse("-10"),
			core.ResourceSecrets:                resource.MustParse("-10"),
		},
	}

	fractionalComputeSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU: resource.MustParse("100m"),
		},
	}

	fractionalPodSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourcePods:                   resource.MustParse(".1"),
			core.ResourceServices:               resource.MustParse(".5"),
			core.ResourceReplicationControllers: resource.MustParse("1.25"),
			core.ResourceQuotas:                 resource.MustParse("2.5"),
		},
	}

	invalidTerminatingScopePairsSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScopeTerminating, core.ResourceQuotaScopeNotTerminating},
	}

	invalidBestEffortScopePairsSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourcePods: resource.MustParse("100"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScopeBestEffort, core.ResourceQuotaScopeNotBestEffort},
	}

	invalidScopeNameSpec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU: resource.MustParse("100"),
		},
		Scopes: []core.ResourceQuotaScope{core.ResourceQuotaScope("foo")},
	}

	successCases := []core.ResourceQuota{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: spec,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: fractionalComputeSpec,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: terminatingSpec,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: nonTerminatingSpec,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: bestEffortSpec,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "abc",
				Namespace: "foo",
			},
			Spec: nonBestEffortSpec,
		},
	}

	for _, successCase := range successCases {
		if errs := ValidateResourceQuota(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]struct {
		R core.ResourceQuota
		D string
	}{
		"zero-length Name": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: "foo"}, Spec: spec},
			"name or generateName is required",
		},
		"zero-length Namespace": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: ""}, Spec: spec},
			"",
		},
		"invalid Name": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "^Invalid", Namespace: "foo"}, Spec: spec},
			dnsSubdomainLabelErrMsg,
		},
		"invalid Namespace": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "^Invalid"}, Spec: spec},
			dnsLabelErrMsg,
		},
		"negative-limits": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: negativeSpec},
			isNegativeErrorMsg,
		},
		"fractional-api-resource": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: fractionalPodSpec},
			isNotIntegerErrorMsg,
		},
		"invalid-quota-resource": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidQuotaResourceSpec},
			isInvalidQuotaResource,
		},
		"invalid-quota-terminating-pair": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidTerminatingScopePairsSpec},
			"conflicting scopes",
		},
		"invalid-quota-besteffort-pair": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidBestEffortScopePairsSpec},
			"conflicting scopes",
		},
		"invalid-quota-scope-name": {
			core.ResourceQuota{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: invalidScopeNameSpec},
			"unsupported scope",
		},
	}
	for k, v := range errorCases {
		errs := ValidateResourceQuota(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			if !strings.Contains(errs[i].Detail, v.D) {
				t.Errorf("[%s]: expected error detail either empty or %s, got %s", k, v.D, errs[i].Detail)
			}
		}
	}
}

func TestValidateResourceQuotaWithAlphaLocalStorageCapacityIsolation(t *testing.T) {
	spec := core.ResourceQuotaSpec{
		Hard: core.ResourceList{
			core.ResourceCPU:                      resource.MustParse("100"),
			core.ResourceMemory:                   resource.MustParse("10000"),
			core.ResourceRequestsCPU:              resource.MustParse("100"),
			core.ResourceRequestsMemory:           resource.MustParse("10000"),
			core.ResourceLimitsCPU:                resource.MustParse("100"),
			core.ResourceLimitsMemory:             resource.MustParse("10000"),
			core.ResourcePods:                     resource.MustParse("10"),
			core.ResourceServices:                 resource.MustParse("0"),
			core.ResourceReplicationControllers:   resource.MustParse("10"),
			core.ResourceQuotas:                   resource.MustParse("10"),
			core.ResourceConfigMaps:               resource.MustParse("10"),
			core.ResourceSecrets:                  resource.MustParse("10"),
			core.ResourceEphemeralStorage:         resource.MustParse("10000"),
			core.ResourceRequestsEphemeralStorage: resource.MustParse("10000"),
			core.ResourceLimitsEphemeralStorage:   resource.MustParse("10000"),
		},
	}
	resourceQuota := &core.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Spec: spec,
	}

	// Enable alpha feature LocalStorageCapacityIsolation
	err := utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	if errs := ValidateResourceQuota(resourceQuota); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	// Disable alpha feature LocalStorageCapacityIsolation
	err = utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	errs := ValidateResourceQuota(resourceQuota)
	if len(errs) == 0 {
		t.Errorf("expected failure for %s", resourceQuota.Name)
	}
	expectedErrMes := "ResourceEphemeralStorage field disabled by feature-gate for ResourceQuota"
	for i := range errs {
		if !strings.Contains(errs[i].Detail, expectedErrMes) {
			t.Errorf("[%s]: expected error detail either empty or %s, got %s", resourceQuota.Name, expectedErrMes, errs[i].Detail)
		}
	}
}

func TestValidateResourceNames(t *testing.T) {
	table := []struct {
		input   string
		success bool
		expect  string
	}{
		{"memory", true, ""},
		{"cpu", true, ""},
		{"storage", true, ""},
		{"requests.cpu", true, ""},
		{"requests.memory", true, ""},
		{"requests.storage", true, ""},
		{"limits.cpu", true, ""},
		{"limits.memory", true, ""},
		{"network", false, ""},
		{"disk", false, ""},
		{"", false, ""},
		{".", false, ""},
		{"..", false, ""},
		{"my.favorite.app.co/12345", true, ""},
		{"my.favorite.app.co/_12345", false, ""},
		{"my.favorite.app.co/12345_", false, ""},
		{"kubernetes.io/..", false, ""},
		{"kubernetes.io/" + strings.Repeat("a", 63), true, ""},
		{"kubernetes.io/" + strings.Repeat("a", 64), false, ""},
		{"kubernetes.io//", false, ""},
		{"kubernetes.io", false, ""},
		{"kubernetes.io/will/not/work/", false, ""},
	}
	for k, item := range table {
		err := validateResourceName(item.input, field.NewPath("field"))
		if len(err) != 0 && item.success {
			t.Errorf("expected no failure for input %q", item.input)
		} else if len(err) == 0 && !item.success {
			t.Errorf("expected failure for input %q", item.input)
			for i := range err {
				detail := err[i].Detail
				if detail != "" && !strings.Contains(detail, item.expect) {
					t.Errorf("%d: expected error detail either empty or %s, got %s", k, item.expect, detail)
				}
			}
		}
	}
}
