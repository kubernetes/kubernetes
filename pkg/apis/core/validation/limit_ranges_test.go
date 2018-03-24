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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateLimitRangeForLocalStorage(t *testing.T) {
	testCases := []struct {
		name string
		spec core.LimitRangeSpec
	}{
		{
			name: "all-fields-valid",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 core.LimitTypePod,
						Max:                  getLocalStorageResourceList("10000Mi"),
						Min:                  getLocalStorageResourceList("100Mi"),
						MaxLimitRequestRatio: getLocalStorageResourceList(""),
					},
					{
						Type:                 core.LimitTypeContainer,
						Max:                  getLocalStorageResourceList("10000Mi"),
						Min:                  getLocalStorageResourceList("100Mi"),
						Default:              getLocalStorageResourceList("500Mi"),
						DefaultRequest:       getLocalStorageResourceList("200Mi"),
						MaxLimitRequestRatio: getLocalStorageResourceList(""),
					},
				},
			},
		},
	}

	// Enable alpha feature LocalStorageCapacityIsolation
	err := utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}

	for _, testCase := range testCases {
		limitRange := &core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: testCase.name, Namespace: "foo"}, Spec: testCase.spec}
		if errs := ValidateLimitRange(limitRange); len(errs) != 0 {
			t.Errorf("Case %v, unexpected error: %v", testCase.name, errs)
		}
	}

	// Disable alpha feature LocalStorageCapacityIsolation
	err = utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	for _, testCase := range testCases {
		limitRange := &core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: testCase.name, Namespace: "foo"}, Spec: testCase.spec}
		if errs := ValidateLimitRange(limitRange); len(errs) == 0 {
			t.Errorf("Case %v, expected feature gate unable error but actually no error", testCase.name)
		}
	}
}

func TestValidateLimitRange(t *testing.T) {
	successCases := []struct {
		name string
		spec core.LimitRangeSpec
	}{
		{
			name: "all-fields-valid",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 core.LimitTypePod,
						Max:                  getResourceList("100m", "10000Mi"),
						Min:                  getResourceList("5m", "100Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
					{
						Type:                 core.LimitTypeContainer,
						Max:                  getResourceList("100m", "10000Mi"),
						Min:                  getResourceList("5m", "100Mi"),
						Default:              getResourceList("50m", "500Mi"),
						DefaultRequest:       getResourceList("10m", "200Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
					{
						Type: core.LimitTypePersistentVolumeClaim,
						Max:  getStorageResourceList("10Gi"),
						Min:  getStorageResourceList("5Gi"),
					},
				},
			},
		},
		{
			name: "pvc-min-only",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type: core.LimitTypePersistentVolumeClaim,
						Min:  getStorageResourceList("5Gi"),
					},
				},
			},
		},
		{
			name: "pvc-max-only",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type: core.LimitTypePersistentVolumeClaim,
						Max:  getStorageResourceList("10Gi"),
					},
				},
			},
		},
		{
			name: "all-fields-valid-big-numbers",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 core.LimitTypeContainer,
						Max:                  getResourceList("100m", "10000T"),
						Min:                  getResourceList("5m", "100Mi"),
						Default:              getResourceList("50m", "500Mi"),
						DefaultRequest:       getResourceList("10m", "200Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
				},
			},
		},
		{
			name: "thirdparty-fields-all-valid-standard-container-resources",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 "thirdparty.com/foo",
						Max:                  getResourceList("100m", "10000T"),
						Min:                  getResourceList("5m", "100Mi"),
						Default:              getResourceList("50m", "500Mi"),
						DefaultRequest:       getResourceList("10m", "200Mi"),
						MaxLimitRequestRatio: getResourceList("10", ""),
					},
				},
			},
		},
		{
			name: "thirdparty-fields-all-valid-storage-resources",
			spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 "thirdparty.com/foo",
						Max:                  getStorageResourceList("10000T"),
						Min:                  getStorageResourceList("100Mi"),
						Default:              getStorageResourceList("500Mi"),
						DefaultRequest:       getStorageResourceList("200Mi"),
						MaxLimitRequestRatio: getStorageResourceList(""),
					},
				},
			},
		},
	}

	for _, successCase := range successCases {
		limitRange := &core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: successCase.name, Namespace: "foo"}, Spec: successCase.spec}
		if errs := ValidateLimitRange(limitRange); len(errs) != 0 {
			t.Errorf("Case %v, unexpected error: %v", successCase.name, errs)
		}
	}

	errorCases := map[string]struct {
		R core.LimitRange
		D string
	}{
		"zero-length-name": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: "foo"}, Spec: core.LimitRangeSpec{}},
			"name or generateName is required",
		},
		"zero-length-namespace": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: ""}, Spec: core.LimitRangeSpec{}},
			"",
		},
		"invalid-name": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "^Invalid", Namespace: "foo"}, Spec: core.LimitRangeSpec{}},
			dnsSubdomainLabelErrMsg,
		},
		"invalid-namespace": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "^Invalid"}, Spec: core.LimitRangeSpec{}},
			dnsLabelErrMsg,
		},
		"duplicate-limit-type": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type: core.LimitTypePod,
						Max:  getResourceList("100m", "10000m"),
						Min:  getResourceList("0m", "100m"),
					},
					{
						Type: core.LimitTypePod,
						Min:  getResourceList("0m", "100m"),
					},
				},
			}},
			"",
		},
		"default-limit-type-pod": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:    core.LimitTypePod,
						Max:     getResourceList("100m", "10000m"),
						Min:     getResourceList("0m", "100m"),
						Default: getResourceList("10m", "100m"),
					},
				},
			}},
			"may not be specified when `type` is 'Pod'",
		},
		"default-request-limit-type-pod": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:           core.LimitTypePod,
						Max:            getResourceList("100m", "10000m"),
						Min:            getResourceList("0m", "100m"),
						DefaultRequest: getResourceList("10m", "100m"),
					},
				},
			}},
			"may not be specified when `type` is 'Pod'",
		},
		"min value 100m is greater than max value 10m": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type: core.LimitTypePod,
						Max:  getResourceList("10m", ""),
						Min:  getResourceList("100m", ""),
					},
				},
			}},
			"min value 100m is greater than max value 10m",
		},
		"invalid spec default outside range": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:    core.LimitTypeContainer,
						Max:     getResourceList("1", ""),
						Min:     getResourceList("100m", ""),
						Default: getResourceList("2000m", ""),
					},
				},
			}},
			"default value 2 is greater than max value 1",
		},
		"invalid spec default request outside range": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:           core.LimitTypeContainer,
						Max:            getResourceList("1", ""),
						Min:            getResourceList("100m", ""),
						DefaultRequest: getResourceList("2000m", ""),
					},
				},
			}},
			"default request value 2 is greater than max value 1",
		},
		"invalid spec default request more than default": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:           core.LimitTypeContainer,
						Max:            getResourceList("2", ""),
						Min:            getResourceList("100m", ""),
						Default:        getResourceList("500m", ""),
						DefaultRequest: getResourceList("800m", ""),
					},
				},
			}},
			"default request value 800m is greater than default limit value 500m",
		},
		"invalid spec maxLimitRequestRatio less than 1": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 core.LimitTypePod,
						MaxLimitRequestRatio: getResourceList("800m", ""),
					},
				},
			}},
			"ratio 800m is less than 1",
		},
		"invalid spec maxLimitRequestRatio greater than max/min": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 core.LimitTypeContainer,
						Max:                  getResourceList("", "2Gi"),
						Min:                  getResourceList("", "512Mi"),
						MaxLimitRequestRatio: getResourceList("", "10"),
					},
				},
			}},
			"ratio 10 is greater than max/min = 4.000000",
		},
		"invalid non standard limit type": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type:                 "foo",
						Max:                  getStorageResourceList("10000T"),
						Min:                  getStorageResourceList("100Mi"),
						Default:              getStorageResourceList("500Mi"),
						DefaultRequest:       getStorageResourceList("200Mi"),
						MaxLimitRequestRatio: getStorageResourceList(""),
					},
				},
			}},
			"must be a standard limit type or fully qualified",
		},
		"min and max values missing, one required": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type: core.LimitTypePersistentVolumeClaim,
					},
				},
			}},
			"either minimum or maximum storage value is required, but neither was provided",
		},
		"invalid min greater than max": {
			core.LimitRange{ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "foo"}, Spec: core.LimitRangeSpec{
				Limits: []core.LimitRangeItem{
					{
						Type: core.LimitTypePersistentVolumeClaim,
						Min:  getStorageResourceList("10Gi"),
						Max:  getStorageResourceList("1Gi"),
					},
				},
			}},
			"min value 10Gi is greater than max value 1Gi",
		},
	}

	for k, v := range errorCases {
		errs := ValidateLimitRange(&v.R)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			detail := errs[i].Detail
			if !strings.Contains(detail, v.D) {
				t.Errorf("[%s]: expected error detail either empty or %q, got %q", k, v.D, detail)
			}
		}
	}

}

func getResourceList(cpu, memory string) core.ResourceList {
	res := core.ResourceList{}
	if cpu != "" {
		res[core.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[core.ResourceMemory] = resource.MustParse(memory)
	}
	return res
}

func getStorageResourceList(storage string) core.ResourceList {
	res := core.ResourceList{}
	if storage != "" {
		res[core.ResourceStorage] = resource.MustParse(storage)
	}
	return res
}

func getLocalStorageResourceList(ephemeralStorage string) core.ResourceList {
	res := core.ResourceList{}
	if ephemeralStorage != "" {
		res[core.ResourceEphemeralStorage] = resource.MustParse(ephemeralStorage)
	}
	return res
}
