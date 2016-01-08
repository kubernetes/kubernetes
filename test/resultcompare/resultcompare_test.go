/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package resultcompare

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/test/e2e"
)

func TestLogCompareSuccess(t *testing.T) {
	left := make(e2e.ResourceUsageSummary)
	left[90] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-2r6la/heapster", Cpu: 0.266, Mem: 846700000},
		{Name: "kibana-logging-v1-041ex/kibana-logging", Cpu: 0.000, Mem: 61200000},
		{Name: "kube-dns-v10-udc0j/etcd", Cpu: 0.008, Mem: 18530000},
		{Name: "kube-dns-v10-udc0j/healthz", Cpu: 0.001, Mem: 950000},
		{Name: "kube-dns-v10-udc0j/kube2sky", Cpu: 0.005, Mem: 37810000},
		{Name: "kube-dns-v10-udc0j/skydns", Cpu: 0.019, Mem: 3290000},
		{Name: "kube-proxy-e2e-scalability-minion-0ana/kube-proxy", Cpu: 0.001, Mem: 4430000},
		{Name: "kube-proxy-e2e-scalability-minion-0eu6/kube-proxy", Cpu: 0.000, Mem: 4460000},
	}
	left[99] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-2r6la/heapster", Cpu: 0.729, Mem: 1333160000},
		{Name: "kibana-logging-v1-041ex/kibana-logging", Cpu: 0.000, Mem: 61200000},
		{Name: "kube-dns-v10-udc0j/etcd", Cpu: 0.028, Mem: 24640000},
		{Name: "kube-dns-v10-udc0j/healthz", Cpu: 0.002, Mem: 970000},
		{Name: "kube-dns-v10-udc0j/kube2sky", Cpu: 0.041, Mem: 48220000},
		{Name: "kube-dns-v10-udc0j/skydns", Cpu: 0.023, Mem: 3300000},
		{Name: "kube-proxy-e2e-scalability-minion-0ana/kube-proxy", Cpu: 0.002, Mem: 4480000},
		{Name: "kube-proxy-e2e-scalability-minion-0eu6/kube-proxy", Cpu: 0.002, Mem: 4490000},
	}

	right := make(e2e.ResourceUsageSummary)
	right[90] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-a7j0p/heapster", Cpu: 0.354, Mem: 1256930000},
		{Name: "kibana-logging-v1-9qqsp/kibana-logging", Cpu: 0.000, Mem: 61700000},
		{Name: "kube-dns-v10-p15su/etcd", Cpu: 0.008, Mem: 17050000},
		{Name: "kube-dns-v10-p15su/healthz", Cpu: 0.001, Mem: 930000},
		{Name: "kube-dns-v10-p15su/kube2sky", Cpu: 0.004, Mem: 28330000},
		{Name: "kube-dns-v10-p15su/skydns", Cpu: 0.022, Mem: 3340000},
		{Name: "kube-proxy-e2e-scalability-minion-0490/kube-proxy", Cpu: 0.001, Mem: 4420000},
		{Name: "kube-proxy-e2e-scalability-minion-0n32/kube-proxy", Cpu: 0.001, Mem: 4440000},
	}
	right[99] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-a7j0p/heapster", Cpu: 0.772, Mem: 1398390000},
		{Name: "kibana-logging-v1-9qqsp/kibana-logging", Cpu: 0.000, Mem: 61700000},
		{Name: "kube-dns-v10-p15su/etcd", Cpu: 0.030, Mem: 21900000},
		{Name: "kube-dns-v10-p15su/healthz", Cpu: 0.001, Mem: 980000},
		{Name: "kube-dns-v10-p15su/kube2sky", Cpu: 0.045, Mem: 43860000},
		{Name: "kube-dns-v10-p15su/skydns", Cpu: 0.028, Mem: 3350000},
		{Name: "kube-proxy-e2e-scalability-minion-0490/kube-proxy", Cpu: 0.002, Mem: 4450000},
		{Name: "kube-proxy-e2e-scalability-minion-0n32/kube-proxy", Cpu: 0.002, Mem: 4490000},
	}

	if violating := compareResourceUsages(left, right); violating != nil {
		t.Errorf("Expected compare to return true, got violating containers: %v", violating)
	}
}

func TestLogCompareDifferentSizes(t *testing.T) {
	left := make(e2e.ResourceUsageSummary)
	left[90] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-2r6la/heapster", Cpu: 0.266, Mem: 846700000},
		{Name: "kibana-logging-v1-041ex/kibana-logging", Cpu: 0.000, Mem: 61200000},
		{Name: "kube-dns-v10-udc0j/etcd", Cpu: 0.008, Mem: 18530000},
		{Name: "kube-dns-v10-udc0j/healthz", Cpu: 0.001, Mem: 950000},
		{Name: "kube-dns-v10-udc0j/kube2sky", Cpu: 0.005, Mem: 37810000},
		{Name: "kube-dns-v10-udc0j/skydns", Cpu: 0.019, Mem: 3290000},
		{Name: "kube-proxy-e2e-scalability-minion-0ana/kube-proxy", Cpu: 0.001, Mem: 4430000},
		{Name: "kube-proxy-e2e-scalability-minion-0eu6/kube-proxy", Cpu: 0.000, Mem: 4460000},
	}
	left[99] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-2r6la/heapster", Cpu: 0.729, Mem: 1333160000},
		{Name: "kibana-logging-v1-041ex/kibana-logging", Cpu: 0.000, Mem: 61200000},
		{Name: "kube-dns-v10-udc0j/etcd", Cpu: 0.028, Mem: 24640000},
		{Name: "kube-dns-v10-udc0j/healthz", Cpu: 0.002, Mem: 970000},
		{Name: "kube-dns-v10-udc0j/kube2sky", Cpu: 0.041, Mem: 48220000},
		{Name: "kube-dns-v10-udc0j/skydns", Cpu: 0.023, Mem: 3300000},
		{Name: "kube-proxy-e2e-scalability-minion-0ana/kube-proxy", Cpu: 0.002, Mem: 4480000},
		{Name: "kube-proxy-e2e-scalability-minion-0eu6/kube-proxy", Cpu: 0.002, Mem: 4490000},
	}

	right := make(e2e.ResourceUsageSummary)
	right[90] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-a7j0p/heapster", Cpu: 0.354, Mem: 1256930000},
		{Name: "kibana-logging-v1-9qqsp/kibana-logging", Cpu: 0.000, Mem: 61700000},
		{Name: "kube-dns-v10-p15su/etcd", Cpu: 0.008, Mem: 17050000},
		{Name: "kube-dns-v10-p15su/healthz", Cpu: 0.001, Mem: 930000},
		{Name: "kube-dns-v10-p15su/kube2sky", Cpu: 0.004, Mem: 28330000},
		{Name: "kube-dns-v10-p15su/skydns", Cpu: 0.022, Mem: 3340000},
		{Name: "kube-proxy-e2e-scalability-minion-0490/kube-proxy", Cpu: 0.001, Mem: 4420000},
		{Name: "kube-proxy-e2e-scalability-minion-0n32/kube-proxy", Cpu: 0.001, Mem: 4440000},
	}

	if violating := compareResourceUsages(left, right); violating != nil {
		t.Errorf("Expected compare to return true, got violating containers: %v", violating)
	}

	if violating := compareResourceUsages(right, left); violating != nil {
		t.Errorf("Expected compare to return true, got violating containers: %v", violating)
	}
}

func TestLogCompareFailure(t *testing.T) {
	left := make(e2e.ResourceUsageSummary)
	left[90] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-2r6la/heapster", Cpu: 0.266, Mem: 846700000},
		{Name: "kibana-logging-v1-041ex/kibana-logging", Cpu: 0.000, Mem: 61200000},
		{Name: "kube-dns-v10-udc0j/etcd", Cpu: 0.18, Mem: 18530000},
		{Name: "kube-dns-v10-udc0j/healthz", Cpu: 0.001, Mem: 950000},
		{Name: "kube-dns-v10-udc0j/kube2sky", Cpu: 0.005, Mem: 37810000},
		{Name: "kube-dns-v10-udc0j/skydns", Cpu: 0.019, Mem: 3290000},
		{Name: "kube-proxy-e2e-scalability-minion-0ana/kube-proxy", Cpu: 0.001, Mem: 4430000},
		{Name: "kube-proxy-e2e-scalability-minion-0eu6/kube-proxy", Cpu: 0.000, Mem: 4460000},
	}

	right := make(e2e.ResourceUsageSummary)
	right[90] = []e2e.SingleContainerSummary{
		{Name: "heapster-v10-a7j0p/heapster", Cpu: 0.354, Mem: 1256930000},
		{Name: "kibana-logging-v1-9qqsp/kibana-logging", Cpu: 0.000, Mem: 61700000},
		{Name: "kube-dns-v10-p15su/etcd", Cpu: 0.008, Mem: 17050000},
		{Name: "kube-dns-v10-p15su/healthz", Cpu: 0.001, Mem: 930000},
		{Name: "kube-dns-v10-p15su/kube2sky", Cpu: 0.004, Mem: 28330000},
		{Name: "kube-dns-v10-p15su/skydns", Cpu: 0.022, Mem: 3340000},
		{Name: "kube-proxy-e2e-scalability-minion-0490/kube-proxy", Cpu: 0.001, Mem: 4420000},
		{Name: "kube-proxy-e2e-scalability-minion-0n32/kube-proxy", Cpu: 0.001, Mem: 4440000},
	}
	expectedViolating := map[string]ViolatingDataPair{
		"etcd": {
			perc:         90,
			leftCpuData:  []float64{0.18},
			leftMemData:  []int64{18530000},
			rightCpuData: []float64{0.008},
			rightMemData: []int64{17050000},
		},
	}

	if violating := compareResourceUsages(left, right); violating == nil {
		t.Errorf("Expected compare to return non empty list of violating results")
	} else if !reflect.DeepEqual(violating, expectedViolating) {
		t.Errorf("Expected compare to see violating containers:\n%v,\ngot:\n%v", expectedViolating, violating)
	}
}
