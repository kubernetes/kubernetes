/*
Copyright 2025 The Kubernetes Authors.

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

package collectors

import (
	"k8s.io/component-base/metrics"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/podcertificate"
)

var (
	// A gauge vector (implemented by a custom collector) reporting the current
	// number of pod certificate projected volume sources being maintained by
	// this kubelet instance.
	podCertificateStatesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.PodCertificateStatesKey),
		"Gauge vector reporting the number of pod certificate projected volume sources, faceted by signer_name and state.",
		[]string{"signer_name", "state"},
		nil,
		metrics.ALPHA,
		"",
	)
)

type podCertificateCollector struct {
	metrics.BaseStableCollector
	manager podcertificate.Manager
}

func PodCertificateCollectorFor(m podcertificate.Manager) *podCertificateCollector {
	return &podCertificateCollector{
		manager: m,
	}
}

func (c *podCertificateCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- podCertificateStatesDesc
}

func (c *podCertificateCollector) CollectWithStability(ch chan<- metrics.Metric) {
	report := c.manager.MetricReport()

	for k, count := range report.PodCertificateStates {
		ch <- metrics.NewLazyConstMetric(
			podCertificateStatesDesc,
			metrics.GaugeValue,
			float64(count),
			k.SignerName,
			k.State,
		)
	}
}
