/*
Copyright 2018 The Kubernetes Authors.

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

package metrics

import (
	"sync"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/apimachinery/pkg/labels"
	apiregistration "k8s.io/kube-aggregator/pkg/apis/apiregistration"
	listers "k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion"
)

var (
	descAPIServiceLabelsDefaultLabels = []string{"apiservice"}

	descAPIServiceStatusCondition = prometheus.NewDesc(
		"aggregator_apiservice_status_condition",
		"The condition of an apiservice",
		append(descAPIServiceLabelsDefaultLabels, "condition", "status"),
		nil,
	)
)

// apiServiceCollector collects metrics about all apiservices in the cluster.
type apiServiceCollector struct {
	apiServiceLister listers.APIServiceLister
}

// Describe implements the prometheus.Collector interface.
func (ac *apiServiceCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descAPIServiceStatusCondition
}

// Collect implements the prometheus.Collector interface.
func (ac *apiServiceCollector) Collect(ch chan<- prometheus.Metric) {
	apiServices, err := ac.apiServiceLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("listing apiservices failed: %s", err)
		return
	}
	for _, apiService := range apiServices {
		ac.collectAPIService(ch, apiService)
	}
	glog.V(4).Infof("collected %d apiservices", len(apiServices))
}

func (ac *apiServiceCollector) collectAPIService(ch chan<- prometheus.Metric, apiService *apiregistration.APIService) {
	// Collect apiservice conditions
	for _, c := range apiService.Status.Conditions {
		addConditionMetrics(ch, descAPIServiceStatusCondition, c.Status, apiService.Name, string(c.Type))
	}
}

// addConditionMetrics generates one metric for each possible apiservice condition
// status. For this function to work properly, the last label in the metric
// description must be the condition type.
func addConditionMetrics(ch chan<- prometheus.Metric, desc *prometheus.Desc, cs apiregistration.ConditionStatus, lv ...string) {
	ch <- prometheus.MustNewConstMetric(
		desc, prometheus.GaugeValue, boolFloat64(cs == apiregistration.ConditionTrue),
		append(lv, "true")...,
	)
	ch <- prometheus.MustNewConstMetric(
		desc, prometheus.GaugeValue, boolFloat64(cs == apiregistration.ConditionFalse),
		append(lv, "false")...,
	)
	ch <- prometheus.MustNewConstMetric(
		desc, prometheus.GaugeValue, boolFloat64(cs == apiregistration.ConditionUnknown),
		append(lv, "unknown")...,
	)
}

func boolFloat64(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

var registerAPIServiceCollector sync.Once

// RegisterAPIServiceCollector registers a collector once.
func RegisterAPIServiceCollector(apiServiceLister listers.APIServiceLister) {
	registerAPIServiceCollector.Do(func() {
		prometheus.MustRegister(&apiServiceCollector{apiServiceLister: apiServiceLister})
	})
}
