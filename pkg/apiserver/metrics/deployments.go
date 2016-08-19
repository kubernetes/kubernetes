/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"

	"golang.org/x/net/context"
)

var (
	variableLabels = []string{"name", "namespace"}

	deploymentStatusObservedGenerationDesc = prometheus.NewDesc(
		"kubernetes_deployment_status_observed_generation",
		"The observed generation of this deployment.",
		variableLabels,
		nil,
	)

	deploymentStatusReplicasDesc = prometheus.NewDesc(
		"kubernetes_deployment_status_replicas",
		"The number of replicas for the specific deployment.",
		variableLabels,
		nil,
	)

	deploymentStatusUpdatedReplicasDesc = prometheus.NewDesc(
		"kubernetes_deployment_status_updated_replicas",
		"The number of updated replicas for the specific deployment.",
		variableLabels,
		nil,
	)

	deploymentStatusAvailableReplicasDesc = prometheus.NewDesc(
		"kubernetes_deployment_status_available_replicas",
		"The number of available replicas for the specific deployment.",
		variableLabels,
		nil,
	)

	deploymentStatusUnavailableReplicasDesc = prometheus.NewDesc(
		"kubernetes_deployment_status_unavailable_replicas",
		"The number of unavailable replicas for the specific deployment.",
		variableLabels,
		nil,
	)
)

// DeploymentCollector collects various metrics of deployment objects.
type DeploymentCollector struct {
	Storage rest.StandardStorage
}

// Returns a new instance of DeploymentCollector.
func NewDeploymentsCollector(s rest.StandardStorage) *DeploymentCollector {
	return &DeploymentCollector{
		Storage: s,
	}
}

// Describe implements the prometheus.Collector interface.
func (c *DeploymentCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- deploymentStatusObservedGenerationDesc
	ch <- deploymentStatusReplicasDesc
	ch <- deploymentStatusUpdatedReplicasDesc
	ch <- deploymentStatusAvailableReplicasDesc
	ch <- deploymentStatusUnavailableReplicasDesc
}

// Collect implements the prometheus.Collector interface.
func (c *DeploymentCollector) Collect(ch chan<- prometheus.Metric) {
	var err error
	deployments, err := c.Storage.List(context.Background(), nil)
	if err != nil {
		glog.Errorf("Error while collecting deployment metrics: %s", err.Error())
	}

	for _, d := range deployments.(*extensions.DeploymentList).Items {
		err = addMetrics(ch, d)
		if err != nil {
			glog.Errorf("Error while collecting deployment metrics: %s", err.Error())
		}
	}
}

// addMetrics adds Metrics for a single deployment to the specified channel
func addMetrics(ch chan<- prometheus.Metric, d extensions.Deployment) error {
	metric, err := prometheus.NewConstMetric(
		deploymentStatusObservedGenerationDesc,
		prometheus.GaugeValue,
		float64(d.Status.ObservedGeneration),
		d.ObjectMeta.Name,
		d.ObjectMeta.Namespace,
	)
	if err != nil {
		return err
	}
	ch <- metric

	metric, err = prometheus.NewConstMetric(
		deploymentStatusReplicasDesc,
		prometheus.GaugeValue,
		float64(d.Status.Replicas),
		d.ObjectMeta.Name,
		d.ObjectMeta.Namespace,
	)
	if err != nil {
		return err
	}
	ch <- metric

	metric, err = prometheus.NewConstMetric(
		deploymentStatusUpdatedReplicasDesc,
		prometheus.GaugeValue,
		float64(d.Status.UpdatedReplicas),
		d.ObjectMeta.Name,
		d.ObjectMeta.Namespace,
	)
	if err != nil {
		return err
	}
	ch <- metric

	metric, err = prometheus.NewConstMetric(
		deploymentStatusAvailableReplicasDesc,
		prometheus.GaugeValue,
		float64(d.Status.AvailableReplicas),
		d.ObjectMeta.Name,
		d.ObjectMeta.Namespace,
	)
	if err != nil {
		return err
	}
	ch <- metric

	metric, err = prometheus.NewConstMetric(
		deploymentStatusUnavailableReplicasDesc,
		prometheus.GaugeValue,
		float64(d.Status.UnavailableReplicas),
		d.ObjectMeta.Name,
		d.ObjectMeta.Namespace,
	)
	if err != nil {
		return err
	}
	ch <- metric

	return nil
}
