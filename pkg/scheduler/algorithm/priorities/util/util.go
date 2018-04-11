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

package util

import (
	"fmt"
	"github.com/golang/glog"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/metrics/pkg/apis/metrics/v1beta1"
	resourceclient "k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1beta1"
)

// NodeMetricsInfo contains node metric values as a map from node names to metric values (the metric values are expected
// to be the metric as a milli-value). As of now, we are focusing on CPU.
type NodeMetricsInfo map[string]int64

// MetricsClient knows how to query a remote interface to retrieve node-level resource metrics.
type MetricsClient interface {
	// GetResourceMetric gets the given resource metric (and an associated oldest timestamp)
	// for all nodes in the cluster.
	GetResourceMetric() (NodeMetricsInfo, time.Time, error)
}

// resourceMetricsClient contains a client to NodeMetricsGetter.
type resourceMetricsClient struct {
	client resourceclient.NodeMetricsesGetter
}

// restMetricsClient is a client which supports fetching metrics from both the resource metrics API. This could be
// extended for custom metrics as well.
type restMetricsClient struct {
	*resourceMetricsClient
}

// GetControllerRef gets pod's owner controller reference from a pod object.
func GetControllerRef(pod *v1.Pod) *metav1.OwnerReference {
	if len(pod.OwnerReferences) == 0 {
		return nil
	}
	for i := range pod.OwnerReferences {
		ref := &pod.OwnerReferences[i]
		if ref.Controller != nil && *ref.Controller {
			return ref
		}
	}
	return nil
}

// NewRESTMetricsClient returns a metricsClient.
func NewRESTMetricsClient(resourceClient resourceclient.NodeMetricsesGetter) MetricsClient {
	return &restMetricsClient{
		&resourceMetricsClient{resourceClient},
	}
}

// populateNodeMetricsInfo returns a map of nodes with their CPU usages.
func populateNodeMetricsInfo(metrics *v1beta1.NodeMetricsList) NodeMetricsInfo {
	nodeMetrics := NodeMetricsInfo{}
	for _, m := range metrics.Items {
		nodeMetrics[m.Name] = m.Usage.Cpu().MilliValue()
	}
	return nodeMetrics
}

// GetResourceMetric gets the given resource metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (c *resourceMetricsClient) GetResourceMetric() (NodeMetricsInfo, time.Time, error) {
	metrics, err := c.client.NodeMetricses().List(metav1.ListOptions{})
	if err != nil {
		glog.V(5).Infof("unable to fetch metrics from API: %v", err)
		return nil, time.Time{}, fmt.Errorf("unable to fetch metrics from API: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from metric-server")
	}
	nodeMetricsInfo := populateNodeMetricsInfo(metrics)
	timestamp := metrics.Items[0].Timestamp.Time
	return nodeMetricsInfo, timestamp, nil
}
