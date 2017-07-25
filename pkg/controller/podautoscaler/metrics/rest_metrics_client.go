/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"time"

	"github.com/golang/glog"

	autoscaling "k8s.io/api/autoscaling/v2alpha1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	customapi "k8s.io/metrics/pkg/apis/custom_metrics/v1alpha1"
	resourceclient "k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1"
	customclient "k8s.io/metrics/pkg/client/custom_metrics"
)

func NewRESTMetricsClient(resourceClient resourceclient.PodMetricsesGetter, customClient customclient.CustomMetricsClient) MetricsClient {
	return &restMetricsClient{
		&resourceMetricsClient{resourceClient},
		&customMetricsClient{customClient},
	}
}

// restMetricsClient is a client which supports fetching
// metrics from both the resource metrics API and the
// custom metrics API.
type restMetricsClient struct {
	*resourceMetricsClient
	*customMetricsClient
}

// resourceMetricsClient implements the resource-metrics-related parts of MetricsClient,
// using data from the resource metrics API.
type resourceMetricsClient struct {
	client resourceclient.PodMetricsesGetter
}

// GetResourceMetric gets the given resource metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (c *resourceMetricsClient) GetResourceMetric(resource v1.ResourceName, namespace string, selector labels.Selector) (PodMetricsInfo, time.Time, error) {
	metrics, err := c.client.PodMetricses(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("unable to fetch metrics from API: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from heapster")
	}

	res := make(PodMetricsInfo, len(metrics.Items))

	for _, m := range metrics.Items {
		podSum := int64(0)
		missing := len(m.Containers) == 0
		for _, c := range m.Containers {
			resValue, found := c.Usage[v1.ResourceName(resource)]
			if !found {
				missing = true
				glog.V(2).Infof("missing resource metric %v for container %s in pod %s/%s", resource, c.Name, namespace, m.Name)
				break // containers loop
			}
			podSum += resValue.MilliValue()
		}

		if !missing {
			res[m.Name] = int64(podSum)
		}
	}

	timestamp := metrics.Items[0].Timestamp.Time

	return res, timestamp, nil
}

// customMetricsClient implements the custom-metrics-related parts of MetricsClient,
// using data from the custom metrics API.
type customMetricsClient struct {
	client customclient.CustomMetricsClient
}

// GetRawMetric gets the given metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (c *customMetricsClient) GetRawMetric(metricName string, namespace string, selector labels.Selector) (PodMetricsInfo, time.Time, error) {
	metrics, err := c.client.NamespacedMetrics(namespace).GetForObjects(schema.GroupKind{Kind: "Pod"}, selector, metricName)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("unable to fetch metrics from API: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from custom metrics API")
	}

	res := make(PodMetricsInfo, len(metrics.Items))
	for _, m := range metrics.Items {
		res[m.DescribedObject.Name] = m.Value.MilliValue()
	}

	timestamp := metrics.Items[0].Timestamp.Time

	return res, timestamp, nil
}

// GetObjectMetric gets the given metric (and an associated timestamp) for the given
// object in the given namespace
func (c *customMetricsClient) GetObjectMetric(metricName string, namespace string, objectRef *autoscaling.CrossVersionObjectReference) (int64, time.Time, error) {
	gvk := schema.FromAPIVersionAndKind(objectRef.APIVersion, objectRef.Kind)
	var metricValue *customapi.MetricValue
	var err error
	if gvk.Kind == "Namespace" && gvk.Group == "" {
		// handle namespace separately
		// NB: we ignore namespace name here, since CrossVersionObjectReference isn't
		// supposed to allow you to escape your namespace
		metricValue, err = c.client.RootScopedMetrics().GetForObject(gvk.GroupKind(), namespace, metricName)
	} else {
		metricValue, err = c.client.NamespacedMetrics(namespace).GetForObject(gvk.GroupKind(), objectRef.Name, metricName)
	}

	if err != nil {
		return 0, time.Time{}, fmt.Errorf("unable to fetch metrics from API: %v", err)
	}

	return metricValue.Value.MilliValue(), metricValue.Timestamp.Time, nil
}
