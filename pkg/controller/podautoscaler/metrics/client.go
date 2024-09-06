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
	"context"
	"fmt"
	"time"

	"k8s.io/klog/v2"

	autoscaling "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	customapi "k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	resourceclient "k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1"
	customclient "k8s.io/metrics/pkg/client/custom_metrics"
	externalclient "k8s.io/metrics/pkg/client/external_metrics"
)

const (
	metricServerDefaultMetricWindow = time.Minute
)

func NewRESTMetricsClient(resourceClient resourceclient.PodMetricsesGetter, customClient customclient.CustomMetricsClient, externalClient externalclient.ExternalMetricsClient) MetricsClient {
	return &restMetricsClient{
		&resourceMetricsClient{resourceClient},
		&customMetricsClient{customClient},
		&externalMetricsClient{externalClient},
	}
}

// restMetricsClient is a client which supports fetching
// metrics from the resource metrics API, the
// custom metrics API and the external metrics API.
type restMetricsClient struct {
	*resourceMetricsClient
	*customMetricsClient
	*externalMetricsClient
}

// resourceMetricsClient implements the resource-metrics-related parts of MetricsClient,
// using data from the resource metrics API.
type resourceMetricsClient struct {
	client resourceclient.PodMetricsesGetter
}

// GetResourceMetric gets the given resource metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (c *resourceMetricsClient) GetResourceMetric(ctx context.Context, resource v1.ResourceName, namespace string, selector labels.Selector, container string) (PodMetricsInfo, time.Time, error) {
	metrics, err := c.client.PodMetricses(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("unable to fetch metrics from resource metrics API: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from resource metrics API")
	}
	var res PodMetricsInfo
	if container != "" {
		res = getContainerMetrics(ctx, metrics.Items, resource, container)
	} else {
		res = getPodMetrics(ctx, metrics.Items, resource)
	}
	timestamp := metrics.Items[0].Timestamp.Time
	return res, timestamp, nil
}

func getContainerMetrics(ctx context.Context, rawMetrics []metricsapi.PodMetrics, resource v1.ResourceName, container string) PodMetricsInfo {
	res := make(PodMetricsInfo, len(rawMetrics))
	for _, m := range rawMetrics {
		containerFound := false
		for _, c := range m.Containers {
			if c.Name == container {
				containerFound = true
				if val, resFound := c.Usage[resource]; resFound {
					res[m.Name] = PodMetric{
						Timestamp: m.Timestamp.Time,
						Window:    m.Window.Duration,
						Value:     val.MilliValue(),
					}
				}
				break
			}
		}
		if !containerFound {
			klog.FromContext(ctx).V(2).Info("Missing container metric", "container", container, "pod", klog.KRef(m.Namespace, m.Name))
		}
	}
	return res
}

func getPodMetrics(ctx context.Context, rawMetrics []metricsapi.PodMetrics, resource v1.ResourceName) PodMetricsInfo {
	res := make(PodMetricsInfo, len(rawMetrics))
	for _, m := range rawMetrics {
		podSum := int64(0)
		missing := len(m.Containers) == 0
		for _, c := range m.Containers {
			resValue, found := c.Usage[resource]
			if !found {
				missing = true
				klog.FromContext(ctx).V(2).Info("Missing resource metric", "resourceMetric", resource, "pod", klog.KRef(m.Namespace, m.Name))
				break
			}
			podSum += resValue.MilliValue()
		}
		if !missing {
			res[m.Name] = PodMetric{
				Timestamp: m.Timestamp.Time,
				Window:    m.Window.Duration,
				Value:     podSum,
			}
		}
	}
	return res
}

// customMetricsClient implements the custom-metrics-related parts of MetricsClient,
// using data from the custom metrics API.
type customMetricsClient struct {
	client customclient.CustomMetricsClient
}

// GetRawMetric gets the given metric (and an associated oldest timestamp)
// for all pods matching the specified selector in the given namespace
func (c *customMetricsClient) GetRawMetric(metricName string, namespace string, selector labels.Selector, metricSelector labels.Selector) (PodMetricsInfo, time.Time, error) {
	metrics, err := c.client.NamespacedMetrics(namespace).GetForObjects(schema.GroupKind{Kind: "Pod"}, selector, metricName, metricSelector)
	if err != nil {
		return nil, time.Time{}, fmt.Errorf("unable to fetch metrics from custom metrics API: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from custom metrics API")
	}

	res := make(PodMetricsInfo, len(metrics.Items))
	for _, m := range metrics.Items {
		window := metricServerDefaultMetricWindow
		if m.WindowSeconds != nil {
			window = time.Duration(*m.WindowSeconds) * time.Second
		}
		res[m.DescribedObject.Name] = PodMetric{
			Timestamp: m.Timestamp.Time,
			Window:    window,
			Value:     int64(m.Value.MilliValue()),
		}

		m.Value.MilliValue()
	}

	timestamp := metrics.Items[0].Timestamp.Time

	return res, timestamp, nil
}

// GetObjectMetric gets the given metric (and an associated timestamp) for the given
// object in the given namespace
func (c *customMetricsClient) GetObjectMetric(metricName string, namespace string, objectRef *autoscaling.CrossVersionObjectReference, metricSelector labels.Selector) (int64, time.Time, error) {
	gvk := schema.FromAPIVersionAndKind(objectRef.APIVersion, objectRef.Kind)
	var metricValue *customapi.MetricValue
	var err error
	if gvk.Kind == "Namespace" && gvk.Group == "" {
		// handle namespace separately
		// NB: we ignore namespace name here, since CrossVersionObjectReference isn't
		// supposed to allow you to escape your namespace
		metricValue, err = c.client.RootScopedMetrics().GetForObject(gvk.GroupKind(), namespace, metricName, metricSelector)
	} else {
		metricValue, err = c.client.NamespacedMetrics(namespace).GetForObject(gvk.GroupKind(), objectRef.Name, metricName, metricSelector)
	}

	if err != nil {
		return 0, time.Time{}, fmt.Errorf("unable to fetch metrics from custom metrics API: %v", err)
	}

	return metricValue.Value.MilliValue(), metricValue.Timestamp.Time, nil
}

// externalMetricsClient implements the external metrics related parts of MetricsClient,
// using data from the external metrics API.
type externalMetricsClient struct {
	client externalclient.ExternalMetricsClient
}

// GetExternalMetric gets all the values of a given external metric
// that match the specified selector.
func (c *externalMetricsClient) GetExternalMetric(metricName, namespace string, selector labels.Selector) ([]int64, time.Time, error) {
	metrics, err := c.client.NamespacedMetrics(namespace).List(metricName, selector)
	if err != nil {
		return []int64{}, time.Time{}, fmt.Errorf("unable to fetch metrics from external metrics API: %v", err)
	}

	if len(metrics.Items) == 0 {
		return nil, time.Time{}, fmt.Errorf("no metrics returned from external metrics API")
	}

	res := make([]int64, 0)
	for _, m := range metrics.Items {
		res = append(res, m.Value.MilliValue())
	}
	timestamp := metrics.Items[0].Timestamp.Time
	return res, timestamp, nil
}
