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

package custom_metrics

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
)

// AvailableAPIsGetter knows how to fetch and cache the preferred custom metrics API version,
// and invalidate that cache when asked.
type AvailableAPIsGetter interface {
	PreferredVersion() (schema.GroupVersion, error)
	Invalidate()
}

// PeriodicallyInvalidate periodically invalidates the preferred version cache until
// told to stop.
func PeriodicallyInvalidate(cache AvailableAPIsGetter, interval time.Duration, stopCh <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			cache.Invalidate()
		case <-stopCh:
			return
		}
	}
}

// NewForConfig creates a new custom metrics client which delegates to a client which
// uses the preferred api version.
func NewForConfig(baseConfig *rest.Config, mapper meta.RESTMapper, availableAPIs AvailableAPIsGetter) CustomMetricsClient {
	return &multiClient{
		clients:       make(map[schema.GroupVersion]CustomMetricsClient),
		availableAPIs: availableAPIs,

		newClient: func(ver schema.GroupVersion) (CustomMetricsClient, error) {
			return NewForVersionForConfig(rest.CopyConfig(baseConfig), mapper, ver)
		},
	}
}

// multiClient is a CustomMetricsClient that can work with *any* metrics API version.
type multiClient struct {
	newClient     func(schema.GroupVersion) (CustomMetricsClient, error)
	clients       map[schema.GroupVersion]CustomMetricsClient
	availableAPIs AvailableAPIsGetter
	mu            sync.RWMutex
}

// getPreferredClient returns a custom metrics client of the preferred api version.
func (c *multiClient) getPreferredClient() (CustomMetricsClient, error) {
	pref, err := c.availableAPIs.PreferredVersion()
	if err != nil {
		return nil, err
	}

	c.mu.RLock()
	client, present := c.clients[pref]
	c.mu.RUnlock()
	if present {
		return client, nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	client, err = c.newClient(pref)
	if err != nil {
		return nil, err
	}
	c.clients[pref] = client

	return client, nil
}

func (c *multiClient) RootScopedMetrics() MetricsInterface {
	return &multiClientInterface{clients: c}
}

func (c *multiClient) NamespacedMetrics(namespace string) MetricsInterface {
	return &multiClientInterface{
		clients:   c,
		namespace: &namespace,
	}
}

type multiClientInterface struct {
	clients   *multiClient
	namespace *string
}

func (m *multiClientInterface) GetForObject(groupKind schema.GroupKind, name string, metricName string, metricSelector labels.Selector) (*v1beta2.MetricValue, error) {
	client, err := m.clients.getPreferredClient()
	if err != nil {
		return nil, err
	}
	if m.namespace == nil {
		return client.RootScopedMetrics().GetForObject(groupKind, name, metricName, metricSelector)
	} else {
		return client.NamespacedMetrics(*m.namespace).GetForObject(groupKind, name, metricName, metricSelector)
	}
}

func (m *multiClientInterface) GetForObjects(groupKind schema.GroupKind, selector labels.Selector, metricName string, metricSelector labels.Selector) (*v1beta2.MetricValueList, error) {
	client, err := m.clients.getPreferredClient()
	if err != nil {
		return nil, err
	}
	if m.namespace == nil {
		return client.RootScopedMetrics().GetForObjects(groupKind, selector, metricName, metricSelector)
	} else {
		return client.NamespacedMetrics(*m.namespace).GetForObjects(groupKind, selector, metricName, metricSelector)
	}
}
