/*
Copyright The Kubernetes Authors.

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
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/discovery"
	"k8s.io/klog/v2"
	metricsint "k8s.io/metrics/pkg/apis/metrics"
	metricsv1 "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsv1client "k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1"
	metricsv1beta1client "k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1"
)

// metricVersions is the set of supported resource metrics API versions, in
// descending order of preference. Unlike the aggregated discovery preferred
// version, this client-side ordering guarantees that v1 is used whenever the
// server serves it, even if the server still reports v1beta1 as its
// preferred version (e.g. due to legacy APIService priorities).
var metricVersions = []schema.GroupVersion{
	metricsv1.SchemeGroupVersion,
	metricsv1beta1.SchemeGroupVersion,
}

// VersionedPodMetricsGetter is a metricsv1beta1client.PodMetricsesGetter that works
// with whichever version of the resource metrics API (metrics.k8s.io) the
// server serves, so consumers keep working against v1beta1 types.
type VersionedPodMetricsGetter interface {
	metricsv1beta1client.PodMetricsesGetter
	// Invalidate clears the cached preferred API version, forcing the next
	// request to re-negotiate it via discovery.
	Invalidate()
}

// NewVersionedPodMetricsGetter creates a VersionedPodMetricsGetter that uses
// discovery to pick the most-preferred available version of the resource
// metrics API, and converts responses to v1beta1 for its consumers.
func NewVersionedPodMetricsGetter(v1Client metricsv1client.MetricsV1Interface, v1beta1Client metricsv1beta1client.MetricsV1beta1Interface, discoveryClient discovery.DiscoveryInterface) VersionedPodMetricsGetter {
	return &versionedPodMetricsGetter{
		v1Client:        v1Client,
		v1beta1Client:   v1beta1Client,
		discoveryClient: discoveryClient,
	}
}

// PeriodicallyInvalidate periodically invalidates the preferred version cache
// until the context is cancelled, so that a metrics server upgrade (or
// downgrade) is picked up without restarting the consumer.
func PeriodicallyInvalidate(ctx context.Context, getter VersionedPodMetricsGetter, interval time.Duration) {
	wait.UntilWithContext(ctx, func(_ context.Context) {
		getter.Invalidate()
	}, interval)
}

type versionedPodMetricsGetter struct {
	v1Client        metricsv1client.MetricsV1Interface
	v1beta1Client   metricsv1beta1client.MetricsV1beta1Interface
	discoveryClient discovery.DiscoveryInterface

	// cache the preferred version directly since the discovery interface
	// doesn't yet allow asking for a single API group's versions.
	prefVersion *schema.GroupVersion
	mu          sync.RWMutex
}

func (g *versionedPodMetricsGetter) PodMetricses(namespace string) metricsv1beta1client.PodMetricsInterface {
	return &versionedPodMetrics{
		getter:    g,
		namespace: namespace,
	}
}

func (g *versionedPodMetricsGetter) Invalidate() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.prefVersion = nil
}

// preferredVersion returns the current preferred version of the resource
// metrics API, negotiating it via discovery on the first call after an
// invalidation.
func (g *versionedPodMetricsGetter) preferredVersion(ctx context.Context) (schema.GroupVersion, error) {
	g.mu.RLock()
	if g.prefVersion != nil {
		// if we've already got one, proceed with that
		defer g.mu.RUnlock()
		return *g.prefVersion, nil
	}
	g.mu.RUnlock()

	g.mu.Lock()
	defer g.mu.Unlock()

	// double check, someone might have beaten us to it
	if g.prefVersion != nil {
		return *g.prefVersion, nil
	}

	prefVersion, err := g.fetchPreferredVersion(ctx)
	if err != nil {
		return schema.GroupVersion{}, err
	}
	klog.FromContext(ctx).V(4).Info("Negotiated resource metrics API version", "version", prefVersion)

	g.prefVersion = &prefVersion
	return *g.prefVersion, nil
}

// fetchPreferredVersion picks the most-preferred supported version from the
// ones the server actually serves.
func (g *versionedPodMetricsGetter) fetchPreferredVersion(ctx context.Context) (schema.GroupVersion, error) {
	groups, err := discovery.ToServerGroupsInterfaceWithContext(
		g.discoveryClient,
	).ServerGroupsWithContext(ctx)
	if err != nil {
		return schema.GroupVersion{}, err
	}

	var apiGroup *metav1.APIGroup
	for _, group := range groups.Groups {
		if group.Name == metricsint.GroupName {
			apiGroup = &group
			break
		}
	}
	if apiGroup == nil {
		return schema.GroupVersion{}, fmt.Errorf("no resource metrics API (%s) registered", metricsint.GroupName)
	}

	served := make(map[string]struct{}, len(apiGroup.Versions))
	for _, version := range apiGroup.Versions {
		served[version.GroupVersion] = struct{}{}
	}
	for _, gv := range metricVersions {
		if _, present := served[gv.String()]; present {
			return gv, nil
		}
	}

	return schema.GroupVersion{}, fmt.Errorf("no known available resource metrics API versions found")
}

// versionedPodMetrics implements the v1beta1 PodMetricsInterface on top of
// whichever API version was negotiated, converting v1 responses to v1beta1.
type versionedPodMetrics struct {
	getter    *versionedPodMetricsGetter
	namespace string
}

func (m *versionedPodMetrics) Get(ctx context.Context, name string, opts metav1.GetOptions) (*metricsv1beta1.PodMetrics, error) {
	version, err := m.getter.preferredVersion(ctx)
	if err != nil {
		return nil, err
	}
	if version == metricsv1.SchemeGroupVersion {
		metrics, err := m.getter.v1Client.PodMetricses(m.namespace).Get(ctx, name, opts)
		if err != nil {
			return nil, err
		}
		return convertV1PodMetrics(metrics)
	}
	return m.getter.v1beta1Client.PodMetricses(m.namespace).Get(ctx, name, opts)
}

func (m *versionedPodMetrics) List(ctx context.Context, opts metav1.ListOptions) (*metricsv1beta1.PodMetricsList, error) {
	version, err := m.getter.preferredVersion(ctx)
	if err != nil {
		return nil, err
	}
	if version == metricsv1.SchemeGroupVersion {
		metrics, err := m.getter.v1Client.PodMetricses(m.namespace).List(ctx, opts)
		if err != nil {
			return nil, err
		}
		return convertV1PodMetricsList(metrics)
	}
	return m.getter.v1beta1Client.PodMetricses(m.namespace).List(ctx, opts)
}

func (m *versionedPodMetrics) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	version, err := m.getter.preferredVersion(ctx)
	if err != nil {
		return nil, err
	}
	if version == metricsv1.SchemeGroupVersion {
		watcher, err := m.getter.v1Client.PodMetricses(m.namespace).Watch(ctx, opts)
		if err != nil {
			return nil, err
		}
		return watch.Filter(watcher, func(in watch.Event) (watch.Event, bool) {
			metrics, ok := in.Object.(*metricsv1.PodMetrics)
			if !ok {
				// pass through non-PodMetrics objects (e.g. Status on error events)
				return in, true
			}
			converted, err := convertV1PodMetrics(metrics)
			if err != nil {
				return in, false
			}
			in.Object = converted
			return in, true
		}), nil
	}
	return m.getter.v1beta1Client.PodMetricses(m.namespace).Watch(ctx, opts)
}

func convertV1PodMetrics(in *metricsv1.PodMetrics) (*metricsv1beta1.PodMetrics, error) {
	internal := &metricsint.PodMetrics{}
	if err := metricsv1.Convert_v1_PodMetrics_To_metrics_PodMetrics(in, internal, nil); err != nil {
		return nil, fmt.Errorf("failed to convert pod metrics to the internal version: %w", err)
	}
	out := &metricsv1beta1.PodMetrics{}
	if err := metricsv1beta1.Convert_metrics_PodMetrics_To_v1beta1_PodMetrics(internal, out, nil); err != nil {
		return nil, fmt.Errorf("failed to convert pod metrics to v1beta1: %w", err)
	}
	return out, nil
}

func convertV1PodMetricsList(in *metricsv1.PodMetricsList) (*metricsv1beta1.PodMetricsList, error) {
	internal := &metricsint.PodMetricsList{}
	if err := metricsv1.Convert_v1_PodMetricsList_To_metrics_PodMetricsList(in, internal, nil); err != nil {
		return nil, fmt.Errorf("failed to convert pod metrics list to the internal version: %w", err)
	}
	out := &metricsv1beta1.PodMetricsList{}
	if err := metricsv1beta1.Convert_metrics_PodMetricsList_To_v1beta1_PodMetricsList(internal, out, nil); err != nil {
		return nil, fmt.Errorf("failed to convert pod metrics list to v1beta1: %w", err)
	}
	return out, nil
}
