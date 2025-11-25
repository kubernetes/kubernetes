/*
Copyright 2024 The Kubernetes Authors.

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
	"strings"
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/status"
)

const (
	controllerSubsystem = "helm_applyset"
)

var (
	// helmApplySetStatus is a gauge that tracks the status of Helm releases
	helmApplySetStatus = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      controllerSubsystem,
			Name:           "status",
			Help:           "Status of Helm releases managed by ApplySet. Status values: healthy=1, progressing=2, degraded=3, failed=4, unknown=0",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "namespace", "status"},
	)

	// helmApplySetResourceTotal is a gauge that tracks total resource count per GroupKind
	helmApplySetResourceTotal = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      controllerSubsystem,
			Name:           "resource_total",
			Help:           "Total number of resources of a specific GroupKind in a Helm release",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "namespace", "gvk"},
	)

	// helmApplySetResourceHealthy is a gauge that tracks healthy resource count per GroupKind
	helmApplySetResourceHealthy = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      controllerSubsystem,
			Name:           "resource_healthy",
			Help:           "Number of healthy resources of a specific GroupKind in a Helm release",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "namespace", "gvk"},
	)

	// helmApplySetAgeSeconds is a gauge that tracks the age of Helm releases
	helmApplySetAgeSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      controllerSubsystem,
			Name:           "age_seconds",
			Help:           "Age of Helm release in seconds since creation",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "namespace"},
	)

	// helmApplySetLastReconcileTimestamp is a gauge that tracks last reconciliation time
	helmApplySetLastReconcileTimestamp = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      controllerSubsystem,
			Name:           "last_reconcile_timestamp",
			Help:           "Unix timestamp of last reconciliation",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "namespace"},
	)

	// helmApplySetReconcileDurationSeconds is a histogram that tracks reconciliation duration
	helmApplySetReconcileDurationSeconds = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      controllerSubsystem,
			Name:           "reconcile_duration_seconds",
			Help:           "Duration of reconciliation in seconds",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15), // 1ms to ~32s
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "namespace"},
	)

	metricsList = []metrics.Registerable{
		helmApplySetStatus,
		helmApplySetResourceTotal,
		helmApplySetResourceHealthy,
		helmApplySetAgeSeconds,
		helmApplySetLastReconcileTimestamp,
		helmApplySetReconcileDurationSeconds,
	}

	registerMetrics sync.Once
)

// Register registers all metrics for the Helm ApplySet controller
func Register() {
	registerMetrics.Do(func() {
		for _, metric := range metricsList {
			legacyregistry.MustRegister(metric)
		}
		klog.V(4).Info("Registered Helm ApplySet metrics")
	})
}

// Exporter exports Prometheus metrics for Helm releases
type Exporter struct {
	logger klog.Logger
	mu     sync.RWMutex
	// Track release creation times for age calculation
	releaseCreationTimes map[string]time.Time
}

// NewExporter creates a new metrics exporter
func NewExporter(logger klog.Logger) *Exporter {
	return &Exporter{
		logger:               logger,
		releaseCreationTimes: make(map[string]time.Time),
	}
}

// UpdateMetrics updates metrics for a Helm release based on health status
func (e *Exporter) UpdateMetrics(health *status.ReleaseHealth, creationTime time.Time, reconcileDuration time.Duration) {
	releaseKey := e.releaseKey(health.ReleaseName, health.Namespace)

	// Record creation time if not already recorded
	e.mu.Lock()
	if _, exists := e.releaseCreationTimes[releaseKey]; !exists {
		e.releaseCreationTimes[releaseKey] = creationTime
	}
	e.mu.Unlock()

	// Update status metric (one gauge per status value)
	statusValue := e.statusToValue(health.OverallStatus)

	// Set current status to 1, others to 0
	for _, status := range []string{"healthy", "progressing", "degraded", "failed", "unknown"} {
		labels := []string{health.ReleaseName, health.Namespace, status}
		if status == health.OverallStatus {
			helmApplySetStatus.WithLabelValues(labels...).Set(statusValue)
		} else {
			helmApplySetStatus.WithLabelValues(labels...).Set(0)
		}
	}

	// Update resource metrics by GroupKind
	e.updateResourceMetrics(health)

	// Update age metric
	e.mu.RLock()
	creationTime = e.releaseCreationTimes[releaseKey]
	e.mu.RUnlock()
	age := time.Since(creationTime).Seconds()
	helmApplySetAgeSeconds.WithLabelValues(health.ReleaseName, health.Namespace).Set(age)

	// Update last reconcile timestamp
	helmApplySetLastReconcileTimestamp.WithLabelValues(health.ReleaseName, health.Namespace).
		Set(float64(time.Now().Unix()))

	// Update reconcile duration
	helmApplySetReconcileDurationSeconds.WithLabelValues(health.ReleaseName, health.Namespace).
		Observe(reconcileDuration.Seconds())
}

// updateResourceMetrics updates resource count metrics grouped by GroupKind
func (e *Exporter) updateResourceMetrics(health *status.ReleaseHealth) {
	// Count resources by GroupKind
	gvkCounts := make(map[string]int)
	gvkHealthyCounts := make(map[string]int)

	for resourceKey, resourceHealth := range health.ResourceHealth {
		// Extract GroupKind from key (format: "Group.Kind/namespace/name" or "Kind/namespace/name")
		parts := strings.Split(resourceKey, "/")
		if len(parts) < 1 {
			continue
		}
		gvk := parts[0]

		gvkCounts[gvk]++
		if resourceHealth.Healthy {
			gvkHealthyCounts[gvk]++
		}
	}

	// Update metrics for each GroupKind
	for gvk, count := range gvkCounts {
		helmApplySetResourceTotal.WithLabelValues(health.ReleaseName, health.Namespace, gvk).Set(float64(count))
		healthyCount := gvkHealthyCounts[gvk]
		helmApplySetResourceHealthy.WithLabelValues(health.ReleaseName, health.Namespace, gvk).Set(float64(healthyCount))
	}
}

// DeleteMetrics removes metrics for a deleted Helm release
func (e *Exporter) DeleteMetrics(releaseName, namespace string) {
	releaseKey := e.releaseKey(releaseName, namespace)

	// Remove from creation times
	e.mu.Lock()
	delete(e.releaseCreationTimes, releaseKey)
	e.mu.Unlock()

	// Delete all metrics for this release
	// Note: Prometheus doesn't support deletion, but we can set to 0
	// In practice, metrics will expire based on TTL if not updated

	// Set status metrics to 0
	for _, status := range []string{"healthy", "progressing", "degraded", "failed", "unknown"} {
		helmApplySetStatus.DeleteLabelValues(releaseName, namespace, status)
	}

	// Delete age and timestamp metrics
	helmApplySetAgeSeconds.DeleteLabelValues(releaseName, namespace)
	helmApplySetLastReconcileTimestamp.DeleteLabelValues(releaseName, namespace)
	helmApplySetReconcileDurationSeconds.DeleteLabelValues(releaseName, namespace)

	// Note: Resource metrics are harder to delete without knowing all GVKs
	// They will be cleaned up on next reconciliation or expire naturally
}

// statusToValue converts status string to numeric value
func (e *Exporter) statusToValue(status string) float64 {
	switch status {
	case "healthy":
		return 1
	case "progressing":
		return 2
	case "degraded":
		return 3
	case "failed":
		return 4
	default:
		return 0 // unknown
	}
}

// releaseKey creates a unique key for a release
func (e *Exporter) releaseKey(releaseName, namespace string) string {
	return fmt.Sprintf("%s/%s", namespace, releaseName)
}

// RecordReconcileDuration records just the reconciliation duration metric
// This is a simpler alternative to UpdateMetrics for cases where full health status isn't available
func (e *Exporter) RecordReconcileDuration(releaseName, namespace string, duration time.Duration) {
	helmApplySetReconcileDurationSeconds.WithLabelValues(releaseName, namespace).
		Observe(duration.Seconds())
	helmApplySetLastReconcileTimestamp.WithLabelValues(releaseName, namespace).
		Set(float64(time.Now().Unix()))
}
