/*
Copyright 2021 The Kubernetes Authors.

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

package controller

import (
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

const controllerNamespace = "controller"

var (
	ReconciliationDurations = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      controllerNamespace,
			Name:           "reconcile_time_seconds",
			Help:           "Duration of a sync of a controller's resource",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
		},
		[]string{"controller", "status"},
	)
)

var registerMetrics sync.Once

// RegisterMetrics registers the controller metrics
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ReconciliationDurations)
	})
}

// SyncAndRecord wraps syncFn with a function, recording its result and duration as a metric.
func SyncAndRecord(
	controllerName string, syncFn func(string) error) func(string) error {
	return func(key string) error {
		return RunSyncAndRecord(controllerName, key, syncFn)
	}
}

// RunSyncAndRecord runs syncFn with key, recording the reconcile time and results
func RunSyncAndRecord(
	controllerName string, key string, syncFn func(string) error) error {
	startTime := time.Now()
	err := syncFn(key)
	recordSyncResult(controllerName, startTime, err)
	return err
}

// SyncAndRecordWithBool wraps syncFn with a function, recording its result and duration as a metric.
func SyncAndRecordWithBool(
	controllerName string, syncFn func(string) (bool, error)) func(string) (bool, error) {
	return func(key string) (bool, error) {
		return RunSyncAndRecordWithBool(controllerName, key, syncFn)
	}
}

// RunSyncAndRecordWithBool runs syncFn with key, recording the reconcile time and results
func RunSyncAndRecordWithBool(
	controllerName string, key string, syncFn func(string) (bool, error)) (bool, error) {
	startTime := time.Now()
	bool, err := syncFn(key)
	recordSyncResult(controllerName, startTime, err)
	return bool, err
}

// SyncAndRecordAll wraps syncFn with a function, recording its result and duration as a metric.
func SyncAndRecordAll(controllerName string, syncFn func() error) func() {
	return func() {
		RunSyncAndRecordAll(controllerName, syncFn)
	}
}

// RunSyncAndRecordAll runs syncFn, recording the reconcile time and results
func RunSyncAndRecordAll(controllerName string, syncFn func() error) {
	startTime := time.Now()
	err := syncFn()
	recordSyncResult(controllerName, startTime, err)
}

const (
	// success indicates a successful sync operation
	success = "success"
	// failed indicates a failed sync operation
	failed = "failed"
)

// recordSyncResult records the results of a controller sync operation
func recordSyncResult(controllerName string, startTime time.Time, err error) {
	status := success
	if err != nil {
		status = failed
	}
	syncDuration := time.Since(startTime).Seconds()

	ReconciliationDurations.WithLabelValues(controllerName, status).Observe(syncDuration)
	klog.V(4).Infof("Controller %s finished syncing in %.3f seconds with status %s",
		controllerName, syncDuration, status)
}
