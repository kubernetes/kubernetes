/*
Copyright 2022 The Kubernetes Authors.

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
	"context"
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

type sinceFunc func(time.Time) time.Duration

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

	since sinceFunc = time.Since
)

var registerMetrics sync.Once

// RegisterMetrics registers the controller metrics
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(ReconciliationDurations)
	})
}

// SyncAndRecordWithCtx wraps syncFn with a function, recording its result and duration as a metric.
func SyncAndRecordWithCtx(
	controllerName string, syncFn func(context.Context, string) error) func(context.Context, string) error {
	return func(ctx context.Context, key string) error {
		return RunSyncAndRecordWithCtx(ctx, controllerName, key, syncFn)
	}
}

// RunSyncAndRecordWithCtx runs syncFn with the given context and key, recording the reconcile time and results
func RunSyncAndRecordWithCtx(
	ctx context.Context, controllerName string, key string, syncFn func(context.Context, string) error) error {
	startTime := time.Now()
	err := syncFn(ctx, key)
	recordSyncResult(controllerName, startTime, err)
	return err
}

// RunSyncAndRecord runs syncFn with key, recording the reconcile time and results
func RunSyncAndRecord(
	controllerName string, key string, syncFn func(string) error) error {
	startTime := time.Now()
	err := syncFn(key)
	recordSyncResult(controllerName, startTime, err)
	return err
}

// HasSyncedAndRecordedWithCtx wraps syncFn with a function, recording its result and duration as a metric.
func HasSyncedAndRecordedWithCtx(
	controllerName string, syncFn func(context.Context, string) (bool, error)) func(context.Context, string) (bool, error) {
	return func(ctx context.Context, key string) (bool, error) {
		return RunHasSyncedAndRecordedWithCtx(ctx, controllerName, key, syncFn)
	}
}

// HasSyncedAndRecorded wraps syncFn with a function, recording its result and duration as a metric.
func HasSyncedAndRecorded(
	controllerName string, syncFn func(string) (bool, error)) func(string) (bool, error) {
	return func(key string) (bool, error) {
		return RunHasSyncedAndRecorded(controllerName, key, syncFn)
	}
}

// RunHasSyncedAndRecordedWithCtx runs syncFn with key, recording the reconcile time and results
func RunHasSyncedAndRecordedWithCtx(
	ctx context.Context, controllerName string, key string, syncFn func(context.Context, string) (bool, error)) (bool, error) {
	startTime := time.Now()
	bool, err := syncFn(ctx, key)
	recordSyncResult(controllerName, startTime, err)
	return bool, err
}

// RunHasSyncedAndRecorded runs syncFn with key, recording the reconcile time and results
func RunHasSyncedAndRecorded(
	controllerName string, key string, syncFn func(string) (bool, error)) (bool, error) {
	startTime := time.Now()
	bool, err := syncFn(key)
	recordSyncResult(controllerName, startTime, err)
	return bool, err
}

// SyncAndRecordAll wraps syncFn with a function, recording its result and duration as a metric.
func SyncAndRecordAll(controllerName string, syncFn func(context.Context)) func(context.Context) {
	return func(ctx context.Context) {
		RunSyncAndRecordAll(ctx, controllerName, syncFn)
	}
}

// RunSyncAndRecordAll runs syncFn, recording the reconcile time and results
func RunSyncAndRecordAll(ctx context.Context, controllerName string, syncFn func(context.Context)) {
	startTime := time.Now()
	syncFn(ctx)
	recordSyncResult(controllerName, startTime, nil)
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
	syncDuration := since(startTime).Seconds()

	ReconciliationDurations.WithLabelValues(controllerName, status).Observe(syncDuration)
	klog.V(4).InfoS("Finished syncing",
		"controller", controllerName,
		"duration", syncDuration,
		"status", status)
}
