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
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	metricutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// Subsystem names.
	pvControllerSubsystem = "pv_collector"

	// Metric names.
	boundPVKey    = "bound_pv_count"
	unboundPVKey  = "unbound_pv_count"
	boundPVCKey   = "bound_pvc_count"
	unboundPVCKey = "unbound_pvc_count"

	// Label names.
	namespaceLabel    = "namespace"
	storageClassLabel = "storage_class"
)

var registerMetrics sync.Once

// PVLister used to list persistent volumes.
type PVLister interface {
	List() []interface{}
}

// PVCLister used to list persistent volume claims.
type PVCLister interface {
	List() []interface{}
}

// Register all metrics for pv controller.
func Register(pvLister PVLister, pvcLister PVCLister) {
	registerMetrics.Do(func() {
		legacyregistry.CustomMustRegister(newPVAndPVCCountCollector(pvLister, pvcLister))
		legacyregistry.MustRegister(volumeOperationErrorsMetric)
	})
}

func newPVAndPVCCountCollector(pvLister PVLister, pvcLister PVCLister) *pvAndPVCCountCollector {
	return &pvAndPVCCountCollector{pvLister: pvLister, pvcLister: pvcLister}
}

// Custom collector for current pod and container counts.
type pvAndPVCCountCollector struct {
	metrics.BaseStableCollector

	// Cache for accessing information about PersistentVolumes.
	pvLister PVLister
	// Cache for accessing information about PersistentVolumeClaims.
	pvcLister PVCLister
}

// Check if our collector implements necessary collector interface
var _ metrics.StableCollector = &pvAndPVCCountCollector{}

var (
	boundPVCountDesc = metrics.NewDesc(
		metrics.BuildFQName("", pvControllerSubsystem, boundPVKey),
		"Gauge measuring number of persistent volume currently bound",
		[]string{storageClassLabel}, nil,
		metrics.ALPHA, "")
	unboundPVCountDesc = metrics.NewDesc(
		metrics.BuildFQName("", pvControllerSubsystem, unboundPVKey),
		"Gauge measuring number of persistent volume currently unbound",
		[]string{storageClassLabel}, nil,
		metrics.ALPHA, "")

	boundPVCCountDesc = metrics.NewDesc(
		metrics.BuildFQName("", pvControllerSubsystem, boundPVCKey),
		"Gauge measuring number of persistent volume claim currently bound",
		[]string{namespaceLabel}, nil,
		metrics.ALPHA, "")
	unboundPVCCountDesc = metrics.NewDesc(
		metrics.BuildFQName("", pvControllerSubsystem, unboundPVCKey),
		"Gauge measuring number of persistent volume claim currently unbound",
		[]string{namespaceLabel}, nil,
		metrics.ALPHA, "")

	volumeOperationErrorsMetric = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Name:           "volume_operation_total_errors",
			Help:           "Total volume operation errors",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"plugin_name", "operation_name"})
)

func (collector *pvAndPVCCountCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- boundPVCountDesc
	ch <- unboundPVCountDesc
	ch <- boundPVCCountDesc
	ch <- unboundPVCCountDesc
}

func (collector *pvAndPVCCountCollector) CollectWithStability(ch chan<- metrics.Metric) {
	collector.pvCollect(ch)
	collector.pvcCollect(ch)
}

func (collector *pvAndPVCCountCollector) pvCollect(ch chan<- metrics.Metric) {
	boundNumberByStorageClass := make(map[string]int)
	unboundNumberByStorageClass := make(map[string]int)
	for _, pvObj := range collector.pvLister.List() {
		pv, ok := pvObj.(*v1.PersistentVolume)
		if !ok {
			continue
		}
		if pv.Status.Phase == v1.VolumeBound {
			boundNumberByStorageClass[pv.Spec.StorageClassName]++
		} else {
			unboundNumberByStorageClass[pv.Spec.StorageClassName]++
		}
	}
	for storageClassName, number := range boundNumberByStorageClass {
		ch <- metrics.NewLazyConstMetric(
			boundPVCountDesc,
			metrics.GaugeValue,
			float64(number),
			storageClassName)
	}
	for storageClassName, number := range unboundNumberByStorageClass {
		ch <- metrics.NewLazyConstMetric(
			unboundPVCountDesc,
			metrics.GaugeValue,
			float64(number),
			storageClassName)
	}
}

func (collector *pvAndPVCCountCollector) pvcCollect(ch chan<- metrics.Metric) {
	boundNumberByNamespace := make(map[string]int)
	unboundNumberByNamespace := make(map[string]int)
	for _, pvcObj := range collector.pvcLister.List() {
		pvc, ok := pvcObj.(*v1.PersistentVolumeClaim)
		if !ok {
			continue
		}
		if pvc.Status.Phase == v1.ClaimBound {
			boundNumberByNamespace[pvc.Namespace]++
		} else {
			unboundNumberByNamespace[pvc.Namespace]++
		}
	}
	for namespace, number := range boundNumberByNamespace {
		ch <- metrics.NewLazyConstMetric(
			boundPVCCountDesc,
			metrics.GaugeValue,
			float64(number),
			namespace)
	}
	for namespace, number := range unboundNumberByNamespace {
		ch <- metrics.NewLazyConstMetric(
			unboundPVCCountDesc,
			metrics.GaugeValue,
			float64(number),
			namespace)
	}
}

// RecordVolumeOperationErrorMetric records error count into metric
// volume_operation_total_errors for provisioning/deletion operations
func RecordVolumeOperationErrorMetric(pluginName, opName string) {
	if pluginName == "" {
		pluginName = "N/A"
	}
	volumeOperationErrorsMetric.WithLabelValues(pluginName, opName).Inc()
}

// operationTimestamp stores the start time of an operation by a plugin
type operationTimestamp struct {
	pluginName string
	operation  string
	startTs    time.Time
}

func newOperationTimestamp(pluginName, operationName string) *operationTimestamp {
	return &operationTimestamp{
		pluginName: pluginName,
		operation:  operationName,
		startTs:    time.Now(),
	}
}

// OperationStartTimeCache concurrent safe cache for operation start timestamps
type OperationStartTimeCache struct {
	cache sync.Map // [string]operationTimestamp
}

// NewOperationStartTimeCache creates a operation timestamp cache
func NewOperationStartTimeCache() OperationStartTimeCache {
	return OperationStartTimeCache{
		cache: sync.Map{}, // [string]operationTimestamp {}
	}
}

// AddIfNotExist returns directly if there exists an entry with the key. Otherwise, it
// creates a new operation timestamp using operationName, pluginName, and current timestamp
// and stores the operation timestamp with the key
func (c *OperationStartTimeCache) AddIfNotExist(key, pluginName, operationName string) {
	ts := newOperationTimestamp(pluginName, operationName)
	c.cache.LoadOrStore(key, ts)
}

// Delete deletes a value for a key.
func (c *OperationStartTimeCache) Delete(key string) {
	c.cache.Delete(key)
}

// Has returns a bool value indicates the existence of a key in the cache
func (c *OperationStartTimeCache) Has(key string) bool {
	_, exists := c.cache.Load(key)
	return exists
}

// RecordMetric records either an error count metric or a latency metric if there
// exists a start timestamp entry in the cache. For a successful operation, i.e.,
// err == nil, the corresponding timestamp entry will be removed from cache
func RecordMetric(key string, c *OperationStartTimeCache, err error) {
	obj, exists := c.cache.Load(key)
	if !exists {
		return
	}
	ts, ok := obj.(*operationTimestamp)
	if !ok {
		return
	}
	if err != nil {
		RecordVolumeOperationErrorMetric(ts.pluginName, ts.operation)
	} else {
		timeTaken := time.Since(ts.startTs).Seconds()
		metricutil.RecordOperationLatencyMetric(ts.pluginName, ts.operation, timeTaken)
		// end of this operation, remove the timestamp entry from cache
		c.Delete(key)
	}
}
