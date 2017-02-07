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

package daemon

import (
	"hash/adler32"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	hashutil "k8s.io/kubernetes/pkg/util/hash"

	"github.com/golang/groupcache/lru"
)

type daemonSetNode struct {
	namespace, daemonSetName, nodeName string
}

type podMetrics struct {
	deletedPodName  string
	deleteTime      time.Time
	recreatePodName string
	recreateTime    time.Time
}

// MetricsCache save metrics about pod per node of daemon set
type MetricsCache struct {
	mutex sync.RWMutex
	cache *lru.Cache
}

// NewMetricsCache return a new MetricsCache
func NewMetricsCache(maxCacheEntries int) *MetricsCache {
	return &MetricsCache{
		cache: lru.New(maxCacheEntries),
	}
}

func (c *MetricsCache) RecordDeletePod(ds *v1beta1.DaemonSet, pod *v1.Pod, deleteTime time.Time) {
	if ds.Namespace != pod.Namespace || len(pod.Spec.NodeName) == 0 {
		return
	}
	c.Add(daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: pod.Spec.NodeName},
		podMetrics{deletedPodName: pod.Name, deleteTime: deleteTime})
}

func (c *MetricsCache) RecordRecreatePod(ds *v1beta1.DaemonSet, pod *v1.Pod, recreateTime time.Time) {
	if ds.Namespace != pod.Namespace || len(pod.Spec.NodeName) == 0 {
		return
	}
	dsNode := daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: pod.Spec.NodeName}
	if obj, cached := c.Get(dsNode); cached {
		m, ok := obj.(podMetrics)
		if !ok || !validForRecreate(m, recreateTime) {
			return
		}
		m.recreatePodName = pod.Name
		m.recreateTime = recreateTime
		c.Update(dsNode, m)
		PodKillCreateLatency.Observe(subInMilliseconds(m.deleteTime, recreateTime))
	}
}

func validForRecreate(metrics podMetrics, recreateTime time.Time) bool {
	return !metrics.deleteTime.IsZero() && len(metrics.deletedPodName) > 0
}

func (c *MetricsCache) RecordRunningPod(ds *v1beta1.DaemonSet, pod *v1.Pod, runningTime time.Time) {
	if ds.Namespace != pod.Namespace || len(pod.Spec.NodeName) == 0 {
		return
	}
	dsNode := daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: pod.Spec.NodeName}
	if obj, cached := c.Get(dsNode); cached {
		m, ok := obj.(podMetrics)
		if !ok || !validForRunning(m, runningTime) {
			return
		}
		PodKillRunningLatency.Observe(subInMilliseconds(m.deleteTime, runningTime))
	}
}

func validForRunning(metrics podMetrics, runningTime time.Time) bool {
	return !metrics.recreateTime.IsZero() && len(metrics.recreatePodName) > 0 && validForRecreate(metrics, metrics.recreateTime)
}

// Add will add matching information to the cache.
func (c *MetricsCache) Add(dsNode daemonSetNode, metrics podMetrics) {
	key := keyFunc(dsNode)
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache.Add(key, metrics)
}

// keyFunc returns the key of an object, which is used to look up in the cache for it's matching object.
func keyFunc(dsNode daemonSetNode) uint64 {
	hash := adler32.New()
	hashutil.DeepHashObject(hash, &dsNode)
	return uint64(hash.Sum32())
}

// Get lookup the matching metrics for a given daemonSetNode.
// Note: the cache information may be invalid since the controller may be deleted or updated,
// we need check in the external request to ensure the cache data is not dirty.
func (c *MetricsCache) Get(dsNode daemonSetNode) (obj interface{}, exists bool) {
	key := keyFunc(dsNode)
	// NOTE: we use Lock() instead of RLock() here because lru's Get() method also modifies state(
	// it need update the least recently usage information). So we can not call it concurrently.
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.cache.Get(key)
}

// Update update the cached matching metrics.
func (c *MetricsCache) Update(dsNode daemonSetNode, metrics podMetrics) {
	c.Add(dsNode, metrics)
}
