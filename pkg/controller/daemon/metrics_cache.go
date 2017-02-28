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
	"hash/fnv"
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

// metricsCache save metrics about pod per node of daemon set
type metricsCache struct {
	mutex sync.RWMutex
	cache *lru.Cache
}

// newMetricsCache return a new metricsCache
func newMetricsCache(maxCacheEntries int) *metricsCache {
	return &metricsCache{
		cache: lru.New(maxCacheEntries),
	}
}

func (c *metricsCache) recordDeletePod(ds *v1beta1.DaemonSet, pod *v1.Pod, deleteTime time.Time) {
	if ds.Namespace != pod.Namespace || len(pod.Spec.NodeName) == 0 ||
		ds.Spec.UpdateStrategy.Type != v1beta1.RollingUpdateDaemonSetStrategyType {
		return
	}
	c.Add(daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: pod.Spec.NodeName},
		podMetrics{deletedPodName: pod.Name, deleteTime: deleteTime})
}

func (c *metricsCache) recordRecreatePod(ds *v1beta1.DaemonSet, pod *v1.Pod, recreateTime time.Time) {
	if ds.Namespace != pod.Namespace || len(pod.Spec.NodeName) == 0 ||
		ds.Spec.UpdateStrategy.Type != v1beta1.RollingUpdateDaemonSetStrategyType {
		return
	}
	dsNode := daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: pod.Spec.NodeName}
	if obj, cached := c.Get(dsNode); cached {
		m, ok := obj.(podMetrics)
		if !ok || !validForRecreate(m) {
			return
		}
		podKillCreateLatency.Observe(subInMilliseconds(m.deleteTime, recreateTime))
		// Update pod metrics
		m.recreatePodName = pod.Name
		m.recreateTime = recreateTime
		c.Update(dsNode, m)
	}
}

func validForRecreate(metrics podMetrics) bool {
	return !metrics.deleteTime.IsZero() && len(metrics.deletedPodName) > 0
}

func (c *metricsCache) recordReadyPod(ds *v1beta1.DaemonSet, pod *v1.Pod, readyTime time.Time) {
	if ds.Namespace != pod.Namespace || len(pod.Spec.NodeName) == 0 ||
		ds.Spec.UpdateStrategy.Type != v1beta1.RollingUpdateDaemonSetStrategyType {
		return
	}
	dsNode := daemonSetNode{namespace: ds.Namespace, daemonSetName: ds.Name, nodeName: pod.Spec.NodeName}
	if obj, cached := c.Get(dsNode); cached {
		m, ok := obj.(podMetrics)
		if !ok || !validForReady(m) {
			return
		}
		podKillReadyLatency.Observe(subInMilliseconds(m.deleteTime, readyTime))
		// Clean up this pod metrics
		m = podMetrics{}
		c.Update(dsNode, m)
	}
}

func validForReady(metrics podMetrics) bool {
	return !metrics.recreateTime.IsZero() && len(metrics.recreatePodName) > 0 && validForRecreate(metrics)
}

// Add will add matching information to the cache.
func (c *metricsCache) Add(dsNode daemonSetNode, metrics podMetrics) {
	key := keyFunc(dsNode)
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.cache.Add(key, metrics)
}

// keyFunc returns the key of an object, which is used to look up in the cache for it's matching object.
func keyFunc(dsNode daemonSetNode) uint64 {
	hash := fnv.New32a()
	hashutil.DeepHashObject(hash, &dsNode)
	return uint64(hash.Sum32())
}

// Get lookup the matching metrics for a given daemonSetNode.
// Note: the cache information may be invalid since the controller may be deleted or updated,
// we need check in the external request to ensure the cache data is not dirty.
func (c *metricsCache) Get(dsNode daemonSetNode) (obj interface{}, exists bool) {
	key := keyFunc(dsNode)
	// NOTE: we use Lock() instead of RLock() here because lru's Get() method also modifies state(
	// it need update the least recently usage information). So we can not call it concurrently.
	c.mutex.Lock()
	defer c.mutex.Unlock()
	return c.cache.Get(key)
}

// Update update the cached matching metrics.
func (c *metricsCache) Update(dsNode daemonSetNode, metrics podMetrics) {
	c.Add(dsNode, metrics)
}
