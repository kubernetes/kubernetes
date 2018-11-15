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

package debugger

import (
	corelisters "k8s.io/client-go/listers/core/v1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
)

// CacheDebugger provides ways to check and write cache information for debugging.
type CacheDebugger struct {
	Comparer CacheComparer
	Dumper   CacheDumper
}

// New creates a CacheDebugger.
func New(
	nodeLister corelisters.NodeLister,
	podLister corelisters.PodLister,
	cache internalcache.Cache,
	podQueue internalqueue.SchedulingQueue,
) *CacheDebugger {
	return &CacheDebugger{
		Comparer: CacheComparer{
			NodeLister: nodeLister,
			PodLister:  podLister,
			Cache:      cache,
			PodQueue:   podQueue,
		},
		Dumper: CacheDumper{
			cache:    cache,
			podQueue: podQueue,
		},
	}
}
