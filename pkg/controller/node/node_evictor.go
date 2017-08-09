/*
Copyright 2014 The Kubernetes Authors.

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

package node

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/controller/node/scheduler"
)

type nodeEvictor interface {
	// startEvictionLoop manages eviction of nodes.
	startEvictionLoop()

	// addZoneWorker adds zone worker to map of pod evictors/tainters.
	addZoneWorker(zone string)

	cancelPodEvictionOrMarkNodeAsHealthy(node *v1.Node)
}

// taintBasedNodeEvictor satisfies the Evictor interface
type taintBasedNodeEvictor struct {
	nc *Controller
}

// defaultNodeEvictor satisfies the Evictor interface
type defaultNodeEvictor struct {
	nc *Controller
}

func newTaintBasedNodeEvictor(nc *Controller) *taintBasedNodeEvictor {
	return &taintBasedNodeEvictor{nc}
}

func newDefaultNodeEvictor(nc *Controller) *defaultNodeEvictor {
	return &defaultNodeEvictor{nc}
}

func (ev *taintBasedNodeEvictor) startEvictionLoop() {
	// Because we don't want a dedicated logic in TaintManager for NC-originated taints and we
	// normally don't rate limit evictions caused by taints, we need to rate limit adding taints.
	go wait.Until(ev.nc.doNoExecuteTaintingPass, scheduler.NodeEvictionPeriod, wait.NeverStop)
}

func (ev *defaultNodeEvictor) startEvictionLoop() {
	// When we delete pods off a node, if the node was not empty at the time we then queue an
	// eviction watcher. If we hit an error, retry deletion.
	go wait.Until(ev.nc.doEvictionPass, scheduler.NodeEvictionPeriod, wait.NeverStop)

}

func (ev *taintBasedNodeEvictor) addZoneWorker(zone string) {
	ev.nc.zoneNoExecuteTainter[zone] = scheduler.NewRateLimitedTimedQueue(
		flowcontrol.NewTokenBucketRateLimiter(
			ev.nc.evictionLimiterQPS, scheduler.EvictionRateLimiterBurst))
}

func (ev *defaultNodeEvictor) addZoneWorker(zone string) {
	ev.nc.zonePodEvictor[zone] = scheduler.NewRateLimitedTimedQueue(
		flowcontrol.NewTokenBucketRateLimiter(
			ev.nc.evictionLimiterQPS, scheduler.EvictionRateLimiterBurst))
}

func (ev *taintBasedNodeEvictor) cancelPodEvictionOrMarkNodeAsHealthy(node *v1.Node) {
	ev.nc.markNodeAsHealthy(node)
}

func (ev *defaultNodeEvictor) cancelPodEvictionOrMarkNodeAsHealthy(node *v1.Node) {
	ev.nc.cancelPodEviction(node)
}
