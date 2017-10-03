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
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/controller/node/scheduler"
	"k8s.io/kubernetes/pkg/controller/node/util"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
)

type nodeEvictor interface {
	// startEvictionLoop manages eviction of nodes.
	startEvictionLoop()

	// addZoneWorker adds zone worker to map of pod evictors/tainters.
	addZoneWorker(zone string)

	cancelPodEvictionOrMarkNodeAsHealthy(node *v1.Node)

	// setLimiterForZone sets queue limiter for zone within map of pod evictors/tainters.
	setLimiterForZone(newQPS float32, zone string)

	checkEvictionTimeoutAgainstDecisionTimestamp(
		node *v1.Node,
		observedReadyCondition v1.NodeCondition,
		decisionTimestamp metav1.Time,
		gracePeriod time.Duration)
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

func (ev *taintBasedNodeEvictor) setLimiterForZone(newQPS float32, zone string) {
	ev.nc.zoneNoExecuteTainter[zone].SwapLimiter(newQPS)
}

func (ev *defaultNodeEvictor) setLimiterForZone(newQPS float32, zone string) {
	ev.nc.zonePodEvictor[zone].SwapLimiter(newQPS)
}

func (ev *taintBasedNodeEvictor) checkEvictionTimeoutAgainstDecisionTimestamp(
	node *v1.Node,
	observedReadyCondition v1.NodeCondition,
	decisionTimestamp metav1.Time,
	gracePeriod time.Duration) {

	switch observedReadyCondition.Status {
	case v1.ConditionFalse:
		// We want to update the taint straight away if Node is already tainted with the UnreachableTaint.
		if taintutils.TaintExists(node.Spec.Taints, UnreachableTaintTemplate) {
			taintToAdd := *NotReadyTaintTemplate
			if !util.SwapNodeControllerTaint(ev.nc.kubeClient, &taintToAdd, UnreachableTaintTemplate, node) {
				glog.Errorf("Failed to instantly swap UnreachableTaint to NotReadyTaint. Will try again in the next cycle.")
			}
		} else if ev.nc.markNodeForTainting(node) {
			glog.V(2).Infof("Node %v is NotReady as of %v. Adding it to the Taint queue.",
				node.Name, decisionTimestamp,
			)
		}

	case v1.ConditionUnknown:
		// We want to update the taint straight away if Node is already tainted with the UnreachableTaint
		if taintutils.TaintExists(node.Spec.Taints, NotReadyTaintTemplate) {
			taintToAdd := *UnreachableTaintTemplate
			if !util.SwapNodeControllerTaint(ev.nc.kubeClient, &taintToAdd, NotReadyTaintTemplate, node) {
				glog.Errorf("Failed to instantly swap UnreachableTaint to NotReadyTaint. Will try again in the next cycle.")
			}
		} else if ev.nc.markNodeForTainting(node) {
			glog.V(2).Infof("Node %v is unresponsive as of %v. Adding it to the Taint queue.",
				node.Name, decisionTimestamp,
			)
		}
	case v1.ConditionTrue:
		removed, _ := ev.nc.markNodeAsHealthy(node) // Error path already logged.
		if removed {
			glog.V(2).Infof("Node %s is healthy again, removing all taints", node.Name)
		}
	}
}

func (ev *defaultNodeEvictor) checkEvictionTimeoutAgainstDecisionTimestamp(
	node *v1.Node,
	observedReadyCondition v1.NodeCondition,
	decisionTimestamp metav1.Time,
	gracePeriod time.Duration) {

	switch observedReadyCondition.Status {
	case v1.ConditionFalse:
		if decisionTimestamp.After(ev.nc.nodeStatusMap[node.Name].readyTransitionTimestamp.Add(ev.nc.podEvictionTimeout)) {
			if ev.nc.evictPods(node) {
				glog.V(2).Infof("Node is NotReady. Adding Pods on Node %s to eviction queue: %v is later than %v + %v",
					node.Name,
					decisionTimestamp,
					ev.nc.nodeStatusMap[node.Name].readyTransitionTimestamp,
					ev.nc.podEvictionTimeout,
				)
			}
		}

	case v1.ConditionUnknown:
		if decisionTimestamp.After(ev.nc.nodeStatusMap[node.Name].probeTimestamp.Add(ev.nc.podEvictionTimeout)) {
			if ev.nc.evictPods(node) {
				glog.V(2).Infof("Node is unresponsive. Adding Pods on Node %s to eviction queues: %v is later than %v + %v",
					node.Name,
					decisionTimestamp,
					ev.nc.nodeStatusMap[node.Name].readyTransitionTimestamp,
					ev.nc.podEvictionTimeout-gracePeriod,
				)
			}
		}

	case v1.ConditionTrue:
		ev.nc.cancelPodEviction(node)
	}
}
