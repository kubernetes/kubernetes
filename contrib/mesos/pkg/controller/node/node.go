/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"time"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/contrib/mesos/pkg/node"
	"k8s.io/kubernetes/contrib/mesos/pkg/podutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/mesos"
	"k8s.io/kubernetes/pkg/controller/framework"
	kuberuntime "k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	nodeStatusUpdateRetry = 5
)

type (
	monitor struct {
		heartbeatPeriod time.Duration
		worker          func()
	}

	podIndex struct {
		nodePods map[string]sets.String // map node to pod
	}

	Controller struct {
		client    *clientset.Clientset
		clockFunc func() time.Time
		monitors  []*monitor

		// Node framework and store
		nodeController *framework.Controller
		nodeStore      cache.StoreToNodeLister

		// Pod framework and store
		podController *framework.Controller
		pods          cache.Indexer
	}

	// Option is a functional option for Controller
	Option func(*Controller)
)

// PodSpecNodeNameIndexFunc is a default index function that indexes based on an pod's node name
func PodSpecNodeNameIndexFunc(obj interface{}) ([]string, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return []string{""}, fmt.Errorf("object was not a pod")
	}
	return []string{pod.Spec.NodeName}, nil
}

func ClockFunc(f func() time.Time) Option {
	return func(c *Controller) {
		c.clockFunc = f
	}
}

func SlaveStatusController() Option {
	return func(c *Controller) {
		controller := c.newSlaveStatusController()
		c.monitors = append(c.monitors, controller.monitor)
	}
}

func NewController(client *clientset.Clientset, relistPeriod time.Duration, options ...Option) *Controller {
	c := &Controller{
		client:    client,
		clockFunc: time.Now,
	}
	// client is only optional for unit tests
	if client != nil {
		c.nodeStore.Store, c.nodeController = framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (kuberuntime.Object, error) {
					return c.client.Core().Nodes().List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return c.client.Core().Nodes().Watch(options)
				},
			},
			&api.Node{},
			relistPeriod,
			framework.ResourceEventHandlerFuncs{},
		)
		c.pods, c.podController = framework.NewIndexerInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (kuberuntime.Object, error) {
					return c.client.Core().Pods(api.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return c.client.Core().Pods(api.NamespaceAll).Watch(options)
				},
			},
			&api.Pod{},
			relistPeriod,
			framework.ResourceEventHandlerFuncs{},
			cache.Indexers{api.PodHostField: PodSpecNodeNameIndexFunc},
		)
	}
	// apply functional options
	for _, opt := range options {
		opt(c)
	}
	return c
}

func (c *Controller) Run(terminate <-chan struct{}) {
	go c.nodeController.Run(terminate)
	for _, m := range c.monitors {
		if m.worker != nil {
			go runtime.Until(func() { m.worker() }, m.heartbeatPeriod, terminate)
		}
	}
	<-terminate
}

type slaveStatusController struct {
	*monitor
	gracePeriod time.Duration
	nodeCache   cache.Store
	clockFunc   func() time.Time
	nodes       unversionedcore.NodesGetter
	podsForNode func(string) ([]interface{}, error) // field for easier unit testing
}

func (c *Controller) newSlaveStatusController() *slaveStatusController {
	kubecfg := options.NewKubeletServer() // only create to get the config, this is without side-effects
	controller := &slaveStatusController{
		monitor: &monitor{
			heartbeatPeriod: kubecfg.NodeStatusUpdateFrequency.Duration,
		},
		clockFunc: c.clockFunc,
		nodeCache: c.nodeStore.Store,
	}

	// optional for easier unit testing
	if c.client != nil {
		controller.nodes = c.client.Core()
	}
	controller.podsForNode = func(nodeName string) ([]interface{}, error) {
		key := &api.Pod{Spec: api.PodSpec{NodeName: nodeName}}
		return c.pods.Index(api.PodHostField, key)
	}

	// avoid flapping by waiting at least twice the kubetlet update frequency, i.e.
	// give the kubelet the chance twice to update the heartbeat. This is necessary
	// because we only poll the Mesos master state.json once in a while and we
	// know that that the information from there can easily be outdated.
	controller.gracePeriod = controller.heartbeatPeriod * 2
	controller.worker = controller.checkStatus
	return controller
}

func (ssc *slaveStatusController) checkStatus() {
	// build up a set of listed slave nodes without a kubelet
	slaves, err := mesos.CloudProvider.ListWithoutKubelet()
	if err != nil {
		log.Errorf("Error listing slaves without kubelet: %v", err)
		return
	}
	slavesWithoutKubelet := sets.NewString(slaves...)

	// update status for nodes which do not have a kubelet running and
	// which are still existing as slave. This status update must be done
	// before the node controller counts down the NodeMonitorGracePeriod
	nodes := ssc.nodeCache.List()

	for _, n := range nodes {
		node := n.(*api.Node)

		updated, err := ssc.checkWithoutKubelet(node, slavesWithoutKubelet)
		if err != nil {
			log.Errorf("Error updating node status: %v", err)
		}
		if updated {
			continue
		}

		_, err = ssc.checkStalePods(node)
		if err != nil {
			log.Errorf("failed stale pod check: %v", err)
		}
	}
}

// it's possible that some number of nodes are no longer part of the cluster. in this
// case k8s should take care of pod state cleanup in apiserver, so we don't worry about
// that here.
// we're primarily interested in cleaning up stale pod records from apiserver:
// - after a pod is bound, kubelet becomes the source of truth re: pod state
// - kubelet isn't guanteed to shut down cleanly
//   - may leave lingering docker containers (solved by mesos orphan CT GC?)
//   - may leave lingering pod records in apiserver (want to solve this here!)
// - our slaveStatusController simulates NodeReady condition for **all** known slaves
//   - this prevents k8s pod GC/cleanup from properly identifying stale pod state for
//     nodes that exist but aren't running kubelet
func (ssc *slaveStatusController) checkStalePods(n *api.Node) (bool, error) {
	cond := getCondition(&n.Status, node.RunningExecutorCondition)
	if cond != nil && cond.Status == api.ConditionTrue {
		// 1) heartbeat is current --> uuid is possibly stale
		//   ?? kubelet cycled but we haven't observed a status update yet
		//   a) delete all pods on the node, except non-matching latest newer UUID;timestamp within grace period or else matching UUID's
		parsed, ok := node.ParseConditionReason(*cond)
		if !ok {
			log.Warningf("unable to parse RunningExecutorCondition reason %q", cond.Reason)
			return false, nil
		}

		pods, err := ssc.podsForNode(n.Name)
		if err != nil {
			return false, err
		}

		type nonmatch struct {
			*api.Pod
			containerTimestamp time.Time
		}
		var (
			execContainerID   = parsed.(string)
			removePods        []*api.Pod
			matchingPods      = make([]*api.Pod, 0, len(pods)) // the most likely case
			matchingTimestamp time.Time
			newestNonMatch    time.Time
			nonmatchPods      []nonmatch
		)
		for _, obj := range pods {
			pod := obj.(*api.Pod)
			podContainer, timestamp, ok := podutil.UUIDAnnotation(pod)
			if !ok {
				// bad pod! missing properly formatted annotation
				log.Warningf("pod missing mesos container UUID annotation?! %v/%v", pod.Namespace, pod.Name)
				removePods = append(removePods, pod)
			} else if podContainer == execContainerID {
				matchingPods = append(matchingPods, pod)
				matchingTimestamp = timestamp
			} else if timestamp.After(cond.LastTransitionTime.Time) {
				nonmatchPods = append(nonmatchPods, nonmatch{pod, timestamp})
				if timestamp.After(newestNonMatch) {
					newestNonMatch = timestamp
				}
			} else {
				// container ID's don't match, and this is old; delete it!
				removePods = append(removePods, pod)
			}
		}
		if len(matchingPods) > 0 {
			foundNewerNonMatching := false
			for _, pod := range nonmatchPods {
				// only keep the newest non-matching pods
				// else, check that this is actually newer than matchingPods
				if pod.containerTimestamp.Before(newestNonMatch) {
					removePods = append(removePods, pod.Pod)
				} else if pod.containerTimestamp.After(matchingTimestamp) {
					foundNewerNonMatching = true
				}
			}
			if foundNewerNonMatching {
				// matchingPods are stale, remove them
				removePods = append(removePods, matchingPods...)
			}
		}
		ssc.deletePods(removePods)

		// we probably can't take action on the following, because this func is only
		// invoked IF the heartbeat is current!!!
		// 2) heartbeat != current --> uuid is stale
		//   ?? kubelet is busy/deadlocked
		//   ?? kubelet is network-partitioned
		//   ?? kubelet died "uncleanly"
		//   ?? new kubelet launched (cycled/upgraded) but we haven't observed a status update yet
		//   a) delete all pods on the node, except non-matching latest newer UUID;timestamp within grace period
	} else {
		// 1) new slave on old node (node bounced)? start evicting pods
		//   ?? node bounced/cycled?
		//     .. if it was gone long enough k8s may have removed it
		//     .. it could come back before all pods assigned to it were deleted
		//   ?? kubelet launched but we haven't observed a status update yet
		//   a) delete all pods on the node, except newest UUID;timestamp within grace period
		// 2) new slave on new node? .. no garbage, no kubelet
		//   ?? kubelet launched but we haven't observed a status update yet
		//   a) delete all pods on the node, except newest UUID;timestamp within grace period
	}
	return false, nil
}

func (ssc *slaveStatusController) deletePods(_ []*api.Pod) {
	// TODO(jdef) implement me
}

func (ssc *slaveStatusController) checkWithoutKubelet(node *api.Node, slavesWithoutKubelet sets.String) (bool, error) {
	if !slavesWithoutKubelet.Has(node.Spec.ExternalID) {
		// let the kubelet do its job updating the status, or the
		// node controller will remove this node if the node does not even
		// exist anymore
		return false, nil
	}
	return true, ssc.setWithoutKubeletStatus(node)
}

func (ssc *slaveStatusController) setWithoutKubeletStatus(n *api.Node) error {
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := ssc.tryUpdateStatus(n); err != nil && !errors.IsConflict(err) {
			log.Errorf("Error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("Update node status exceeds retry count")
}

// nodeWithUpdatedStatus clones the given node and updates the NodeReady condition.
// The updated node is return and a boolean indicating whether the node was changed
// at all.
func (ssc *slaveStatusController) nodeWithUpdatedStatus(n *api.Node) (*api.Node, bool, error) {
	readyCondition := getCondition(&n.Status, api.NodeReady)
	currentTime := unversioned.NewTime(ssc.clockFunc())

	if readyCondition != nil && !currentTime.After(readyCondition.LastHeartbeatTime.Add(ssc.gracePeriod)) {
		return n, false, nil
	}

	clone, err := api.Scheme.DeepCopy(n)
	if err != nil {
		return nil, false, err
	}
	n = clone.(*api.Node)

	newNodeReadyCondition := api.NodeCondition{
		Type:              api.NodeReady,
		Status:            api.ConditionTrue,
		Reason:            node.SlaveReadyReason,
		Message:           node.SlaveReadyMessage,
		LastHeartbeatTime: currentTime,
	}
	newRunningExecutorCondition := api.NodeCondition{
		Type:              node.RunningExecutorCondition,
		Status:            api.ConditionFalse,
		LastHeartbeatTime: currentTime,
	}

	updateCondition(n, newNodeReadyCondition, newRunningExecutorCondition)
	return n, true, nil
}

// updateCondition assumes that the LastHeartbeatTime has been updated
func updateCondition(n *api.Node, conditions ...api.NodeCondition) {
outerLoop:
	for _, targetCondition := range conditions {
		for i := range n.Status.Conditions {
			c := &n.Status.Conditions[i]
			if c.Type == targetCondition.Type {
				if c.Status == targetCondition.Status {
					// preserve last transition time while the status is unchanged
					targetCondition.LastTransitionTime = c.LastTransitionTime
				} else {
					targetCondition.LastTransitionTime = targetCondition.LastHeartbeatTime
				}
				n.Status.Conditions[i] = targetCondition
				continue outerLoop
			}
		}

		targetCondition.LastTransitionTime = targetCondition.LastHeartbeatTime
		n.Status.Conditions = append(n.Status.Conditions, targetCondition)
	}
}

// tryUpdateStatus updates the status of the given node and tries to persist that
// on the apiserver
func (ssc *slaveStatusController) tryUpdateStatus(n *api.Node) error {
	n, updated, err := ssc.nodeWithUpdatedStatus(n)
	if err != nil {
		return err
	}
	if !updated {
		return nil
	}

	_, err = ssc.nodes.Nodes().UpdateStatus(n)
	return err
}

// getCondition returns a condition object for the specific condition
// type, nil if the condition is not set.
func getCondition(status *api.NodeStatus, conditionType api.NodeConditionType) *api.NodeCondition {
	if status == nil {
		return nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return &status.Conditions[i]
		}
	}
	return nil
}
