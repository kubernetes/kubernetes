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

	Controller struct {
		client      *clientset.Clientset
		nowFunc     func() time.Time
		monitors    []*monitor
		addFuncs    []func(_ interface{})
		updateFuncs []func(_, _ interface{})
		deleteFuncs []func(_ interface{})

		// Node framework and store
		nodeController *framework.Controller
		nodeStore      cache.StoreToNodeLister
	}

	// Option is a functional option for Controller
	Option func(*Controller)
)

func NowFunc(f func() time.Time) Option {
	return func(c *Controller) {
		c.nowFunc = f
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
		client:  client,
		nowFunc: time.Now,
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
			framework.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					for _, f := range c.addFuncs {
						f(obj)
					}
				},
				UpdateFunc: func(oldObj, newObj interface{}) {
					for _, f := range c.updateFuncs {
						f(oldObj, newObj)
					}
				},
				DeleteFunc: func(obj interface{}) {
					for _, f := range c.deleteFuncs {
						f(obj)
					}
				},
			},
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

// watch for changes to RunningExecutorCondition
type mesosContainerController struct {
	store       cache.Store
	gracePeriod time.Duration // nodes w/ RunningExecutorCondition updates older than lastHeartbeat+gracePeriod are considered stale
}

func (c *Controller) newMesosContainerController() *mesosContainerController {
	controller := &mesosContainerController{
		store: c.nodeStore.Store,
	}
}

func (mcc *mesosContainerController) checkCondition() {
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
	nodes := ssc.store.List()
	for _, obj := range nodes {
		n := obj.(*api.Node)
		cond := getCondition(&n.Status, node.RunningExecutorCondition)
		if cond != nil {
			// 1) heartbeat is current --> uuid is possibly stale
			//   ?? kubelet cycled but we haven't observed a status update yet
			//   a) delete all pods on the node, except non-matching latest newer UUID;timestamp within grace period or else matching UUID's
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
	}
}

type slaveStatusController struct {
	*monitor
	gracePeriod time.Duration
	store       cache.Store
	nowFunc     func() time.Time
	nodes       unversionedcore.NodesGetter
}

func (c *Controller) newSlaveStatusController() *slaveStatusController {
	kubecfg := options.NewKubeletServer() // only create to get the config, this is without side-effects
	controller := &slaveStatusController{
		monitor: &monitor{
			heartbeatPeriod: kubecfg.NodeStatusUpdateFrequency.Duration,
		},
		nowFunc: c.nowFunc,
		store:   c.nodeStore.Store,
	}

	// optional for easier unit testing
	if c.client != nil {
		controller.nodes = c.client.Core()
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
	nodes := ssc.store.List()

	for _, n := range nodes {
		node := n.(*api.Node)
		if !slavesWithoutKubelet.Has(node.Spec.ExternalID) {
			// let the kubelet do its job updating the status, or the
			// node controller will remove this node if the node does not even
			// exist anymore
			continue
		}

		err := ssc.updateStatus(node)
		if err != nil {
			log.Errorf("Error updating node status: %v", err)
		}
	}
}

func (ssc *slaveStatusController) updateStatus(n *api.Node) error {
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
	currentTime := unversioned.NewTime(ssc.nowFunc())

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

	found := false
	for i := range n.Status.Conditions {
		c := &n.Status.Conditions[i]
		if c.Type == api.NodeReady {
			if c.Status == newNodeReadyCondition.Status {
				newNodeReadyCondition.LastTransitionTime = c.LastTransitionTime
			} else {
				newNodeReadyCondition.LastTransitionTime = currentTime
			}
			n.Status.Conditions[i] = newNodeReadyCondition
			found = true
			break
		}
	}

	if !found {
		newNodeReadyCondition.LastTransitionTime = currentTime
		n.Status.Conditions = append(n.Status.Conditions, newNodeReadyCondition)
	}

	return n, true, nil
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
