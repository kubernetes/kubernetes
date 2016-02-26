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
		go runtime.Until(func() { m.worker() }, m.heartbeatPeriod, terminate)
	}
	<-terminate
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
