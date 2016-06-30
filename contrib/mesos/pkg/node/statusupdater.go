/*
Copyright 2015 The Kubernetes Authors.

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

	log "github.com/golang/glog"
	"k8s.io/kubernetes/cmd/kubelet/app/options"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/mesos"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	nodeStatusUpdateRetry = 5
	slaveReadyReason      = "SlaveReady"
	slaveReadyMessage     = "mesos reports ready status"
)

type StatusUpdater struct {
	client          *clientset.Clientset
	relistPeriod    time.Duration
	heartBeatPeriod time.Duration
	nowFunc         func() time.Time
}

func NewStatusUpdater(client *clientset.Clientset, relistPeriod time.Duration, nowFunc func() time.Time) *StatusUpdater {
	kubecfg := options.NewKubeletServer() // only create to get the config, this is without side-effects
	return &StatusUpdater{
		client:          client,
		relistPeriod:    relistPeriod,
		heartBeatPeriod: kubecfg.NodeStatusUpdateFrequency.Duration,
		nowFunc:         nowFunc,
	}
}

func (u *StatusUpdater) Run(terminate <-chan struct{}) error {
	nodeStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	nodeLW := cache.NewListWatchFromClient(u.client.CoreClient, "nodes", api.NamespaceAll, fields.Everything())
	cache.NewReflector(nodeLW, &api.Node{}, nodeStore, u.relistPeriod).Run()

	monitor := func() {
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
		nodes := nodeStore.List()

		for _, n := range nodes {
			node := n.(*api.Node)
			if !slavesWithoutKubelet.Has(node.Spec.ExternalID) {
				// let the kubelet do its job updating the status, or the
				// node controller will remove this node if the node does not even
				// exist anymore
				continue
			}

			err := u.updateStatus(node)
			if err != nil {
				log.Errorf("Error updating node status: %v", err)
			}
		}
	}

	go runtime.Until(monitor, u.heartBeatPeriod, terminate)
	return nil
}

func (u *StatusUpdater) updateStatus(n *api.Node) error {
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := u.tryUpdateStatus(n); err != nil && !errors.IsConflict(err) {
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
func (u *StatusUpdater) nodeWithUpdatedStatus(n *api.Node) (*api.Node, bool, error) {
	readyCondition := getCondition(&n.Status, api.NodeReady)
	currentTime := unversioned.NewTime(u.nowFunc())

	// avoid flapping by waiting at least twice the kubetlet update frequency, i.e.
	// give the kubelet the chance twice to update the heartbeat. This is necessary
	// because we only poll the Mesos master state.json once in a while and we
	// know that that the information from there can easily be outdated.
	gracePeriod := u.heartBeatPeriod * 2
	if readyCondition != nil && !currentTime.After(readyCondition.LastHeartbeatTime.Add(gracePeriod)) {
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
		Reason:            slaveReadyReason,
		Message:           slaveReadyMessage,
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
func (u *StatusUpdater) tryUpdateStatus(n *api.Node) error {
	n, updated, err := u.nodeWithUpdatedStatus(n)
	if err != nil {
		return err
	}
	if !updated {
		return nil
	}

	_, err = u.client.Nodes().UpdateStatus(n)
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
