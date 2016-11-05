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

package cloud

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/types"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
)

type CloudNodeController struct {
	nodeInformer informers.NodeInformer
	kubeClient   clientset.Interface
	recorder     record.EventRecorder

	cloud cloudprovider.Interface

	// Value controlling NodeController monitoring period, i.e. how often does NodeController
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod
	// set in controller-manager
	nodeMonitorPeriod time.Duration
}

const (
	// nodeStatusUpdateRetry controls the number of retries of writing NodeStatus update.
	nodeStatusUpdateRetry = 5

	// The amount of time the nodecontroller should sleep between retrying NodeStatus updates
	retrySleepTime = 20 * time.Millisecond
)

func NewCloudNodeController(
	nodeInformer informers.NodeInformer,
	kubeClient clientset.Interface,
	cloud cloudprovider.Interface,
	nodeMonitorPeriod time.Duration) (*CloudNodeController, error) {

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(api.EventSource{Component: "cloudcontrollermanager"})
	eventBroadcaster.StartLogging(glog.Infof)
	if kubeClient != nil {
		glog.V(0).Infof("Sending events to api server.")
		eventBroadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})
	} else {
		glog.V(0).Infof("No api server defined - no events will be sent to API server.")
	}

	cnc := &CloudNodeController{
		nodeInformer:      nodeInformer,
		kubeClient:        kubeClient,
		recorder:          recorder,
		cloud:             cloud,
		nodeMonitorPeriod: nodeMonitorPeriod,
	}
	return cnc, nil
}

// This controller deletes a node if kubelet is not reporting
// and the node is gone from the cloud provider.
func (cnc *CloudNodeController) Run() {
	go func() {
		defer utilruntime.HandleCrash()

		go wait.Until(func() {
			nodes, err := cnc.kubeClient.Core().Nodes().List(api.ListOptions{ResourceVersion: "0"})
			if err != nil {
				glog.Errorf("Error monitoring node status: %v", err)
			}

			for i := range nodes.Items {
				var currentReadyCondition *api.NodeCondition
				node := &nodes.Items[i]
				for rep := 0; rep < nodeStatusUpdateRetry; rep++ {
					_, currentReadyCondition = api.GetNodeCondition(&node.Status, api.NodeReady)
					if err == nil {
						break
					}
					name := node.Name
					node, err = cnc.kubeClient.Core().Nodes().Get(name)
					if err != nil {
						glog.Errorf("Failed while getting a Node to retry updating NodeStatus. Probably Node %s was deleted.", name)
						break
					}
					time.Sleep(retrySleepTime)
				}
				if err != nil {
					glog.Errorf("Update status of Node %v from CloudNodeController exceeds retry count.", node.Name)
					continue
				}
				if currentReadyCondition != nil {
					if currentReadyCondition.Status != api.ConditionTrue {
						instances, ok := cnc.cloud.Instances()
						if !ok {
							glog.Errorf("cloud provider does not support instances.")
							continue
						}
						// Check with the cloud provider to see if the node still exists. If it
						// doesn't, delete the node immediately.
						if _, err := instances.ExternalID(types.NodeName(node.Name)); err != nil {
							if err == cloudprovider.InstanceNotFound {
								glog.V(2).Infof("Deleting node no longer present in cloud provider: %s", node.Name)
								ref := &api.ObjectReference{
									Kind:      "Node",
									Name:      node.Name,
									UID:       types.UID(node.UID),
									Namespace: "",
								}
								glog.V(2).Infof("Recording %s event message for node %s", "DeletingNode", node.Name)
								cnc.recorder.Eventf(ref, api.EventTypeNormal, fmt.Sprintf("Deleting Node %v because it's not present according to cloud provider", node.Name), "Node %s event: %s", node.Name, "DeletingNode")
								go func(nodeName string) {
									defer utilruntime.HandleCrash()
									if err := cnc.kubeClient.Core().Nodes().Delete(node.Name, nil); err != nil {
										glog.Errorf("unable to delete node %q: %v", node.Name, err)
									}
								}(node.Name)
							}
							glog.Errorf("Error getting node data from cloud: %v", err)
						}
					}
				}
			}
		}, cnc.nodeMonitorPeriod, wait.NeverStop)
	}()
}
