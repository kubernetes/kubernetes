/*
Copyright 2016 The Kubernetes Authors.

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
	"context"
	"errors"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	v1lister "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	cloudproviderapi "k8s.io/cloud-provider/api"
	cloudnodeutil "k8s.io/cloud-provider/node/helpers"
	"k8s.io/klog/v2"
)

const (
	deleteNodeEvent = "DeletingNode"
)

var ShutdownTaint = &v1.Taint{
	Key:    cloudproviderapi.TaintNodeShutdown,
	Effect: v1.TaintEffectNoSchedule,
}

// CloudNodeLifecycleController is responsible for deleting/updating kubernetes
// nodes that have been deleted/shutdown on the cloud provider
type CloudNodeLifecycleController struct {
	kubeClient clientset.Interface
	nodeLister v1lister.NodeLister
	recorder   record.EventRecorder

	cloud cloudprovider.Interface

	// Value controlling NodeController monitoring period, i.e. how often does NodeController
	// check node status posted from kubelet. This value should be lower than nodeMonitorGracePeriod
	// set in controller-manager
	nodeMonitorPeriod time.Duration
}

func NewCloudNodeLifecycleController(
	nodeInformer coreinformers.NodeInformer,
	kubeClient clientset.Interface,
	cloud cloudprovider.Interface,
	nodeMonitorPeriod time.Duration) (*CloudNodeLifecycleController, error) {

	eventBroadcaster := record.NewBroadcaster()
	recorder := eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-lifecycle-controller"})
	eventBroadcaster.StartStructuredLogging(0)

	klog.Info("Sending events to api server")
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	if kubeClient == nil {
		return nil, errors.New("kubernetes client is nil")
	}

	if cloud == nil {
		return nil, errors.New("no cloud provider provided")
	}

	_, instancesSupported := cloud.Instances()
	_, instancesV2Supported := cloud.InstancesV2()
	if !instancesSupported && !instancesV2Supported {
		return nil, errors.New("cloud provider does not support instances")
	}

	c := &CloudNodeLifecycleController{
		kubeClient:        kubeClient,
		nodeLister:        nodeInformer.Lister(),
		recorder:          recorder,
		cloud:             cloud,
		nodeMonitorPeriod: nodeMonitorPeriod,
	}

	return c, nil
}

// Run starts the main loop for this controller. Run is blocking so should
// be called via a goroutine
func (c *CloudNodeLifecycleController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	// The following loops run communicate with the APIServer with a worst case complexity
	// of O(num_nodes) per cycle. These functions are justified here because these events fire
	// very infrequently. DO NOT MODIFY this to perform frequent operations.

	// Start a loop to periodically check if any nodes have been
	// deleted or shutdown from the cloudprovider
	wait.Until(c.MonitorNodes, c.nodeMonitorPeriod, stopCh)
}

// MonitorNodes checks to see if nodes in the cluster have been deleted
// or shutdown. If deleted, it deletes the node resource. If shutdown it
// applies a shutdown taint to the node
func (c *CloudNodeLifecycleController) MonitorNodes() {
	nodes, err := c.nodeLister.List(labels.Everything())
	if err != nil {
		klog.Errorf("error listing nodes from cache: %s", err)
		return
	}

	for _, node := range nodes {
		// Default NodeReady status to v1.ConditionUnknown
		status := v1.ConditionUnknown
		if _, c := cloudnodeutil.GetNodeCondition(&node.Status, v1.NodeReady); c != nil {
			status = c.Status
		}

		if status == v1.ConditionTrue {
			// if taint exist remove taint
			err = cloudnodeutil.RemoveTaintOffNode(c.kubeClient, node.Name, node, ShutdownTaint)
			if err != nil {
				klog.Errorf("error patching node taints: %v", err)
			}
			continue
		}

		// At this point the node has NotReady status, we need to check if the node has been removed
		// from the cloud provider. If node cannot be found in cloudprovider, then delete the node
		exists, err := ensureNodeExistsByProviderID(context.TODO(), c.cloud, node)
		if err != nil {
			klog.Errorf("error checking if node %s exists: %v", node.Name, err)
			continue
		}

		if !exists {
			// Current node does not exist, we should delete it, its taints do not matter anymore

			klog.V(2).Infof("deleting node since it is no longer present in cloud provider: %s", node.Name)

			ref := &v1.ObjectReference{
				Kind:      "Node",
				Name:      node.Name,
				UID:       types.UID(node.UID),
				Namespace: "",
			}

			c.recorder.Eventf(ref, v1.EventTypeNormal, deleteNodeEvent,
				"Deleting node %s because it does not exist in the cloud provider", node.Name)

			if err := c.kubeClient.CoreV1().Nodes().Delete(context.TODO(), node.Name, metav1.DeleteOptions{}); err != nil {
				klog.Errorf("unable to delete node %q: %v", node.Name, err)
			}
		} else {
			// Node exists. We need to check this to get taint working in similar in all cloudproviders
			// current problem is that shutdown nodes are not working in similar way ie. all cloudproviders
			// does not delete node from kubernetes cluster when instance it is shutdown see issue #46442
			shutdown, err := shutdownInCloudProvider(context.TODO(), c.cloud, node)
			if err != nil {
				klog.Errorf("error checking if node %s is shutdown: %v", node.Name, err)
			}

			if shutdown && err == nil {
				// if node is shutdown add shutdown taint
				err = cloudnodeutil.AddOrUpdateTaintOnNode(c.kubeClient, node.Name, ShutdownTaint)
				if err != nil {
					klog.Errorf("failed to apply shutdown taint to node %s, it may have been deleted.", node.Name)
				}
			}
		}
	}
}

// shutdownInCloudProvider returns true if the node is shutdown on the cloud provider
func shutdownInCloudProvider(ctx context.Context, cloud cloudprovider.Interface, node *v1.Node) (bool, error) {
	if instanceV2, ok := cloud.InstancesV2(); ok {
		return instanceV2.InstanceShutdown(ctx, node)
	}

	instances, ok := cloud.Instances()
	if !ok {
		return false, errors.New("cloud provider does not support instances")
	}

	shutdown, err := instances.InstanceShutdownByProviderID(ctx, node.Spec.ProviderID)
	if err == cloudprovider.NotImplemented {
		return false, nil
	}

	return shutdown, err
}

// ensureNodeExistsByProviderID checks if the instance exists by the provider id,
// If provider id in spec is empty it calls instanceId with node name to get provider id
func ensureNodeExistsByProviderID(ctx context.Context, cloud cloudprovider.Interface, node *v1.Node) (bool, error) {
	if instanceV2, ok := cloud.InstancesV2(); ok {
		return instanceV2.InstanceExists(ctx, node)
	}

	instances, ok := cloud.Instances()
	if !ok {
		return false, errors.New("instances interface not supported in the cloud provider")
	}

	providerID := node.Spec.ProviderID
	if providerID == "" {
		var err error
		providerID, err = instances.InstanceID(ctx, types.NodeName(node.Name))
		if err != nil {
			if err == cloudprovider.InstanceNotFound {
				return false, nil
			}
			return false, err
		}

		if providerID == "" {
			klog.Warningf("Cannot find valid providerID for node name %q, assuming non existence", node.Name)
			return false, nil
		}
	}

	return instances.InstanceExistsByProviderID(ctx, providerID)
}
