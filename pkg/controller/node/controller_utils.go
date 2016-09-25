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

package node

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/types"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/version"

	"github.com/golang/glog"
)

const (
	// Number of Nodes that needs to be in the cluster for it to be treated as "large"
	LargeClusterThreshold = 20
)

// cleanupOrphanedPods deletes pods that are bound to nodes that don't
// exist.
func cleanupOrphanedPods(pods []*api.Pod, nodeStore cache.Store, forcefulDeletePodFunc func(*api.Pod) error) {
	for _, pod := range pods {
		if pod.Spec.NodeName == "" {
			continue
		}
		if _, exists, _ := nodeStore.GetByKey(pod.Spec.NodeName); exists {
			continue
		}
		if err := forcefulDeletePodFunc(pod); err != nil {
			utilruntime.HandleError(err)
		}
	}
}

// deletePods will delete all pods from master running on given node, and return true
// if any pods were deleted, or were found pending deletion.
func deletePods(kubeClient clientset.Interface, recorder record.EventRecorder, nodeName, nodeUID string, daemonStore cache.StoreToDaemonSetLister) (bool, error) {
	remaining := false
	selector := fields.OneTermEqualSelector(api.PodHostField, nodeName)
	options := api.ListOptions{FieldSelector: selector}
	pods, err := kubeClient.Core().Pods(api.NamespaceAll).List(options)
	if err != nil {
		return remaining, err
	}

	if len(pods.Items) > 0 {
		recordNodeEvent(recorder, nodeName, nodeUID, api.EventTypeNormal, "DeletingAllPods", fmt.Sprintf("Deleting all Pods from Node %v.", nodeName))
	}

	for _, pod := range pods.Items {
		// Defensive check, also needed for tests.
		if pod.Spec.NodeName != nodeName {
			continue
		}
		// if the pod has already been marked for deletion, we still return true that there are remaining pods.
		if pod.DeletionGracePeriodSeconds != nil {
			remaining = true
			continue
		}
		// if the pod is managed by a daemonset, ignore it
		_, err := daemonStore.GetPodDaemonSets(&pod)
		if err == nil { // No error means at least one daemonset was found
			continue
		}

		glog.V(2).Infof("Starting deletion of pod %v", pod.Name)
		recorder.Eventf(&pod, api.EventTypeNormal, "NodeControllerEviction", "Marking for deletion Pod %s from Node %s", pod.Name, nodeName)
		if err := kubeClient.Core().Pods(pod.Namespace).Delete(pod.Name, nil); err != nil {
			return false, err
		}
		remaining = true
	}
	return remaining, nil
}

func forcefullyDeletePod(c clientset.Interface, pod *api.Pod) error {
	var zero int64
	err := c.Core().Pods(pod.Namespace).Delete(pod.Name, &api.DeleteOptions{GracePeriodSeconds: &zero})
	if err == nil {
		glog.V(4).Infof("forceful deletion of %s succeeded", pod.Name)
	}
	return err
}

// forcefullyDeleteNode immediately deletes all pods on the node, and then
// deletes the node itself.
func forcefullyDeleteNode(kubeClient clientset.Interface, nodeName string, forcefulDeletePodFunc func(*api.Pod) error) error {
	selector := fields.OneTermEqualSelector(api.PodHostField, nodeName)
	options := api.ListOptions{FieldSelector: selector}
	pods, err := kubeClient.Core().Pods(api.NamespaceAll).List(options)
	if err != nil {
		return fmt.Errorf("unable to list pods on node %q: %v", nodeName, err)
	}
	for _, pod := range pods.Items {
		if pod.Spec.NodeName != nodeName {
			continue
		}
		if err := forcefulDeletePodFunc(&pod); err != nil {
			return fmt.Errorf("unable to delete pod %q on node %q: %v", pod.Name, nodeName, err)
		}
	}
	if err := kubeClient.Core().Nodes().Delete(nodeName, nil); err != nil {
		return fmt.Errorf("unable to delete node %q: %v", nodeName, err)
	}
	return nil
}

// maybeDeleteTerminatingPod non-gracefully deletes pods that are terminating
// that should not be gracefully terminated.
func (nc *NodeController) maybeDeleteTerminatingPod(obj interface{}) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return
	}

	// consider only terminating pods
	if pod.DeletionTimestamp == nil {
		return
	}

	// delete terminating pods that have not yet been scheduled
	if len(pod.Spec.NodeName) == 0 {
		utilruntime.HandleError(nc.forcefullyDeletePod(pod))
		return
	}

	nodeObj, found, err := nc.nodeStore.Store.GetByKey(pod.Spec.NodeName)
	if err != nil {
		// this can only happen if the Store.KeyFunc has a problem creating
		// a key for the pod. If it happens once, it will happen again so
		// don't bother requeuing the pod.
		utilruntime.HandleError(err)
		return
	}

	// delete terminating pods that have been scheduled on
	// nonexistent nodes
	if !found {
		glog.Warningf("Unable to find Node: %v, deleting all assigned Pods.", pod.Spec.NodeName)
		utilruntime.HandleError(nc.forcefullyDeletePod(pod))
		return
	}

	// delete terminating pods that have been scheduled on
	// nodes that do not support graceful termination
	// TODO(mikedanese): this can be removed when we no longer
	// guarantee backwards compatibility of master API to kubelets with
	// versions less than 1.1.0
	node := nodeObj.(*api.Node)
	v, err := version.Parse(node.Status.NodeInfo.KubeletVersion)
	if err != nil {
		glog.V(0).Infof("couldn't parse verions %q of minion: %v", node.Status.NodeInfo.KubeletVersion, err)
		utilruntime.HandleError(nc.forcefullyDeletePod(pod))
		return
	}
	if gracefulDeletionVersion.GT(v) {
		utilruntime.HandleError(nc.forcefullyDeletePod(pod))
		return
	}
}

// update ready status of all pods running on given node from master
// return true if success
func markAllPodsNotReady(kubeClient clientset.Interface, node *api.Node) error {
	// Don't set pods to NotReady if the kubelet is running a version that
	// doesn't understand how to correct readiness.
	// TODO: Remove this check when we no longer guarantee backward compatibility
	// with node versions < 1.2.0.
	if nodeRunningOutdatedKubelet(node) {
		return nil
	}
	nodeName := node.Name
	glog.V(2).Infof("Update ready status of pods on node [%v]", nodeName)
	opts := api.ListOptions{FieldSelector: fields.OneTermEqualSelector(api.PodHostField, nodeName)}
	pods, err := kubeClient.Core().Pods(api.NamespaceAll).List(opts)
	if err != nil {
		return err
	}

	errMsg := []string{}
	for _, pod := range pods.Items {
		// Defensive check, also needed for tests.
		if pod.Spec.NodeName != nodeName {
			continue
		}

		for i, cond := range pod.Status.Conditions {
			if cond.Type == api.PodReady {
				pod.Status.Conditions[i].Status = api.ConditionFalse
				glog.V(2).Infof("Updating ready status of pod %v to false", pod.Name)
				_, err := kubeClient.Core().Pods(pod.Namespace).UpdateStatus(&pod)
				if err != nil {
					glog.Warningf("Failed to update status for pod %q: %v", format.Pod(&pod), err)
					errMsg = append(errMsg, fmt.Sprintf("%v", err))
				}
				break
			}
		}
	}
	if len(errMsg) == 0 {
		return nil
	}
	return fmt.Errorf("%v", strings.Join(errMsg, "; "))
}

// nodeRunningOutdatedKubelet returns true if the kubeletVersion reported
// in the nodeInfo of the given node is "outdated", meaning < 1.2.0.
// Older versions were inflexible and modifying pod.Status directly through
// the apiserver would result in unexpected outcomes.
func nodeRunningOutdatedKubelet(node *api.Node) bool {
	v, err := version.Parse(node.Status.NodeInfo.KubeletVersion)
	if err != nil {
		glog.Errorf("couldn't parse version %q of node %v", node.Status.NodeInfo.KubeletVersion, err)
		return true
	}
	if podStatusReconciliationVersion.GT(v) {
		glog.Infof("Node %v running kubelet at (%v) which is less than the minimum version that allows nodecontroller to mark pods NotReady (%v).", node.Name, v, podStatusReconciliationVersion)
		return true
	}
	return false
}

func nodeExistsInCloudProvider(cloud cloudprovider.Interface, nodeName string) (bool, error) {
	instances, ok := cloud.Instances()
	if !ok {
		return false, fmt.Errorf("%v", ErrCloudInstance)
	}
	if _, err := instances.ExternalID(nodeName); err != nil {
		if err == cloudprovider.InstanceNotFound {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

func recordNodeEvent(recorder record.EventRecorder, nodeName, nodeUID, eventtype, reason, event string) {
	ref := &api.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeUID),
		Namespace: "",
	}
	glog.V(2).Infof("Recording %s event message for node %s", event, nodeName)
	recorder.Eventf(ref, eventtype, reason, "Node %s event: %s", nodeName, event)
}

func recordNodeStatusChange(recorder record.EventRecorder, node *api.Node, new_status string) {
	ref := &api.ObjectReference{
		Kind:      "Node",
		Name:      node.Name,
		UID:       node.UID,
		Namespace: "",
	}
	glog.V(2).Infof("Recording status change %s event message for node %s", new_status, node.Name)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	recorder.Eventf(ref, api.EventTypeNormal, new_status, "Node %s status is now: %s", node.Name, new_status)
}

// terminatePods will ensure all pods on the given node that are in terminating state are eventually
// cleaned up. Returns true if the node has no pods in terminating state, a duration that indicates how
// long before we should check again (the next deadline for a pod to complete), or an error.
func terminatePods(kubeClient clientset.Interface, recorder record.EventRecorder, nodeName string, nodeUID string, since time.Time, maxGracePeriod time.Duration) (bool, time.Duration, error) {
	// the time before we should try again
	nextAttempt := time.Duration(0)
	// have we deleted all pods
	complete := true

	selector := fields.OneTermEqualSelector(api.PodHostField, nodeName)
	options := api.ListOptions{FieldSelector: selector}
	pods, err := kubeClient.Core().Pods(api.NamespaceAll).List(options)
	if err != nil {
		return false, nextAttempt, err
	}

	now := time.Now()
	elapsed := now.Sub(since)
	for _, pod := range pods.Items {
		// Defensive check, also needed for tests.
		if pod.Spec.NodeName != nodeName {
			continue
		}
		// only clean terminated pods
		if pod.DeletionGracePeriodSeconds == nil {
			continue
		}

		// the user's requested grace period
		grace := time.Duration(*pod.DeletionGracePeriodSeconds) * time.Second
		if grace > maxGracePeriod {
			grace = maxGracePeriod
		}

		// the time remaining before the pod should have been deleted
		remaining := grace - elapsed
		if remaining < 0 {
			remaining = 0
			glog.V(2).Infof("Removing pod %v after %s grace period", pod.Name, grace)
			recordNodeEvent(recorder, nodeName, nodeUID, api.EventTypeNormal, "TerminatingEvictedPod", fmt.Sprintf("Pod %s has exceeded the grace period for deletion after being evicted from Node %q and is being force killed", pod.Name, nodeName))
			if err := kubeClient.Core().Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0)); err != nil {
				glog.Errorf("Error completing deletion of pod %s: %v", pod.Name, err)
				complete = false
			}
		} else {
			glog.V(2).Infof("Pod %v still terminating, requested grace period %s, %s remaining", pod.Name, grace, remaining)
			complete = false
		}

		if nextAttempt < remaining {
			nextAttempt = remaining
		}
	}
	return complete, nextAttempt, nil
}
