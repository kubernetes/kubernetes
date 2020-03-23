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
	"context"
	"fmt"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	appsv1listers "k8s.io/client-go/listers/apps/v1"
	utilpod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	nodepkg "k8s.io/kubernetes/pkg/util/node"

	"k8s.io/klog"
)

// DeletePods will delete all pods from master running on given node,
// and return true if any pods were deleted, or were found pending
// deletion.
func DeletePods(kubeClient clientset.Interface, pods []*v1.Pod, recorder record.EventRecorder, nodeName, nodeUID string, daemonStore appsv1listers.DaemonSetLister) (bool, error) {
	remaining := false
	var updateErrList []error

	if len(pods) > 0 {
		RecordNodeEvent(recorder, nodeName, nodeUID, v1.EventTypeNormal, "DeletingAllPods", fmt.Sprintf("Deleting all Pods from Node %v.", nodeName))
	}

	for i := range pods {
		// Defensive check, also needed for tests.
		if pods[i].Spec.NodeName != nodeName {
			continue
		}

		// Pod will be modified, so making copy is required.
		pod := pods[i].DeepCopy()
		// Set reason and message in the pod object.
		if _, err := SetPodTerminationReason(kubeClient, pod, nodeName); err != nil {
			if apierrors.IsConflict(err) {
				updateErrList = append(updateErrList,
					fmt.Errorf("update status failed for pod %q: %v", format.Pod(pod), err))
				continue
			}
		}
		// if the pod has already been marked for deletion, we still return true that there are remaining pods.
		if pod.DeletionGracePeriodSeconds != nil {
			remaining = true
			continue
		}
		// if the pod is managed by a daemonset, ignore it
		if _, err := daemonStore.GetPodDaemonSets(pod); err == nil {
			// No error means at least one daemonset was found
			continue
		}

		klog.V(2).Infof("Starting deletion of pod %v/%v", pod.Namespace, pod.Name)
		recorder.Eventf(pod, v1.EventTypeNormal, "NodeControllerEviction", "Marking for deletion Pod %s from Node %s", pod.Name, nodeName)
		if err := kubeClient.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{}); err != nil {
			if apierrors.IsNotFound(err) {
				// NotFound error means that pod was already deleted.
				// There is nothing left to do with this pod.
				continue
			}
			return false, err
		}
		remaining = true
	}

	if len(updateErrList) > 0 {
		return false, utilerrors.NewAggregate(updateErrList)
	}
	return remaining, nil
}

// SetPodTerminationReason attempts to set a reason and message in the
// pod status, updates it in the apiserver, and returns an error if it
// encounters one.
func SetPodTerminationReason(kubeClient clientset.Interface, pod *v1.Pod, nodeName string) (*v1.Pod, error) {
	if pod.Status.Reason == nodepkg.NodeUnreachablePodReason {
		return pod, nil
	}

	pod.Status.Reason = nodepkg.NodeUnreachablePodReason
	pod.Status.Message = fmt.Sprintf(nodepkg.NodeUnreachablePodMessage, nodeName, pod.Name)

	var updatedPod *v1.Pod
	var err error
	if updatedPod, err = kubeClient.CoreV1().Pods(pod.Namespace).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{}); err != nil {
		return nil, err
	}
	return updatedPod, nil
}

// MarkPodsNotReady updates ready status of given pods running on
// given node from master return true if success
func MarkPodsNotReady(kubeClient clientset.Interface, pods []*v1.Pod, nodeName string) error {
	klog.V(2).Infof("Update ready status of pods on node [%v]", nodeName)

	errMsg := []string{}
	for i := range pods {
		// Defensive check, also needed for tests.
		if pods[i].Spec.NodeName != nodeName {
			continue
		}

		// Pod will be modified, so making copy is required.
		pod := pods[i].DeepCopy()
		for _, cond := range pod.Status.Conditions {
			if cond.Type == v1.PodReady {
				cond.Status = v1.ConditionFalse
				if !utilpod.UpdatePodCondition(&pod.Status, &cond) {
					break
				}
				klog.V(2).Infof("Updating ready status of pod %v to false", pod.Name)
				_, err := kubeClient.CoreV1().Pods(pod.Namespace).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{})
				if err != nil {
					if apierrors.IsNotFound(err) {
						// NotFound error means that pod was already deleted.
						// There is nothing left to do with this pod.
						continue
					}
					klog.Warningf("Failed to update status for pod %q: %v", format.Pod(pod), err)
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

// RecordNodeEvent records a event related to a node.
func RecordNodeEvent(recorder record.EventRecorder, nodeName, nodeUID, eventtype, reason, event string) {
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       nodeName,
		UID:        types.UID(nodeUID),
		Namespace:  "",
	}
	klog.V(2).Infof("Recording %s event message for node %s", event, nodeName)
	recorder.Eventf(ref, eventtype, reason, "Node %s event: %s", nodeName, event)
}

// RecordNodeStatusChange records a event related to a node status change. (Common to lifecycle and ipam)
func RecordNodeStatusChange(recorder record.EventRecorder, node *v1.Node, newStatus string) {
	ref := &v1.ObjectReference{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       node.Name,
		UID:        node.UID,
		Namespace:  "",
	}
	klog.V(2).Infof("Recording status change %s event message for node %s", newStatus, node.Name)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	recorder.Eventf(ref, v1.EventTypeNormal, newStatus, "Node %s status is now: %s", node.Name, newStatus)
}

// SwapNodeControllerTaint returns true in case of success and false
// otherwise.
func SwapNodeControllerTaint(kubeClient clientset.Interface, taintsToAdd, taintsToRemove []*v1.Taint, node *v1.Node) bool {
	for _, taintToAdd := range taintsToAdd {
		now := metav1.Now()
		taintToAdd.TimeAdded = &now
	}

	err := controller.AddOrUpdateTaintOnNode(kubeClient, node.Name, taintsToAdd...)
	if err != nil {
		utilruntime.HandleError(
			fmt.Errorf(
				"unable to taint %+v unresponsive Node %q: %v",
				taintsToAdd,
				node.Name,
				err))
		return false
	}
	klog.V(4).Infof("Added %+v Taint to Node %v", taintsToAdd, node.Name)

	err = controller.RemoveTaintOffNode(kubeClient, node.Name, node, taintsToRemove...)
	if err != nil {
		utilruntime.HandleError(
			fmt.Errorf(
				"unable to remove %+v unneeded taint from unresponsive Node %q: %v",
				taintsToRemove,
				node.Name,
				err))
		return false
	}
	klog.V(4).Infof("Made sure that Node %+v has no %v Taint", node.Name, taintsToRemove)

	return true
}

// AddOrUpdateLabelsOnNode updates the labels on the node and returns true on
// success and false on failure.
func AddOrUpdateLabelsOnNode(kubeClient clientset.Interface, labelsToUpdate map[string]string, node *v1.Node) bool {
	err := controller.AddOrUpdateLabelsOnNode(kubeClient, node.Name, labelsToUpdate)
	if err != nil {
		utilruntime.HandleError(
			fmt.Errorf(
				"unable to update labels %+v for Node %q: %v",
				labelsToUpdate,
				node.Name,
				err))
		return false
	}
	klog.V(4).Infof("Updated labels %+v to Node %v", labelsToUpdate, node.Name)
	return true
}

// CreateAddNodeHandler creates an add node handler.
func CreateAddNodeHandler(f func(node *v1.Node) error) func(obj interface{}) {
	return func(originalObj interface{}) {
		node := originalObj.(*v1.Node).DeepCopy()
		if err := f(node); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing Node Add: %v", err))
		}
	}
}

// CreateUpdateNodeHandler creates a node update handler. (Common to lifecycle and ipam)
func CreateUpdateNodeHandler(f func(oldNode, newNode *v1.Node) error) func(oldObj, newObj interface{}) {
	return func(origOldObj, origNewObj interface{}) {
		node := origNewObj.(*v1.Node).DeepCopy()
		prevNode := origOldObj.(*v1.Node).DeepCopy()

		if err := f(prevNode, node); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing Node Add/Delete: %v", err))
		}
	}
}

// CreateDeleteNodeHandler creates a delete node handler. (Common to lifecycle and ipam)
func CreateDeleteNodeHandler(f func(node *v1.Node) error) func(obj interface{}) {
	return func(originalObj interface{}) {
		originalNode, isNode := originalObj.(*v1.Node)
		// We can get DeletedFinalStateUnknown instead of *v1.Node here and
		// we need to handle that correctly. #34692
		if !isNode {
			deletedState, ok := originalObj.(cache.DeletedFinalStateUnknown)
			if !ok {
				klog.Errorf("Received unexpected object: %v", originalObj)
				return
			}
			originalNode, ok = deletedState.Obj.(*v1.Node)
			if !ok {
				klog.Errorf("DeletedFinalStateUnknown contained non-Node object: %v", deletedState.Obj)
				return
			}
		}
		node := originalNode.DeepCopy()
		if err := f(node); err != nil {
			utilruntime.HandleError(fmt.Errorf("Error while processing Node Add/Delete: %v", err))
		}
	}
}

// GetNodeCondition extracts the provided condition from the given status and returns that.
// Returns nil and -1 if the condition is not present, and the index of the located condition.
func GetNodeCondition(status *v1.NodeStatus, conditionType v1.NodeConditionType) (int, *v1.NodeCondition) {
	if status == nil {
		return -1, nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return i, &status.Conditions[i]
		}
	}
	return -1, nil
}
