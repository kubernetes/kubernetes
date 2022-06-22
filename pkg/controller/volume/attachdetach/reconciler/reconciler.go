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

// Package reconciler implements interfaces that attempt to reconcile the
// desired state of the with the actual state of the world by triggering
// actions.
package reconciler

import (
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/metrics"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/statusupdater"
	"k8s.io/kubernetes/pkg/features"
	kevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// Reconciler runs a periodic loop to reconcile the desired state of the world with
// the actual state of the world by triggering attach detach operations.
// Note: This is distinct from the Reconciler implemented by the kubelet volume
// manager. This reconciles state for the attach/detach controller. That
// reconciles state for the kubelet volume manager.
type Reconciler interface {
	// Starts running the reconciliation loop which executes periodically, checks
	// if volumes that should be attached are attached and volumes that should
	// be detached are detached. If not, it will trigger attach/detach
	// operations to rectify.
	Run(stopCh <-chan struct{})
}

// NewReconciler returns a new instance of Reconciler that waits loopPeriod
// between successive executions.
// loopPeriod is the amount of time the reconciler loop waits between
// successive executions.
// maxWaitForUnmountDuration is the max amount of time the reconciler will wait
// for the volume to be safely unmounted, after this it will detach the volume
// anyway (to handle crashed/unavailable nodes). If during this time the volume
// becomes used by a new pod, the detach request will be aborted and the timer
// cleared.
func NewReconciler(
	loopPeriod time.Duration,
	maxWaitForUnmountDuration time.Duration,
	syncDuration time.Duration,
	disableReconciliationSync bool,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld,
	attacherDetacher operationexecutor.OperationExecutor,
	nodeStatusUpdater statusupdater.NodeStatusUpdater,
	nodeLister corelisters.NodeLister,
	recorder record.EventRecorder) Reconciler {
	return &reconciler{
		loopPeriod:                loopPeriod,
		maxWaitForUnmountDuration: maxWaitForUnmountDuration,
		syncDuration:              syncDuration,
		disableReconciliationSync: disableReconciliationSync,
		desiredStateOfWorld:       desiredStateOfWorld,
		actualStateOfWorld:        actualStateOfWorld,
		attacherDetacher:          attacherDetacher,
		nodeStatusUpdater:         nodeStatusUpdater,
		nodeLister:                nodeLister,
		timeOfLastSync:            time.Now(),
		recorder:                  recorder,
	}
}

type reconciler struct {
	loopPeriod                time.Duration
	maxWaitForUnmountDuration time.Duration
	syncDuration              time.Duration
	desiredStateOfWorld       cache.DesiredStateOfWorld
	actualStateOfWorld        cache.ActualStateOfWorld
	attacherDetacher          operationexecutor.OperationExecutor
	nodeStatusUpdater         statusupdater.NodeStatusUpdater
	nodeLister                corelisters.NodeLister
	timeOfLastSync            time.Time
	disableReconciliationSync bool
	recorder                  record.EventRecorder
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopPeriod, stopCh)
}

// reconciliationLoopFunc this can be disabled via cli option disableReconciliation.
// It periodically checks whether the attached volumes from actual state
// are still attached to the node and update the status if they are not.
func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {

		rc.reconcile()

		if rc.disableReconciliationSync {
			klog.V(5).Info("Skipping reconciling attached volumes still attached since it is disabled via the command line.")
		} else if rc.syncDuration < time.Second {
			klog.V(5).Info("Skipping reconciling attached volumes still attached since it is set to less than one second via the command line.")
		} else if time.Since(rc.timeOfLastSync) > rc.syncDuration {
			klog.V(5).Info("Starting reconciling attached volumes still attached")
			rc.sync()
		}
	}
}

func (rc *reconciler) sync() {
	defer rc.updateSyncTime()
	rc.syncStates()
}

func (rc *reconciler) updateSyncTime() {
	rc.timeOfLastSync = time.Now()
}

func (rc *reconciler) syncStates() {
	volumesPerNode := rc.actualStateOfWorld.GetAttachedVolumesPerNode()
	rc.attacherDetacher.VerifyVolumesAreAttached(volumesPerNode, rc.actualStateOfWorld)
}

// hasOutOfServiceTaint returns true if the node has out-of-service taint present
// and `NodeOutOfServiceVolumeDetach` feature gate is enabled.
func (rc *reconciler) hasOutOfServiceTaint(nodeName types.NodeName) (bool, error) {
	if utilfeature.DefaultFeatureGate.Enabled(features.NodeOutOfServiceVolumeDetach) {
		node, err := rc.nodeLister.Get(string(nodeName))
		if err != nil {
			return false, err
		}
		return taints.TaintKeyExists(node.Spec.Taints, v1.TaintNodeOutOfService), nil
	}
	return false, nil
}

// nodeIsHealthy returns true if the node looks healthy.
func (rc *reconciler) nodeIsHealthy(nodeName types.NodeName) (bool, error) {
	node, err := rc.nodeLister.Get(string(nodeName))
	if err != nil {
		return false, err
	}
	return nodeutil.IsNodeReady(node), nil
}

func (rc *reconciler) reconcile() {
	// Detaches are triggered before attaches so that volumes referenced by
	// pods that are rescheduled to a different node are detached first.

	// Ensure volumes that should be detached are detached.
	for _, attachedVolume := range rc.actualStateOfWorld.GetAttachedVolumes() {
		if !rc.desiredStateOfWorld.VolumeExists(
			attachedVolume.VolumeName, attachedVolume.NodeName) {

			// Check whether there already exist an operation pending, and don't even
			// try to start an operation if there is already one running.
			// This check must be done before we do any other checks, as otherwise the other checks
			// may pass while at the same time the volume leaves the pending state, resulting in
			// double detach attempts
			// The operation key format is different depending on whether the volume
			// allows multi attach across different nodes.
			if util.IsMultiAttachAllowed(attachedVolume.VolumeSpec) {
				if !rc.attacherDetacher.IsOperationSafeToRetry(attachedVolume.VolumeName, "" /* podName */, attachedVolume.NodeName, operationexecutor.DetachOperationName) {
					klog.V(10).Infof("Operation for volume %q is already running or still in exponential backoff for node %q. Can't start detach", attachedVolume.VolumeName, attachedVolume.NodeName)
					continue
				}
			} else {
				if !rc.attacherDetacher.IsOperationSafeToRetry(attachedVolume.VolumeName, "" /* podName */, "" /* nodeName */, operationexecutor.DetachOperationName) {
					klog.V(10).Infof("Operation for volume %q is already running or still in exponential backoff in the cluster. Can't start detach for %q", attachedVolume.VolumeName, attachedVolume.NodeName)
					continue
				}
			}

			// Because the detach operation updates the ActualStateOfWorld before
			// marking itself complete, it's possible for the volume to be removed
			// from the ActualStateOfWorld between the GetAttachedVolumes() check
			// and the IsOperationPending() check above.
			// Check the ActualStateOfWorld again to avoid issuing an unnecessary
			// detach.
			// See https://github.com/kubernetes/kubernetes/issues/93902
			attachState := rc.actualStateOfWorld.GetAttachState(attachedVolume.VolumeName, attachedVolume.NodeName)
			if attachState == cache.AttachStateDetached {
				klog.V(5).InfoS("Volume detached--skipping", "volume", attachedVolume)
				continue
			}

			// Set the detach request time
			elapsedTime, err := rc.actualStateOfWorld.SetDetachRequestTime(attachedVolume.VolumeName, attachedVolume.NodeName)
			if err != nil {
				klog.Errorf("Cannot trigger detach because it fails to set detach request time with error %v", err)
				continue
			}
			// Check whether timeout has reached the maximum waiting time
			timeout := elapsedTime > rc.maxWaitForUnmountDuration

			isHealthy, err := rc.nodeIsHealthy(attachedVolume.NodeName)
			if err != nil {
				klog.Errorf("failed to get health of node %s: %s", attachedVolume.NodeName, err.Error())
			}

			// Force detach volumes from unhealthy nodes after maxWaitForUnmountDuration.
			forceDetach := !isHealthy && timeout

			hasOutOfServiceTaint, err := rc.hasOutOfServiceTaint(attachedVolume.NodeName)
			if err != nil {
				klog.Errorf("failed to get taint specs for node %s: %s", attachedVolume.NodeName, err.Error())
			}

			// Check whether volume is still mounted. Skip detach if it is still mounted unless force detach timeout
			// or the node has `node.kubernetes.io/out-of-service` taint.
			if attachedVolume.MountedByNode && !forceDetach && !hasOutOfServiceTaint {
				klog.V(5).InfoS("Cannot detach volume because it is still mounted", "volume", attachedVolume)
				continue
			}

			// Before triggering volume detach, mark volume as detached and update the node status
			// If it fails to update node status, skip detach volume
			// If volume detach operation fails, the volume needs to be added back to report as attached so that node status
			// has the correct volume attachment information.
			err = rc.actualStateOfWorld.RemoveVolumeFromReportAsAttached(attachedVolume.VolumeName, attachedVolume.NodeName)
			if err != nil {
				klog.V(5).Infof("RemoveVolumeFromReportAsAttached failed while removing volume %q from node %q with: %v",
					attachedVolume.VolumeName,
					attachedVolume.NodeName,
					err)
			}

			// Update Node Status to indicate volume is no longer safe to mount.
			err = rc.nodeStatusUpdater.UpdateNodeStatusForNode(attachedVolume.NodeName)
			if err != nil {
				// Skip detaching this volume if unable to update node status
				klog.ErrorS(err, "UpdateNodeStatusForNode failed while attempting to report volume as attached", "volume", attachedVolume)
				continue
			}

			// Trigger detach volume which requires verifying safe to detach step
			// If timeout is true, skip verifySafeToDetach check
			// If the node has node.kubernetes.io/out-of-service taint with NoExecute effect, skip verifySafeToDetach check
			klog.V(5).InfoS("Starting attacherDetacher.DetachVolume", "volume", attachedVolume)
			if hasOutOfServiceTaint {
				klog.V(4).Infof("node %q has out-of-service taint", attachedVolume.NodeName)
			}
			verifySafeToDetach := !(timeout || hasOutOfServiceTaint)
			err = rc.attacherDetacher.DetachVolume(attachedVolume.AttachedVolume, verifySafeToDetach, rc.actualStateOfWorld)
			if err == nil {
				if !timeout {
					klog.InfoS("attacherDetacher.DetachVolume started", "volume", attachedVolume)
				} else {
					metrics.RecordForcedDetachMetric()
					klog.InfoS("attacherDetacher.DetachVolume started: this volume is not safe to detach, but maxWaitForUnmountDuration expired, force detaching", "duration", rc.maxWaitForUnmountDuration, "volume", attachedVolume)
				}
			}
			if err != nil {
				// Add volume back to ReportAsAttached if DetachVolume call failed so that node status updater will add it back to VolumeAttached list.
				// This function is also called during executing the volume detach operation in operation_generoator.
				// It is needed here too because DetachVolume call might fail before executing the actual operation in operation_executor (e.g., cannot find volume plugin etc.)
				rc.actualStateOfWorld.AddVolumeToReportAsAttached(attachedVolume.VolumeName, attachedVolume.NodeName)

				if !exponentialbackoff.IsExponentialBackoff(err) {
					// Ignore exponentialbackoff.IsExponentialBackoff errors, they are expected.
					// Log all other errors.
					klog.ErrorS(err, "attacherDetacher.DetachVolume failed to start", "volume", attachedVolume)
				}
			}
		}
	}

	rc.attachDesiredVolumes()

	// Update Node Status
	err := rc.nodeStatusUpdater.UpdateNodeStatuses()
	if err != nil {
		klog.Warningf("UpdateNodeStatuses failed with: %v", err)
	}
}

func (rc *reconciler) attachDesiredVolumes() {
	// Ensure volumes that should be attached are attached.
	for _, volumeToAttach := range rc.desiredStateOfWorld.GetVolumesToAttach() {
		if util.IsMultiAttachAllowed(volumeToAttach.VolumeSpec) {
			// Don't even try to start an operation if there is already one running for the given volume and node.
			if rc.attacherDetacher.IsOperationPending(volumeToAttach.VolumeName, "" /* podName */, volumeToAttach.NodeName) {
				klog.V(10).Infof("Operation for volume %q is already running for node %q. Can't start attach", volumeToAttach.VolumeName, volumeToAttach.NodeName)
				continue
			}
		} else {
			// Don't even try to start an operation if there is already one running for the given volume
			if rc.attacherDetacher.IsOperationPending(volumeToAttach.VolumeName, "" /* podName */, "" /* nodeName */) {
				klog.V(10).Infof("Operation for volume %q is already running. Can't start attach for %q", volumeToAttach.VolumeName, volumeToAttach.NodeName)
				continue
			}
		}

		// Because the attach operation updates the ActualStateOfWorld before
		// marking itself complete, IsOperationPending() must be checked before
		// GetAttachState() to guarantee the ActualStateOfWorld is
		// up-to-date when it's read.
		// See https://github.com/kubernetes/kubernetes/issues/93902
		attachState := rc.actualStateOfWorld.GetAttachState(volumeToAttach.VolumeName, volumeToAttach.NodeName)
		if attachState == cache.AttachStateAttached {
			// Volume/Node exists, touch it to reset detachRequestedTime
			klog.V(5).InfoS("Volume attached--touching", "volume", volumeToAttach)
			rc.actualStateOfWorld.ResetDetachRequestTime(volumeToAttach.VolumeName, volumeToAttach.NodeName)
			continue
		}

		if !util.IsMultiAttachAllowed(volumeToAttach.VolumeSpec) {
			nodes := rc.actualStateOfWorld.GetNodesForAttachedVolume(volumeToAttach.VolumeName)
			if len(nodes) > 0 {
				if !volumeToAttach.MultiAttachErrorReported {
					rc.reportMultiAttachError(volumeToAttach, nodes)
					rc.desiredStateOfWorld.SetMultiAttachError(volumeToAttach.VolumeName, volumeToAttach.NodeName)
				}
				continue
			}
		}

		// Volume/Node doesn't exist, spawn a goroutine to attach it
		klog.V(5).InfoS("Starting attacherDetacher.AttachVolume", "volume", volumeToAttach)
		err := rc.attacherDetacher.AttachVolume(volumeToAttach.VolumeToAttach, rc.actualStateOfWorld)
		if err == nil {
			klog.InfoS("attacherDetacher.AttachVolume started", "volume", volumeToAttach)
		}
		if err != nil && !exponentialbackoff.IsExponentialBackoff(err) {
			// Ignore exponentialbackoff.IsExponentialBackoff errors, they are expected.
			// Log all other errors.
			klog.ErrorS(err, "attacherDetacher.AttachVolume failed to start", "volume", volumeToAttach)
		}
	}
}

// reportMultiAttachError sends events and logs situation that a volume that
// should be attached to a node is already attached to different node(s).
func (rc *reconciler) reportMultiAttachError(volumeToAttach cache.VolumeToAttach, nodes []types.NodeName) {
	// Filter out the current node from list of nodes where the volume is
	// attached.
	// Some methods need []string, some other needs []NodeName, collect both.
	// In theory, these arrays should have always only one element - the
	// controller does not allow more than one attachment. But use array just
	// in case...
	otherNodes := []types.NodeName{}
	otherNodesStr := []string{}
	for _, node := range nodes {
		if node != volumeToAttach.NodeName {
			otherNodes = append(otherNodes, node)
			otherNodesStr = append(otherNodesStr, string(node))
		}
	}

	// Get list of pods that use the volume on the other nodes.
	pods := rc.desiredStateOfWorld.GetVolumePodsOnNodes(otherNodes, volumeToAttach.VolumeName)

	if len(pods) == 0 {
		// We did not find any pods that requests the volume. The pod must have been deleted already.
		simpleMsg, _ := volumeToAttach.GenerateMsg("Multi-Attach error", "Volume is already exclusively attached to one node and can't be attached to another")
		for _, pod := range volumeToAttach.ScheduledPods {
			rc.recorder.Eventf(pod, v1.EventTypeWarning, kevents.FailedAttachVolume, simpleMsg)
		}
		// Log detailed message to system admin
		klog.InfoS("Multi-Attach error: volume is already exclusively attached and can't be attached to another node", "attachedTo", otherNodesStr, "volume", volumeToAttach)
		return
	}

	// There are pods that require the volume and run on another node. Typically
	// it's user error, e.g. a ReplicaSet uses a PVC and has >1 replicas. Let
	// the user know what pods are blocking the volume.
	for _, scheduledPod := range volumeToAttach.ScheduledPods {
		// Each scheduledPod must get a custom message. They can run in
		// different namespaces and user of a namespace should not see names of
		// pods in other namespaces.
		localPodNames := []string{} // Names of pods in scheduledPods's namespace
		otherPods := 0              // Count of pods in other namespaces
		for _, pod := range pods {
			if pod.Namespace == scheduledPod.Namespace {
				localPodNames = append(localPodNames, pod.Name)
			} else {
				otherPods++
			}
		}

		var msg string
		if len(localPodNames) > 0 {
			msg = fmt.Sprintf("Volume is already used by pod(s) %s", strings.Join(localPodNames, ", "))
			if otherPods > 0 {
				msg = fmt.Sprintf("%s and %d pod(s) in different namespaces", msg, otherPods)
			}
		} else {
			// No local pods, there are pods only in different namespaces.
			msg = fmt.Sprintf("Volume is already used by %d pod(s) in different namespaces", otherPods)
		}
		simpleMsg, _ := volumeToAttach.GenerateMsg("Multi-Attach error", msg)
		rc.recorder.Eventf(scheduledPod, v1.EventTypeWarning, kevents.FailedAttachVolume, simpleMsg)
	}

	// Log all pods for system admin
	klog.InfoS("Multi-Attach error: volume is already used by pods", "pods", klog.KObjs(pods), "attachedTo", otherNodesStr, "volume", volumeToAttach)
}
