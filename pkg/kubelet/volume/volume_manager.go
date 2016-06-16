/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package volumemanager

import (
	"fmt"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/volume/cache"
	"k8s.io/kubernetes/pkg/kubelet/volume/populator"
	"k8s.io/kubernetes/pkg/kubelet/volume/reconciler"
	"k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	"k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

const (
	// reconcilerLoopSleepPeriod is the amount of time the reconciler loop waits
	// between successive executions
	reconcilerLoopSleepPeriod time.Duration = 100 * time.Millisecond

	// desiredStateOfWorldPopulatorLoopSleepPeriod is the amount of time the
	// DesiredStateOfWorldPopulator loop waits between successive executions
	desiredStateOfWorldPopulatorLoopSleepPeriod time.Duration = 100 * time.Millisecond

	// podAttachAndMountTimeout is the maximum amount of time the
	// WaitForAttachAndMount call will wait for all volumes in the specified pod
	// to be attached and mounted. Even though cloud operations can take several
	// minutes to complete, we set the timeout to 2 minutes because kubelet
	// will retry in the next sync iteration. This frees the associated
	// goroutine of the pod to process newer updates if needed (e.g., a delete
	// request to the pod).
	podAttachAndMountTimeout time.Duration = 2 * time.Minute

	// podAttachAndMountRetryInterval is the amount of time the GetVolumesForPod
	// call waits before retrying
	podAttachAndMountRetryInterval time.Duration = 300 * time.Millisecond

	// waitForAttachTimeout is the maximum amount of time a
	// operationexecutor.Mount call will wait for a volume to be attached.
	// Set to 10 minutes because we've seen attach operations take several
	// minutes to complete for some volume plugins in some cases. While this
	// operation is waiting it only blocks other operations on the same device,
	// other devices are not affected.
	waitForAttachTimeout time.Duration = 10 * time.Minute
)

// VolumeManager runs a set of asynchronous loops that figure out which volumes
// need to be attached/mounted/unmounted/detached based on the pods scheduled on
// this node and makes it so.
type VolumeManager interface {
	// Starts the volume manager and all the asynchronous loops that it controls
	Run(stopCh <-chan struct{})

	// WaitForAttachAndMount processes the volumes referenced in the specified
	// pod and blocks until they are all attached and mounted (reflected in
	// actual state of the world).
	// An error is returned if all volumes are not attached and mounted within
	// the duration defined in podAttachAndMountTimeout.
	WaitForAttachAndMount(pod *api.Pod) error

	// GetMountedVolumesForPod returns a VolumeMap containing the volumes
	// referenced by the specified pod that are successfully attached and
	// mounted. The key in the map is the OuterVolumeSpecName (i.e.
	// pod.Spec.Volumes[x].Name). It returns an empty VolumeMap if pod has no
	// volumes.
	GetMountedVolumesForPod(podName types.UniquePodName) container.VolumeMap

	// GetVolumesForPodAndApplySupplementalGroups, like GetVolumesForPod returns
	// a VolumeMap containing the volumes referenced by the specified pod that
	// are successfully attached and mounted. The key in the map is the
	// OuterVolumeSpecName (i.e. pod.Spec.Volumes[x].Name).
	// It returns an empty VolumeMap if pod has no volumes.
	// In addition for every volume that specifies a VolumeGidValue, it appends
	// the SecurityContext.SupplementalGroups for the specified pod.
	// XXX: https://github.com/kubernetes/kubernetes/issues/27197 mutating the
	// pod object is bad, and should be avoided.
	GetVolumesForPodAndAppendSupplementalGroups(pod *api.Pod) container.VolumeMap

	// Returns a list of all volumes that are currently attached according to
	// the actual state of the world cache and implement the volume.Attacher
	// interface.
	GetVolumesInUse() []api.UniqueVolumeName
}

// NewVolumeManager returns a new concrete instance implementing the
// VolumeManager interface.
//
// kubeClient - kubeClient is the kube API client used by DesiredStateOfWorldPopulator
//   to communicate with the API server to fetch PV and PVC objects
// volumePluginMgr - the volume plugin manager used to access volume plugins.
//   Must be pre-initialized.
func NewVolumeManager(
	controllerAttachDetachEnabled bool,
	hostName string,
	podManager pod.Manager,
	kubeClient internalclientset.Interface,
	volumePluginMgr *volume.VolumePluginMgr) (VolumeManager, error) {
	vm := &volumeManager{
		kubeClient:          kubeClient,
		volumePluginMgr:     volumePluginMgr,
		desiredStateOfWorld: cache.NewDesiredStateOfWorld(volumePluginMgr),
		actualStateOfWorld:  cache.NewActualStateOfWorld(hostName, volumePluginMgr),
		operationExecutor:   operationexecutor.NewOperationExecutor(volumePluginMgr),
	}

	vm.reconciler = reconciler.NewReconciler(
		controllerAttachDetachEnabled,
		reconcilerLoopSleepPeriod,
		waitForAttachTimeout,
		hostName,
		vm.desiredStateOfWorld,
		vm.actualStateOfWorld,
		vm.operationExecutor)
	vm.desiredStateOfWorldPopulator = populator.NewDesiredStateOfWorldPopulator(
		kubeClient,
		desiredStateOfWorldPopulatorLoopSleepPeriod,
		podManager,
		vm.desiredStateOfWorld)

	return vm, nil
}

// volumeManager implements the VolumeManager interface
type volumeManager struct {
	// kubeClient is the kube API client used by DesiredStateOfWorldPopulator to
	// communicate with the API server to fetch PV and PVC objects
	kubeClient internalclientset.Interface

	// volumePluginMgr is the volume plugin manager used to access volume
	// plugins. It must be pre-initialized.
	volumePluginMgr *volume.VolumePluginMgr

	// desiredStateOfWorld is a data structure containing the desired state of
	// the world according to the volume manager: i.e. what volumes should be
	// attached and which pods are referencing the volumes).
	// The data structure is populated by the desired state of the world
	// populator using the kubelet pod manager.
	desiredStateOfWorld cache.DesiredStateOfWorld

	// actualStateOfWorld is a data structure containing the actual state of
	// the world according to the manager: i.e. which volumes are attached to
	// this node and what pods the volumes are mounted to.
	// The data structure is populated upon successful completion of attach,
	// detach, mount, and unmount actions triggered by the reconciler.
	actualStateOfWorld cache.ActualStateOfWorld

	// operationExecutor is used to start asynchronous attach, detach, mount,
	// and unmount operations.
	operationExecutor operationexecutor.OperationExecutor

	// reconciler runs an asynchronous periodic loop to reconcile the
	// desiredStateOfWorld with the actualStateOfWorld by triggering attach,
	// detach, mount, and unmount operations using the operationExecutor.
	reconciler reconciler.Reconciler

	// desiredStateOfWorldPopulator runs an asynchronous periodic loop to
	// populate the desiredStateOfWorld using the kubelet PodManager.
	desiredStateOfWorldPopulator populator.DesiredStateOfWorldPopulator
}

func (vm *volumeManager) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	glog.Infof("Starting Kubelet Volume Manager")

	go vm.reconciler.Run(stopCh)
	go vm.desiredStateOfWorldPopulator.Run(stopCh)

	<-stopCh
	glog.Infof("Shutting down Kubelet Volume Manager")
}

func (vm *volumeManager) GetMountedVolumesForPod(
	podName types.UniquePodName) container.VolumeMap {
	return vm.getVolumesForPodHelper(podName, nil /* pod */)
}

func (vm *volumeManager) GetVolumesForPodAndAppendSupplementalGroups(
	pod *api.Pod) container.VolumeMap {
	return vm.getVolumesForPodHelper("" /* podName */, pod)
}

func (vm *volumeManager) GetVolumesInUse() []api.UniqueVolumeName {
	attachedVolumes := vm.actualStateOfWorld.GetAttachedVolumes()
	volumesInUse :=
		make([]api.UniqueVolumeName, 0 /* len */, len(attachedVolumes) /* cap */)
	for _, attachedVolume := range attachedVolumes {
		if attachedVolume.PluginIsAttachable {
			volumesInUse = append(volumesInUse, attachedVolume.VolumeName)
		}
	}

	return volumesInUse
}

// getVolumesForPodHelper is a helper method implements the common logic for
// the GetVolumesForPod methods.
// XXX: https://github.com/kubernetes/kubernetes/issues/27197 mutating the pod
// object is bad, and should be avoided.
func (vm *volumeManager) getVolumesForPodHelper(
	podName types.UniquePodName, pod *api.Pod) container.VolumeMap {
	if pod != nil {
		podName = volumehelper.GetUniquePodName(pod)
	}
	podVolumes := make(container.VolumeMap)
	for _, mountedVolume := range vm.actualStateOfWorld.GetMountedVolumesForPod(podName) {
		podVolumes[mountedVolume.OuterVolumeSpecName] =
			container.VolumeInfo{Mounter: mountedVolume.Mounter}
		if pod != nil {
			err := applyPersistentVolumeAnnotations(
				mountedVolume.VolumeGidValue, pod)
			if err != nil {
				glog.Errorf("applyPersistentVolumeAnnotations failed for pod %q volume %q with: %v",
					podName,
					mountedVolume.VolumeName,
					err)
			}
		}
	}
	return podVolumes
}

func (vm *volumeManager) WaitForAttachAndMount(pod *api.Pod) error {
	expectedVolumes := getExpectedVolumes(pod)
	if len(expectedVolumes) == 0 {
		// No volumes to verify
		return nil
	}

	glog.V(3).Infof("Waiting for volumes to attach and mount for pod %q", format.Pod(pod))
	uniquePodName := volumehelper.GetUniquePodName(pod)

	// Some pods expect to have Setup called over and over again to update.
	// Remount plugins for which this is true. (Atomically updating volumes,
	// like Downward API, depend on this to update the contents of the volume).
	vm.desiredStateOfWorldPopulator.ReprocessPod(uniquePodName)
	vm.actualStateOfWorld.MarkRemountRequired(uniquePodName)

	err := wait.Poll(
		podAttachAndMountRetryInterval,
		podAttachAndMountTimeout,
		vm.verifyVolumesMountedFunc(uniquePodName, expectedVolumes))

	if err != nil {
		// Timeout expired
		ummountedVolumes :=
			vm.getUnmountedVolumes(uniquePodName, expectedVolumes)
		if len(ummountedVolumes) == 0 {
			return nil
		}

		return fmt.Errorf(
			"timeout expired waiting for volumes to attach/mount for pod %q/%q. list of unattached/unmounted volumes=%v",
			pod.Name,
			pod.Namespace,
			ummountedVolumes)
	}

	glog.V(3).Infof("All volumes are attached and mounted for pod %q", format.Pod(pod))
	return nil
}

// verifyVolumesMountedFunc returns a method that returns true when all expected
// volumes are mounted.
func (vm *volumeManager) verifyVolumesMountedFunc(
	podName types.UniquePodName, expectedVolumes []string) wait.ConditionFunc {
	return func() (done bool, err error) {
		return len(vm.getUnmountedVolumes(podName, expectedVolumes)) == 0, nil
	}
}

// getUnmountedVolumes fetches the current list of mounted volumes from
// the actual state of the world, and uses it to process the list of
// expectedVolumes. It returns a list of unmounted volumes.
func (vm *volumeManager) getUnmountedVolumes(
	podName types.UniquePodName, expectedVolumes []string) []string {
	mountedVolumes := sets.NewString()
	for _, mountedVolume := range vm.actualStateOfWorld.GetMountedVolumesForPod(podName) {
		mountedVolumes.Insert(mountedVolume.OuterVolumeSpecName)
	}
	return filterUnmountedVolumes(mountedVolumes, expectedVolumes)
}

// filterUnmountedVolumes adds each element of expectedVolumes that is not in
// mountedVolumes to a list of unmountedVolumes and returns it.
func filterUnmountedVolumes(
	mountedVolumes sets.String, expectedVolumes []string) []string {
	unmountedVolumes := []string{}
	for _, expectedVolume := range expectedVolumes {
		if !mountedVolumes.Has(expectedVolume) {
			unmountedVolumes = append(unmountedVolumes, expectedVolume)
		}
	}
	return unmountedVolumes
}

// getExpectedVolumes returns a list of volumes that must be mounted in order to
// consider the volume setup step for this pod satisfied.
func getExpectedVolumes(pod *api.Pod) []string {
	expectedVolumes := []string{}
	if pod == nil {
		return expectedVolumes
	}

	for _, podVolume := range pod.Spec.Volumes {
		expectedVolumes = append(expectedVolumes, podVolume.Name)
	}

	return expectedVolumes
}

// applyPersistentVolumeAnnotations appends a pod
// SecurityContext.SupplementalGroups if a GID annotation is provided.
// XXX: https://github.com/kubernetes/kubernetes/issues/27197 mutating the pod
// object is bad, and should be avoided.
func applyPersistentVolumeAnnotations(
	volumeGidValue string, pod *api.Pod) error {
	if volumeGidValue != "" {
		gid, err := strconv.ParseInt(volumeGidValue, 10, 64)
		if err != nil {
			return fmt.Errorf(
				"Invalid value for %s %v",
				volumehelper.VolumeGidAnnotationKey,
				err)
		}

		if pod.Spec.SecurityContext == nil {
			pod.Spec.SecurityContext = &api.PodSecurityContext{}
		}
		for _, existingGid := range pod.Spec.SecurityContext.SupplementalGroups {
			if gid == existingGid {
				return nil
			}
		}
		pod.Spec.SecurityContext.SupplementalGroups =
			append(pod.Spec.SecurityContext.SupplementalGroups, gid)
	}

	return nil
}
