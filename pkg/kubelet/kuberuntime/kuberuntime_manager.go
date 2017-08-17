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

package kuberuntime

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/credentialprovider"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/cache"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	utilversion "k8s.io/kubernetes/pkg/util/version"
)

const (
	// The api version of kubelet runtime api
	kubeRuntimeAPIVersion = "0.1.0"
	// The root directory for pod logs
	podLogsRootDirectory = "/var/log/pods"
	// A minimal shutdown window for avoiding unnecessary SIGKILLs
	minimumGracePeriodInSeconds = 2

	// The expiration time of version cache.
	versionCacheTTL = 60 * time.Second
)

var (
	// ErrVersionNotSupported is returned when the api version of runtime interface is not supported
	ErrVersionNotSupported = errors.New("Runtime api version is not supported")
)

// A subset of the pod.Manager interface extracted for garbage collection purposes.
type podGetter interface {
	GetPodByUID(kubetypes.UID) (*v1.Pod, bool)
}

type kubeGenericRuntimeManager struct {
	runtimeName         string
	recorder            record.EventRecorder
	osInterface         kubecontainer.OSInterface
	containerRefManager *kubecontainer.RefManager

	// machineInfo contains the machine information.
	machineInfo *cadvisorapi.MachineInfo

	// Container GC manager
	containerGC *containerGC

	// Keyring for pulling images
	keyring credentialprovider.DockerKeyring

	// Runner of lifecycle events.
	runner kubecontainer.HandlerRunner

	// RuntimeHelper that wraps kubelet to generate runtime container options.
	runtimeHelper kubecontainer.RuntimeHelper

	// Health check results.
	livenessManager proberesults.Manager

	// If true, enforce container cpu limits with CFS quota support
	cpuCFSQuota bool

	// wrapped image puller.
	imagePuller images.ImageManager

	// gRPC service clients
	runtimeService internalapi.RuntimeService
	imageService   internalapi.ImageManagerService

	// The version cache of runtime daemon.
	versionCache *cache.ObjectCache

	// The directory path for seccomp profiles.
	seccompProfileRoot string
}

type KubeGenericRuntime interface {
	kubecontainer.Runtime
	kubecontainer.IndirectStreamingRuntime
	kubecontainer.ContainerCommandRunner
}

// NewKubeGenericRuntimeManager creates a new kubeGenericRuntimeManager
func NewKubeGenericRuntimeManager(
	recorder record.EventRecorder,
	livenessManager proberesults.Manager,
	seccompProfileRoot string,
	containerRefManager *kubecontainer.RefManager,
	machineInfo *cadvisorapi.MachineInfo,
	podGetter podGetter,
	osInterface kubecontainer.OSInterface,
	runtimeHelper kubecontainer.RuntimeHelper,
	httpClient types.HttpGetter,
	imageBackOff *flowcontrol.Backoff,
	serializeImagePulls bool,
	imagePullQPS float32,
	imagePullBurst int,
	cpuCFSQuota bool,
	runtimeService internalapi.RuntimeService,
	imageService internalapi.ImageManagerService,
) (KubeGenericRuntime, error) {
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		recorder:            recorder,
		cpuCFSQuota:         cpuCFSQuota,
		seccompProfileRoot:  seccompProfileRoot,
		livenessManager:     livenessManager,
		containerRefManager: containerRefManager,
		machineInfo:         machineInfo,
		osInterface:         osInterface,
		runtimeHelper:       runtimeHelper,
		runtimeService:      newInstrumentedRuntimeService(runtimeService),
		imageService:        newInstrumentedImageManagerService(imageService),
		keyring:             credentialprovider.NewDockerKeyring(),
	}

	typedVersion, err := kubeRuntimeManager.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Get runtime version failed: %v", err)
		return nil, err
	}

	// Only matching kubeRuntimeAPIVersion is supported now
	// TODO: Runtime API machinery is under discussion at https://github.com/kubernetes/kubernetes/issues/28642
	if typedVersion.Version != kubeRuntimeAPIVersion {
		glog.Errorf("Runtime api version %s is not supported, only %s is supported now",
			typedVersion.Version,
			kubeRuntimeAPIVersion)
		return nil, ErrVersionNotSupported
	}

	kubeRuntimeManager.runtimeName = typedVersion.RuntimeName
	glog.Infof("Container runtime %s initialized, version: %s, apiVersion: %s",
		typedVersion.RuntimeName,
		typedVersion.RuntimeVersion,
		typedVersion.RuntimeApiVersion)

	// If the container logs directory does not exist, create it.
	// TODO: create podLogsRootDirectory at kubelet.go when kubelet is refactored to
	// new runtime interface
	if _, err := osInterface.Stat(podLogsRootDirectory); os.IsNotExist(err) {
		if err := osInterface.MkdirAll(podLogsRootDirectory, 0755); err != nil {
			glog.Errorf("Failed to create directory %q: %v", podLogsRootDirectory, err)
		}
	}

	kubeRuntimeManager.imagePuller = images.NewImageManager(
		kubecontainer.FilterEventRecorder(recorder),
		kubeRuntimeManager,
		imageBackOff,
		serializeImagePulls,
		imagePullQPS,
		imagePullBurst)
	kubeRuntimeManager.runner = lifecycle.NewHandlerRunner(httpClient, kubeRuntimeManager, kubeRuntimeManager)
	kubeRuntimeManager.containerGC = NewContainerGC(runtimeService, podGetter, kubeRuntimeManager)

	kubeRuntimeManager.versionCache = cache.NewObjectCache(
		func() (interface{}, error) {
			return kubeRuntimeManager.getTypedVersion()
		},
		versionCacheTTL,
	)

	return kubeRuntimeManager, nil
}

// Type returns the type of the container runtime.
func (m *kubeGenericRuntimeManager) Type() string {
	return m.runtimeName
}

func newRuntimeVersion(version string) (*utilversion.Version, error) {
	return utilversion.ParseSemantic(version)
}

func (m *kubeGenericRuntimeManager) getTypedVersion() (*runtimeapi.VersionResponse, error) {
	typedVersion, err := m.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Get remote runtime typed version failed: %v", err)
		return nil, err
	}
	return typedVersion, nil
}

// Version returns the version information of the container runtime.
func (m *kubeGenericRuntimeManager) Version() (kubecontainer.Version, error) {
	typedVersion, err := m.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Get remote runtime version failed: %v", err)
		return nil, err
	}

	return newRuntimeVersion(typedVersion.RuntimeVersion)
}

// APIVersion returns the cached API version information of the container
// runtime. Implementation is expected to update this cache periodically.
// This may be different from the runtime engine's version.
func (m *kubeGenericRuntimeManager) APIVersion() (kubecontainer.Version, error) {
	versionObject, err := m.versionCache.Get(m.machineInfo.MachineID)
	if err != nil {
		return nil, err
	}
	typedVersion := versionObject.(*runtimeapi.VersionResponse)

	return newRuntimeVersion(typedVersion.RuntimeApiVersion)
}

// Status returns the status of the runtime. An error is returned if the Status
// function itself fails, nil otherwise.
func (m *kubeGenericRuntimeManager) Status() (*kubecontainer.RuntimeStatus, error) {
	status, err := m.runtimeService.Status()
	if err != nil {
		return nil, err
	}
	return toKubeRuntimeStatus(status), nil
}

// GetPods returns a list of containers grouped by pods. The boolean parameter
// specifies whether the runtime returns all containers including those already
// exited and dead containers (used for garbage collection).
func (m *kubeGenericRuntimeManager) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	pods := make(map[kubetypes.UID]*kubecontainer.Pod)
	sandboxes, err := m.getKubeletSandboxes(all)
	if err != nil {
		return nil, err
	}
	for i := range sandboxes {
		s := sandboxes[i]
		if s.Metadata == nil {
			glog.V(4).Infof("Sandbox does not have metadata: %+v", s)
			continue
		}
		podUID := kubetypes.UID(s.Metadata.Uid)
		if _, ok := pods[podUID]; !ok {
			pods[podUID] = &kubecontainer.Pod{
				ID:        podUID,
				Name:      s.Metadata.Name,
				Namespace: s.Metadata.Namespace,
			}
		}
		p := pods[podUID]
		converted, err := m.sandboxToKubeContainer(s)
		if err != nil {
			glog.V(4).Infof("Convert %q sandbox %v of pod %q failed: %v", m.runtimeName, s, podUID, err)
			continue
		}
		p.Sandboxes = append(p.Sandboxes, converted)
	}

	containers, err := m.getKubeletContainers(all)
	if err != nil {
		return nil, err
	}
	for i := range containers {
		c := containers[i]
		if c.Metadata == nil {
			glog.V(4).Infof("Container does not have metadata: %+v", c)
			continue
		}

		labelledInfo := getContainerInfoFromLabels(c.Labels)
		pod, found := pods[labelledInfo.PodUID]
		if !found {
			pod = &kubecontainer.Pod{
				ID:        labelledInfo.PodUID,
				Name:      labelledInfo.PodName,
				Namespace: labelledInfo.PodNamespace,
			}
			pods[labelledInfo.PodUID] = pod
		}

		converted, err := m.toKubeContainer(c)
		if err != nil {
			glog.V(4).Infof("Convert %s container %v of pod %q failed: %v", m.runtimeName, c, labelledInfo.PodUID, err)
			continue
		}

		pod.Containers = append(pod.Containers, converted)
	}

	// Convert map to list.
	var result []*kubecontainer.Pod
	for _, pod := range pods {
		result = append(result, pod)
	}

	return result, nil
}

// containerToKillInfo contains necessary information to kill a container.
type containerToKillInfo struct {
	// The spec of the container.
	container *v1.Container
	// The name of the container.
	name string
	// The message indicates why the container will be killed.
	message string
}

// podActions keeps information what to do for a pod.
type podActions struct {
	// Stop all running (regular and init) containers and the sandbox for the pod.
	KillPod bool
	// Whether need to create a new sandbox. If needed to kill pod and create a
	// a new pod sandbox, all init containers need to be purged (i.e., removed).
	CreateSandbox bool
	// The id of existing sandbox. It is used for starting containers in ContainersToStart.
	SandboxID string
	// The attempt number of creating sandboxes for the pod.
	Attempt uint32

	// The next init container to start.
	NextInitContainerToStart *v1.Container
	// ContainersToStart keeps a list of indexes for the containers to start,
	// where the index is the index of the specific container in the pod spec (
	// pod.Spec.Containers.
	ContainersToStart []int
	// ContainersToKill keeps a map of containers that need to be killed, note that
	// the key is the container ID of the container, while
	// the value contains necessary information to kill a container.
	ContainersToKill map[kubecontainer.ContainerID]containerToKillInfo
}

// podSandboxChanged checks whether the spec of the pod is changed and returns
// (changed, new attempt, original sandboxID if exist).
func (m *kubeGenericRuntimeManager) podSandboxChanged(pod *v1.Pod, podStatus *kubecontainer.PodStatus) (bool, uint32, string) {
	if len(podStatus.SandboxStatuses) == 0 {
		glog.V(2).Infof("No sandbox for pod %q can be found. Need to start a new one", format.Pod(pod))
		return true, 0, ""
	}

	readySandboxCount := 0
	for _, s := range podStatus.SandboxStatuses {
		if s.State == runtimeapi.PodSandboxState_SANDBOX_READY {
			readySandboxCount++
		}
	}

	// Needs to create a new sandbox when readySandboxCount > 1 or the ready sandbox is not the latest one.
	sandboxStatus := podStatus.SandboxStatuses[0]
	if readySandboxCount > 1 {
		glog.V(2).Infof("More than 1 sandboxes for pod %q are ready. Need to reconcile them", format.Pod(pod))
		return true, sandboxStatus.Metadata.Attempt + 1, sandboxStatus.Id
	}
	if sandboxStatus.State != runtimeapi.PodSandboxState_SANDBOX_READY {
		glog.V(2).Infof("No ready sandbox for pod %q can be found. Need to start a new one", format.Pod(pod))
		return true, sandboxStatus.Metadata.Attempt + 1, sandboxStatus.Id
	}

	// Needs to create a new sandbox when network namespace changed.
	if sandboxStatus.Linux != nil && sandboxStatus.Linux.Namespaces != nil && sandboxStatus.Linux.Namespaces.Options != nil &&
		sandboxStatus.Linux.Namespaces.Options.HostNetwork != kubecontainer.IsHostNetworkPod(pod) {
		glog.V(2).Infof("Sandbox for pod %q has changed. Need to start a new one", format.Pod(pod))
		return true, sandboxStatus.Metadata.Attempt + 1, ""
	}

	return false, sandboxStatus.Metadata.Attempt, sandboxStatus.Id
}

func containerChanged(container *v1.Container, containerStatus *kubecontainer.ContainerStatus) (uint64, uint64, bool) {
	expectedHash := kubecontainer.HashContainer(container)
	return expectedHash, containerStatus.Hash, containerStatus.Hash != expectedHash
}

func shouldRestartOnFailure(pod *v1.Pod) bool {
	return pod.Spec.RestartPolicy != v1.RestartPolicyNever
}

func containerSucceeded(c *v1.Container, podStatus *kubecontainer.PodStatus) bool {
	cStatus := podStatus.FindContainerStatusByName(c.Name)
	if cStatus == nil || cStatus.State == kubecontainer.ContainerStateRunning {
		return false
	}
	return cStatus.ExitCode == 0
}

// computePodActions checks whether the pod spec has changed and returns the changes if true.
func (m *kubeGenericRuntimeManager) computePodActions(pod *v1.Pod, podStatus *kubecontainer.PodStatus) podActions {
	glog.V(5).Infof("Syncing Pod %q: %+v", format.Pod(pod), pod)

	createPodSandbox, attempt, sandboxID := m.podSandboxChanged(pod, podStatus)
	changes := podActions{
		KillPod:           createPodSandbox,
		CreateSandbox:     createPodSandbox,
		SandboxID:         sandboxID,
		Attempt:           attempt,
		ContainersToStart: []int{},
		ContainersToKill:  make(map[kubecontainer.ContainerID]containerToKillInfo),
	}

	// If we need to (re-)create the pod sandbox, everything will need to be
	// killed and recreated, and init containers should be purged.
	if createPodSandbox {
		if !shouldRestartOnFailure(pod) && attempt != 0 {
			// Should not restart the pod, just return.
			return changes
		}
		if len(pod.Spec.InitContainers) != 0 {
			// Pod has init containers, return the first one.
			changes.NextInitContainerToStart = &pod.Spec.InitContainers[0]
			return changes
		}
		// Start all containers by default but exclude the ones that succeeded if
		// RestartPolicy is OnFailure.
		for idx, c := range pod.Spec.Containers {
			if containerSucceeded(&c, podStatus) && pod.Spec.RestartPolicy == v1.RestartPolicyOnFailure {
				continue
			}
			changes.ContainersToStart = append(changes.ContainersToStart, idx)
		}
		return changes
	}

	// Check initialization progress.
	initLastStatus, next, done := findNextInitContainerToRun(pod, podStatus)
	if !done {
		if next != nil {
			initFailed := initLastStatus != nil && isContainerFailed(initLastStatus)
			if initFailed && !shouldRestartOnFailure(pod) {
				changes.KillPod = true
			} else {
				changes.NextInitContainerToStart = next
			}
		}
		// Initialization failed or still in progress. Skip inspecting non-init
		// containers.
		return changes
	}

	// Number of running containers to keep.
	keepCount := 0
	// check the status of containers.
	for idx, container := range pod.Spec.Containers {
		containerStatus := podStatus.FindContainerStatusByName(container.Name)
		// If container does not exist, or is not running, check whether we
		// need to restart it.
		if containerStatus == nil || containerStatus.State != kubecontainer.ContainerStateRunning {
			if kubecontainer.ShouldContainerBeRestarted(&container, pod, podStatus) {
				message := fmt.Sprintf("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				glog.Info(message)
				changes.ContainersToStart = append(changes.ContainersToStart, idx)
			}
			continue
		}
		// The container is running, but kill the container if any of the following condition is met.
		reason := ""
		restart := shouldRestartOnFailure(pod)
		if expectedHash, actualHash, changed := containerChanged(&container, containerStatus); changed {
			reason = fmt.Sprintf("Container spec hash changed (%d vs %d).", actualHash, expectedHash)
			// Restart regardless of the restart policy because the container
			// spec changed.
			restart = true
		} else if liveness, found := m.livenessManager.Get(containerStatus.ID); found && liveness == proberesults.Failure {
			// If the container failed the liveness probe, we should kill it.
			reason = "Container failed liveness probe."
		} else {
			// Keep the container.
			keepCount += 1
			continue
		}

		// We need to kill the container, but if we also want to restart the
		// container afterwards, make the intent clear in the message. Also do
		// not kill the entire pod since we expect container to be running eventually.
		message := reason
		if restart {
			message = fmt.Sprintf("%s. Container will be killed and recreated.", message)
			changes.ContainersToStart = append(changes.ContainersToStart, idx)
		}

		changes.ContainersToKill[containerStatus.ID] = containerToKillInfo{
			name:      containerStatus.Name,
			container: &pod.Spec.Containers[idx],
			message:   message,
		}
		glog.V(2).Infof("Container %q (%q) of pod %s: %s", container.Name, containerStatus.ID, format.Pod(pod), message)
	}

	if keepCount == 0 && len(changes.ContainersToStart) == 0 {
		changes.KillPod = true
	}

	return changes
}

// SyncPod syncs the running pod into the desired pod by executing following steps:
//
//  1. Compute sandbox and container changes.
//  2. Kill pod sandbox if necessary.
//  3. Kill any containers that should not be running.
//  4. Create sandbox if necessary.
//  5. Create init containers.
//  6. Create normal containers.
func (m *kubeGenericRuntimeManager) SyncPod(pod *v1.Pod, _ v1.PodStatus, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret, backOff *flowcontrol.Backoff) (result kubecontainer.PodSyncResult) {
	// Step 1: Compute sandbox and container changes.
	podContainerChanges := m.computePodActions(pod, podStatus)
	glog.V(3).Infof("computePodActions got %+v for pod %q", podContainerChanges, format.Pod(pod))
	if podContainerChanges.CreateSandbox {
		ref, err := ref.GetReference(api.Scheme, pod)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %q: '%v'", format.Pod(pod), err)
		}
		if podContainerChanges.SandboxID != "" {
			m.recorder.Eventf(ref, v1.EventTypeNormal, events.SandboxChanged, "Pod sandbox changed, it will be killed and re-created.")
		} else {
			glog.V(4).Infof("SyncPod received new pod %q, will create a sandbox for it", format.Pod(pod))
		}
	}

	// Step 2: Kill the pod if the sandbox has changed.
	if podContainerChanges.KillPod {
		if !podContainerChanges.CreateSandbox {
			glog.V(4).Infof("Stopping PodSandbox for %q because all other containers are dead.", format.Pod(pod))
		} else {
			glog.V(4).Infof("Stopping PodSandbox for %q, will start new one", format.Pod(pod))
		}

		killResult := m.killPodWithSyncResult(pod, kubecontainer.ConvertPodStatusToRunningPod(m.runtimeName, podStatus), nil)
		result.AddPodSyncResult(killResult)
		if killResult.Error() != nil {
			glog.Errorf("killPodWithSyncResult failed: %v", killResult.Error())
			return
		}

		if podContainerChanges.CreateSandbox {
			m.purgeInitContainers(pod, podStatus)
		}
	} else {
		// Step 3: kill any running containers in this pod which are not to keep.
		for containerID, containerInfo := range podContainerChanges.ContainersToKill {
			glog.V(3).Infof("Killing unwanted container %q(id=%q) for pod %q", containerInfo.name, containerID, format.Pod(pod))
			killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, containerInfo.name)
			result.AddSyncResult(killContainerResult)
			if err := m.killContainer(pod, containerID, containerInfo.name, containerInfo.message, nil); err != nil {
				killContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
				glog.Errorf("killContainer %q(id=%q) for pod %q failed: %v", containerInfo.name, containerID, format.Pod(pod), err)
				return
			}
		}
	}

	// Keep terminated init containers fairly aggressively controlled
	// This is an optmization because container removals are typically handled
	// by container garbage collector.
	m.pruneInitContainersBeforeStart(pod, podStatus)

	// We pass the value of the podIP down to generatePodSandboxConfig and
	// generateContainerConfig, which in turn passes it to various other
	// functions, in order to facilitate functionality that requires this
	// value (hosts file and downward API) and avoid races determining
	// the pod IP in cases where a container requires restart but the
	// podIP isn't in the status manager yet.
	//
	// We default to the IP in the passed-in pod status, and overwrite it if the
	// sandbox needs to be (re)started.
	podIP := ""
	if podStatus != nil {
		podIP = podStatus.IP
	}

	// Step 4: Create a sandbox for the pod if necessary.
	podSandboxID := podContainerChanges.SandboxID
	if podContainerChanges.CreateSandbox {
		var msg string
		var err error

		glog.V(4).Infof("Creating sandbox for pod %q", format.Pod(pod))
		createSandboxResult := kubecontainer.NewSyncResult(kubecontainer.CreatePodSandbox, format.Pod(pod))
		result.AddSyncResult(createSandboxResult)
		podSandboxID, msg, err = m.createPodSandbox(pod, podContainerChanges.Attempt)
		if err != nil {
			createSandboxResult.Fail(kubecontainer.ErrCreatePodSandbox, msg)
			glog.Errorf("createPodSandbox for pod %q failed: %v", format.Pod(pod), err)
			return
		}
		glog.V(4).Infof("Created PodSandbox %q for pod %q", podSandboxID, format.Pod(pod))

		podSandboxStatus, err := m.runtimeService.PodSandboxStatus(podSandboxID)
		if err != nil {
			glog.Errorf("Failed to get pod sandbox status: %v; Skipping pod %q", err, format.Pod(pod))
			result.Fail(err)
			return
		}

		// If we ever allow updating a pod from non-host-network to
		// host-network, we may use a stale IP.
		if !kubecontainer.IsHostNetworkPod(pod) {
			// Overwrite the podIP passed in the pod status, since we just started the pod sandbox.
			podIP = m.determinePodSandboxIP(pod.Namespace, pod.Name, podSandboxStatus)
			glog.V(4).Infof("Determined the ip %q for pod %q after sandbox changed", podIP, format.Pod(pod))
		}
	}

	// Get podSandboxConfig for containers to start.
	configPodSandboxResult := kubecontainer.NewSyncResult(kubecontainer.ConfigPodSandbox, podSandboxID)
	result.AddSyncResult(configPodSandboxResult)
	podSandboxConfig, err := m.generatePodSandboxConfig(pod, podContainerChanges.Attempt)
	if err != nil {
		message := fmt.Sprintf("GeneratePodSandboxConfig for pod %q failed: %v", format.Pod(pod), err)
		glog.Error(message)
		configPodSandboxResult.Fail(kubecontainer.ErrConfigPodSandbox, message)
		return
	}

	// Step 5: start the init container.
	if container := podContainerChanges.NextInitContainerToStart; container != nil {
		// Start the next init container.
		startContainerResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, container.Name)
		result.AddSyncResult(startContainerResult)
		isInBackOff, msg, err := m.doBackOff(pod, container, podStatus, backOff)
		if isInBackOff {
			startContainerResult.Fail(err, msg)
			glog.V(4).Infof("Backing Off restarting init container %+v in pod %v", container, format.Pod(pod))
			return
		}

		glog.V(4).Infof("Creating init container %+v in pod %v", container, format.Pod(pod))
		if msg, err := m.startContainer(podSandboxID, podSandboxConfig, container, pod, podStatus, pullSecrets, podIP); err != nil {
			startContainerResult.Fail(err, msg)
			utilruntime.HandleError(fmt.Errorf("init container start failed: %v: %s", err, msg))
			return
		}

		// Successfully started the container; clear the entry in the failure
		glog.V(4).Infof("Completed init container %q for pod %q", container.Name, format.Pod(pod))
	}

	// Step 6: start containers in podContainerChanges.ContainersToStart.
	for _, idx := range podContainerChanges.ContainersToStart {
		container := &pod.Spec.Containers[idx]
		startContainerResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, container.Name)
		result.AddSyncResult(startContainerResult)

		isInBackOff, msg, err := m.doBackOff(pod, container, podStatus, backOff)
		if isInBackOff {
			startContainerResult.Fail(err, msg)
			glog.V(4).Infof("Backing Off restarting container %+v in pod %v", container, format.Pod(pod))
			continue
		}

		glog.V(4).Infof("Creating container %+v in pod %v", container, format.Pod(pod))
		if msg, err := m.startContainer(podSandboxID, podSandboxConfig, container, pod, podStatus, pullSecrets, podIP); err != nil {
			startContainerResult.Fail(err, msg)
			utilruntime.HandleError(fmt.Errorf("container start failed: %v: %s", err, msg))
			continue
		}
	}

	return
}

// If a container is still in backoff, the function will return a brief backoff error and
// a detailed error message.
func (m *kubeGenericRuntimeManager) doBackOff(pod *v1.Pod, container *v1.Container, podStatus *kubecontainer.PodStatus, backOff *flowcontrol.Backoff) (bool, string, error) {
	var cStatus *kubecontainer.ContainerStatus
	for _, c := range podStatus.ContainerStatuses {
		if c.Name == container.Name && c.State == kubecontainer.ContainerStateExited {
			cStatus = c
			break
		}
	}

	if cStatus == nil {
		return false, "", nil
	}

	glog.Infof("checking backoff for container %q in pod %q", container.Name, format.Pod(pod))
	// Use the finished time of the latest exited container as the start point to calculate whether to do back-off.
	ts := cStatus.FinishedAt
	// backOff requires a unique key to identify the container.
	key := getStableKey(pod, container)
	if backOff.IsInBackOffSince(key, ts) {
		if ref, err := kubecontainer.GenerateContainerRef(pod, container); err == nil {
			m.recorder.Eventf(ref, v1.EventTypeWarning, events.BackOffStartContainer, "Back-off restarting failed container")
		}
		err := fmt.Errorf("Back-off %s restarting failed container=%s pod=%s", backOff.Get(key), container.Name, format.Pod(pod))
		glog.Infof("%s", err.Error())
		return true, err.Error(), kubecontainer.ErrCrashLoopBackOff
	}

	backOff.Next(key, ts)
	return false, "", nil
}

// KillPod kills all the containers of a pod. Pod may be nil, running pod must not be.
// gracePeriodOverride if specified allows the caller to override the pod default grace period.
// only hard kill paths are allowed to specify a gracePeriodOverride in the kubelet in order to not corrupt user data.
// it is useful when doing SIGKILL for hard eviction scenarios, or max grace period during soft eviction scenarios.
func (m *kubeGenericRuntimeManager) KillPod(pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) error {
	err := m.killPodWithSyncResult(pod, runningPod, gracePeriodOverride)
	return err.Error()
}

// killPodWithSyncResult kills a runningPod and returns SyncResult.
// Note: The pod passed in could be *nil* when kubelet restarted.
func (m *kubeGenericRuntimeManager) killPodWithSyncResult(pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) (result kubecontainer.PodSyncResult) {
	killContainerResults := m.killContainersWithSyncResult(pod, runningPod, gracePeriodOverride)
	for _, containerResult := range killContainerResults {
		result.AddSyncResult(containerResult)
	}

	// stop sandbox, the sandbox will be removed in GarbageCollect
	killSandboxResult := kubecontainer.NewSyncResult(kubecontainer.KillPodSandbox, runningPod.ID)
	result.AddSyncResult(killSandboxResult)
	// Stop all sandboxes belongs to same pod
	for _, podSandbox := range runningPod.Sandboxes {
		if err := m.runtimeService.StopPodSandbox(podSandbox.ID.ID); err != nil {
			killSandboxResult.Fail(kubecontainer.ErrKillPodSandbox, err.Error())
			glog.Errorf("Failed to stop sandbox %q", podSandbox.ID)
		}
	}

	return
}

// isHostNetwork checks whether the pod is running in host-network mode.
func (m *kubeGenericRuntimeManager) isHostNetwork(podSandBoxID string, pod *v1.Pod) (bool, error) {
	if pod != nil {
		return kubecontainer.IsHostNetworkPod(pod), nil
	}

	podStatus, err := m.runtimeService.PodSandboxStatus(podSandBoxID)
	if err != nil {
		return false, err
	}

	if nsOpts := podStatus.GetLinux().GetNamespaces().GetOptions(); nsOpts != nil {
		return nsOpts.HostNetwork, nil
	}

	return false, nil
}

// GetPodStatus retrieves the status of the pod, including the
// information of all containers in the pod that are visible in Runtime.
func (m *kubeGenericRuntimeManager) GetPodStatus(uid kubetypes.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	// Now we retain restart count of container as a container label. Each time a container
	// restarts, pod will read the restart count from the registered dead container, increment
	// it to get the new restart count, and then add a label with the new restart count on
	// the newly started container.
	// However, there are some limitations of this method:
	//	1. When all dead containers were garbage collected, the container status could
	//	not get the historical value and would be *inaccurate*. Fortunately, the chance
	//	is really slim.
	//	2. When working with old version containers which have no restart count label,
	//	we can only assume their restart count is 0.
	// Anyhow, we only promised "best-effort" restart count reporting, we can just ignore
	// these limitations now.
	// TODO: move this comment to SyncPod.
	podSandboxIDs, err := m.getSandboxIDByPodUID(uid, nil)
	if err != nil {
		return nil, err
	}

	podFullName := format.Pod(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       uid,
		},
	})
	glog.V(4).Infof("getSandboxIDByPodUID got sandbox IDs %q for pod %q", podSandboxIDs, podFullName)

	sandboxStatuses := make([]*runtimeapi.PodSandboxStatus, len(podSandboxIDs))
	podIP := ""
	for idx, podSandboxID := range podSandboxIDs {
		podSandboxStatus, err := m.runtimeService.PodSandboxStatus(podSandboxID)
		if err != nil {
			glog.Errorf("PodSandboxStatus of sandbox %q for pod %q error: %v", podSandboxID, podFullName, err)
			return nil, err
		}
		sandboxStatuses[idx] = podSandboxStatus

		// Only get pod IP from latest sandbox
		if idx == 0 && podSandboxStatus.State == runtimeapi.PodSandboxState_SANDBOX_READY {
			podIP = m.determinePodSandboxIP(namespace, name, podSandboxStatus)
		}
	}

	// Get statuses of all containers visible in the pod.
	containerStatuses, err := m.getPodContainerStatuses(uid, name, namespace)
	if err != nil {
		glog.Errorf("getPodContainerStatuses for pod %q failed: %v", podFullName, err)
		return nil, err
	}

	return &kubecontainer.PodStatus{
		ID:                uid,
		Name:              name,
		Namespace:         namespace,
		IP:                podIP,
		SandboxStatuses:   sandboxStatuses,
		ContainerStatuses: containerStatuses,
	}, nil
}

// Returns the filesystem path of the pod's network namespace.
//
// For CRI, container network is handled by the runtime completely and this
// function should never be called.
func (m *kubeGenericRuntimeManager) GetNetNS(_ kubecontainer.ContainerID) (string, error) {
	return "", fmt.Errorf("not supported")
}

// GarbageCollect removes dead containers using the specified container gc policy.
func (m *kubeGenericRuntimeManager) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool, evictNonDeletedPods bool) error {
	return m.containerGC.GarbageCollect(gcPolicy, allSourcesReady, evictNonDeletedPods)
}

// GetPodContainerID gets pod sandbox ID
func (m *kubeGenericRuntimeManager) GetPodContainerID(pod *kubecontainer.Pod) (kubecontainer.ContainerID, error) {
	formattedPod := kubecontainer.FormatPod(pod)
	if len(pod.Sandboxes) == 0 {
		glog.Errorf("No sandboxes are found for pod %q", formattedPod)
		return kubecontainer.ContainerID{}, fmt.Errorf("sandboxes for pod %q not found", formattedPod)
	}

	// return sandboxID of the first sandbox since it is the latest one
	return pod.Sandboxes[0].ID, nil
}

// UpdatePodCIDR is just a passthrough method to update the runtimeConfig of the shim
// with the podCIDR supplied by the kubelet.
func (m *kubeGenericRuntimeManager) UpdatePodCIDR(podCIDR string) error {
	// TODO(#35531): do we really want to write a method on this manager for each
	// field of the config?
	glog.Infof("updating runtime config through cri with podcidr %v", podCIDR)
	return m.runtimeService.UpdateRuntimeConfig(
		&runtimeapi.RuntimeConfig{
			NetworkConfig: &runtimeapi.NetworkConfig{
				PodCidr: podCIDR,
			},
		})
}
