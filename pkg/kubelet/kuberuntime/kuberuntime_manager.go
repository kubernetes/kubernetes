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
	"io"
	"os"

	"github.com/coreos/go-semver/semver"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/credentialprovider"
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

const (
	// The api version of kubelet runtime api
	kubeRuntimeAPIVersion = "0.1.0"
	// The root directory for pod logs
	podLogsRootDirectory = "/var/log/pods"
	// A minimal shutdown window for avoiding unnecessary SIGKILLs
	minimumGracePeriodInSeconds = 2
)

var (
	// ErrVersionNotSupported is returned when the api version of runtime interface is not supported
	ErrVersionNotSupported = errors.New("Runtime api version is not supported")
)

type kubeGenericRuntimeManager struct {
	runtimeName         string
	recorder            record.EventRecorder
	osInterface         kubecontainer.OSInterface
	containerRefManager *kubecontainer.RefManager

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

	// Network plugin.
	networkPlugin network.NetworkPlugin

	// wrapped image puller.
	imagePuller images.ImageManager

	// gRPC service clients
	runtimeService internalApi.RuntimeService
	imageService   internalApi.ImageManagerService
}

// NewKubeGenericRuntimeManager creates a new kubeGenericRuntimeManager
func NewKubeGenericRuntimeManager(
	recorder record.EventRecorder,
	livenessManager proberesults.Manager,
	containerRefManager *kubecontainer.RefManager,
	osInterface kubecontainer.OSInterface,
	networkPlugin network.NetworkPlugin,
	runtimeHelper kubecontainer.RuntimeHelper,
	httpClient types.HttpGetter,
	imageBackOff *flowcontrol.Backoff,
	serializeImagePulls bool,
	cpuCFSQuota bool,
	runtimeService internalApi.RuntimeService,
	imageService internalApi.ImageManagerService,
) (kubecontainer.Runtime, error) {
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		recorder:            recorder,
		cpuCFSQuota:         cpuCFSQuota,
		livenessManager:     livenessManager,
		containerRefManager: containerRefManager,
		osInterface:         osInterface,
		networkPlugin:       networkPlugin,
		runtimeHelper:       runtimeHelper,
		runtimeService:      runtimeService,
		imageService:        imageService,
		keyring:             credentialprovider.NewDockerKeyring(),
	}

	typedVersion, err := kubeRuntimeManager.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Get runtime version failed: %v", err)
		return nil, err
	}

	// Only matching kubeRuntimeAPIVersion is supported now
	// TODO: Runtime API machinery is under discussion at https://github.com/kubernetes/kubernetes/issues/28642
	if typedVersion.GetVersion() != kubeRuntimeAPIVersion {
		glog.Errorf("Runtime api version %s is not supported, only %s is supported now",
			typedVersion.GetVersion(),
			kubeRuntimeAPIVersion)
		return nil, ErrVersionNotSupported
	}

	kubeRuntimeManager.runtimeName = typedVersion.GetRuntimeName()
	glog.Infof("Container runtime %s initialized, version: %s, apiVersion: %s",
		typedVersion.GetRuntimeName(),
		typedVersion.GetRuntimeVersion(),
		typedVersion.GetRuntimeApiVersion())

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
		serializeImagePulls)
	kubeRuntimeManager.runner = lifecycle.NewHandlerRunner(httpClient, kubeRuntimeManager, kubeRuntimeManager)

	return kubeRuntimeManager, nil
}

// Type returns the type of the container runtime.
func (m *kubeGenericRuntimeManager) Type() string {
	return m.runtimeName
}

// runtimeVersion implements kubecontainer.Version interface by implementing
// Compare() and String()
type runtimeVersion struct {
	*semver.Version
}

func newRuntimeVersion(version string) (runtimeVersion, error) {
	sem, err := semver.NewVersion(version)
	if err != nil {
		return runtimeVersion{}, err
	}
	return runtimeVersion{sem}, nil
}

func (r runtimeVersion) Compare(other string) (int, error) {
	v, err := semver.NewVersion(other)
	if err != nil {
		return -1, err
	}

	if r.LessThan(*v) {
		return -1, nil
	}
	if v.LessThan(*r.Version) {
		return 1, nil
	}
	return 0, nil
}

// Version returns the version information of the container runtime.
func (m *kubeGenericRuntimeManager) Version() (kubecontainer.Version, error) {
	typedVersion, err := m.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Get remote runtime version failed: %v", err)
		return nil, err
	}

	return newRuntimeVersion(typedVersion.GetVersion())
}

// APIVersion returns the cached API version information of the container
// runtime. Implementation is expected to update this cache periodically.
// This may be different from the runtime engine's version.
func (m *kubeGenericRuntimeManager) APIVersion() (kubecontainer.Version, error) {
	typedVersion, err := m.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Get remote runtime version failed: %v", err)
		return nil, err
	}

	return newRuntimeVersion(typedVersion.GetRuntimeApiVersion())
}

// Status returns error if the runtime is unhealthy; nil otherwise.
func (m *kubeGenericRuntimeManager) Status() error {
	_, err := m.runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		glog.Errorf("Checkout remote runtime status failed: %v", err)
		return err
	}

	return nil
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
		podUID := kubetypes.UID(s.Metadata.GetUid())
		if _, ok := pods[podUID]; !ok {
			pods[podUID] = &kubecontainer.Pod{
				ID:        podUID,
				Name:      s.Metadata.GetName(),
				Namespace: s.Metadata.GetNamespace(),
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

// SyncPod syncs the running pod into the desired pod.
func (m *kubeGenericRuntimeManager) SyncPod(pod *api.Pod, _ api.PodStatus,
	podStatus *kubecontainer.PodStatus, pullSecrets []api.Secret,
	backOff *flowcontrol.Backoff) (result kubecontainer.PodSyncResult) {
	result.Fail(fmt.Errorf("not implemented"))

	return
}

// KillPod kills all the containers of a pod. Pod may be nil, running pod must not be.
// gracePeriodOverride if specified allows the caller to override the pod default grace period.
// only hard kill paths are allowed to specify a gracePeriodOverride in the kubelet in order to not corrupt user data.
// it is useful when doing SIGKILL for hard eviction scenarios, or max grace period during soft eviction scenarios.
func (m *kubeGenericRuntimeManager) KillPod(pod *api.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) error {
	err := m.killPodWithSyncResult(pod, runningPod, gracePeriodOverride)
	return err.Error()
}

// killPodWithSyncResult kills a runningPod and returns SyncResult.
// Note: The pod passed in could be *nil* when kubelet restarted.
func (m *kubeGenericRuntimeManager) killPodWithSyncResult(pod *api.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) (result kubecontainer.PodSyncResult) {
	killContainerResults := m.killContainersWithSyncResult(pod, runningPod, gracePeriodOverride)
	for _, containerResult := range killContainerResults {
		result.AddSyncResult(containerResult)
	}

	// Teardown network plugin
	if len(runningPod.Sandboxes) == 0 {
		glog.V(4).Infof("Can not find pod sandbox by UID %q, assuming already removed.", runningPod.ID)
		return
	}

	sandboxID := runningPod.Sandboxes[0].ID.ID
	isHostNetwork, err := m.isHostNetwork(sandboxID, pod)
	if err != nil {
		result.Fail(err)
		return
	}
	if !isHostNetwork {
		teardownNetworkResult := kubecontainer.NewSyncResult(kubecontainer.TeardownNetwork, pod.UID)
		result.AddSyncResult(teardownNetworkResult)
		// Tear down network plugin with sandbox id
		if err := m.networkPlugin.TearDownPod(runningPod.Namespace, runningPod.Name, kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   sandboxID,
		}); err != nil {
			message := fmt.Sprintf("Failed to teardown network for pod %q using network plugins %q: %v",
				format.Pod(pod), m.networkPlugin.Name(), err)
			teardownNetworkResult.Fail(kubecontainer.ErrTeardownNetwork, message)
			glog.Error(message)
		}
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
func (m *kubeGenericRuntimeManager) isHostNetwork(podSandBoxID string, pod *api.Pod) (bool, error) {
	if pod != nil {
		return kubecontainer.IsHostNetworkPod(pod), nil
	}

	podStatus, err := m.runtimeService.PodSandboxStatus(podSandBoxID)
	if err != nil {
		return false, err
	}

	if podStatus.Linux != nil && podStatus.Linux.Namespaces != nil && podStatus.Linux.Namespaces.Options != nil {
		if podStatus.Linux.Namespaces.Options.HostNetwork != nil {
			return podStatus.Linux.Namespaces.Options.GetHostNetwork(), nil
		}
	}

	return false, nil
}

// GetPodStatus retrieves the status of the pod, including the
// information of all containers in the pod that are visble in Runtime.
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
	podFullName := kubecontainer.BuildPodFullName(name, namespace)
	podSandboxIDs, err := m.getSandboxIDByPodUID(string(uid), nil)
	if err != nil {
		return nil, err
	}
	glog.V(4).Infof("getSandboxIDByPodUID got sandbox IDs %q for pod %q(UID:%q)", podSandboxIDs, podFullName, string(uid))

	sandboxStatuses := make([]*runtimeApi.PodSandboxStatus, len(podSandboxIDs))
	containerStatuses := []*kubecontainer.ContainerStatus{}
	podIP := ""
	for idx, podSandboxID := range podSandboxIDs {
		podSandboxStatus, err := m.runtimeService.PodSandboxStatus(podSandboxID)
		if err != nil {
			glog.Errorf("PodSandboxStatus for pod (uid:%v, name:%s, namespace:%s) error: %v", uid, name, namespace, err)
			return nil, err
		}
		sandboxStatuses[idx] = podSandboxStatus

		// Only get pod IP from latest sandbox
		if idx == 0 && podSandboxStatus.GetState() == runtimeApi.PodSandBoxState_READY {
			podIP = m.determinePodSandboxIP(namespace, name, podSandboxStatus)
		}

		containerStatus, err := m.getKubeletContainerStatuses(podSandboxID)
		if err != nil {
			glog.Errorf("getKubeletContainerStatuses for sandbox %s failed: %v", podSandboxID, err)
			return nil, err
		}
		containerStatuses = append(containerStatuses, containerStatus...)
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

// Returns the filesystem path of the pod's network namespace; if the
// runtime does not handle namespace creation itself, or cannot return
// the network namespace path, it returns an 'not supported' error.
// TODO: Rename param name to sandboxID in kubecontainer.Runtime.GetNetNS().
// TODO: Remove GetNetNS after networking is delegated to the container runtime.
func (m *kubeGenericRuntimeManager) GetNetNS(sandboxID kubecontainer.ContainerID) (string, error) {
	readyState := runtimeApi.PodSandBoxState_READY
	filter := &runtimeApi.PodSandboxFilter{
		State:         &readyState,
		Id:            &sandboxID.ID,
		LabelSelector: map[string]string{kubernetesManagedLabel: "true"},
	}
	sandboxes, err := m.runtimeService.ListPodSandbox(filter)
	if err != nil {
		glog.Errorf("ListPodSandbox with filter %q failed: %v", filter, err)
		return "", err
	}
	if len(sandboxes) == 0 {
		glog.Errorf("No sandbox is found with filter %q", filter)
		return "", fmt.Errorf("Sandbox %q is not found", sandboxID)
	}

	sandboxStatus, err := m.runtimeService.PodSandboxStatus(sandboxes[0].GetId())
	if err != nil {
		glog.Errorf("PodSandboxStatus with id %q failed: %v", sandboxes[0].GetId(), err)
		return "", err
	}

	if sandboxStatus.Linux != nil && sandboxStatus.Linux.Namespaces != nil {
		return sandboxStatus.Linux.Namespaces.GetNetwork(), nil
	}

	return "", fmt.Errorf("not supported")
}

// GetPodContainerID gets pod sandbox ID
func (m *kubeGenericRuntimeManager) GetPodContainerID(pod *kubecontainer.Pod) (kubecontainer.ContainerID, error) {
	podFullName := kubecontainer.BuildPodFullName(pod.Name, pod.Namespace)
	if len(pod.Sandboxes) == 0 {
		glog.Errorf("No sandboxes are found for pod %q", podFullName)
		return kubecontainer.ContainerID{}, fmt.Errorf("sandboxes for pod %q not found", podFullName)
	}

	// return sandboxID of the first sandbox since it is the latest one
	return pod.Sandboxes[0].ID, nil
}

// Forward the specified port from the specified pod to the stream.
func (m *kubeGenericRuntimeManager) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	return fmt.Errorf("not implemented")
}

// GarbageCollect removes dead containers using the specified container gc policy
func (m *kubeGenericRuntimeManager) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	return fmt.Errorf("not implemented")
}
