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
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/types"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

const (
	// The api version of kubelet runtime api
	kubeRuntimeAPIVersion = "0.1.0"
	// The root directory for pod logs
	podLogsRootDirectory = "/var/log/pods"
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
) (*kubeGenericRuntimeManager, error) {
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
	return nil, fmt.Errorf("not implemented")
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
	return fmt.Errorf("not implemented")
}

// GetPodStatus retrieves the status of the pod, including the
// information of all containers in the pod that are visble in Runtime.
func (m *kubeGenericRuntimeManager) GetPodStatus(uid kubetypes.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	return nil, fmt.Errorf("not implemented")
}

// Returns the filesystem path of the pod's network namespace; if the
// runtime does not handle namespace creation itself, or cannot return
// the network namespace path, it should return an error.
// TODO: Change ContainerID to a Pod ID since the namespace is shared
// by all containers in the pod.
func (m *kubeGenericRuntimeManager) GetNetNS(containerID kubecontainer.ContainerID) (string, error) {
	return "", fmt.Errorf("not implemented")
}

// GetPodContainerID gets pod sandbox ID
func (m *kubeGenericRuntimeManager) GetPodContainerID(pod *kubecontainer.Pod) (kubecontainer.ContainerID, error) {
	return kubecontainer.ContainerID{}, fmt.Errorf("not implemented")
}

// Forward the specified port from the specified pod to the stream.
func (m *kubeGenericRuntimeManager) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	return fmt.Errorf("not implemented")
}

// GarbageCollect removes dead containers using the specified container gc policy
func (m *kubeGenericRuntimeManager) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	return fmt.Errorf("not implemented")
}
