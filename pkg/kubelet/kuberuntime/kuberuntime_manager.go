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
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"go.opentelemetry.io/otel/trace"
	grpcstatus "google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	ref "k8s.io/client-go/tools/reference"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/component-base/logs/logreduction"
	resourcehelper "k8s.io/component-helpers/resource"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	crierror "k8s.io/cri-api/pkg/errors"
	remote "k8s.io/cri-client/pkg"
	"k8s.io/klog/v2"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/credentialprovider"
	"k8s.io/kubernetes/pkg/credentialprovider/plugin"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	imagepullmanager "k8s.io/kubernetes/pkg/kubelet/images/pullmanager"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/logs"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	"k8s.io/kubernetes/pkg/kubelet/sysctl"
	"k8s.io/kubernetes/pkg/kubelet/token"
	"k8s.io/kubernetes/pkg/kubelet/types"
	kubeutil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/kubelet/util/cache"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/reconciler"
	"k8s.io/utils/ptr"
)

const (
	// The api version of kubelet runtime api
	kubeRuntimeAPIVersion = "0.1.0"
	// A minimal shutdown window for avoiding unnecessary SIGKILLs
	minimumGracePeriodInSeconds = 2

	// The expiration time of version cache.
	versionCacheTTL = 60 * time.Second
	// How frequently to report identical errors
	identicalErrorDelay = 1 * time.Minute
	// OpenTelemetry instrumentation scope name
	instrumentationScope = "k8s.io/kubernetes/pkg/kubelet/kuberuntime"
)

var (
	// ErrVersionNotSupported is returned when the api version of runtime interface is not supported
	ErrVersionNotSupported = errors.New("runtime api version is not supported")
)

// podStateProvider can determine if none of the elements are necessary to retain (pod content)
// or if none of the runtime elements are necessary to retain (containers)
type podStateProvider interface {
	IsPodTerminationRequested(kubetypes.UID) bool
	ShouldPodContentBeRemoved(kubetypes.UID) bool
	ShouldPodRuntimeBeRemoved(kubetypes.UID) bool
}

type PodInitContainerTimeRecorder interface {
	RecordInitContainerStarted(podUID kubetypes.UID, startedAt time.Time)
	RecordInitContainerFinished(podUID kubetypes.UID, finishedAt time.Time)
}

type kubeGenericRuntimeManager struct {
	runtimeName string
	recorder    record.EventRecorderLogger
	osInterface kubecontainer.OSInterface

	// machineInfo contains the machine information.
	machineInfo *cadvisorapi.MachineInfo

	// Container GC manager
	containerGC *containerGC

	// Runner of lifecycle events.
	runner kubecontainer.HandlerRunner

	// RuntimeHelper that wraps kubelet to generate runtime container options.
	runtimeHelper kubecontainer.RuntimeHelper

	// Health check results.
	livenessManager  proberesults.Manager
	readinessManager proberesults.Manager
	startupManager   proberesults.Manager

	// probeLifecycle handles container start/stop events for the probe manager.
	probeLifecycle kubecontainer.ContainerProbeLifecycle

	// If false, pass "memory.oom.group" to container cgroups when using cgroups v2 to cause processes
	// in those cgroups to be killed as a unit by the OOM killer.
	// It must be nil except for linux
	singleProcessOOMKill *bool

	// If true, enforce container cpu limits with CFS quota support
	cpuCFSQuota bool

	// CPUCFSQuotaPeriod sets the CPU CFS quota period value, cpu.cfs_period_us, defaults to 100ms
	cpuCFSQuotaPeriod metav1.Duration

	// Collection of Linux kernel parameters (sysctls) that will be applied to
	// the pods running on this node.
	defaultPodSysctls map[string]string

	// wrapped image puller.
	imagePuller images.ImageManager

	// gRPC service clients
	runtimeService internalapi.RuntimeService
	imageService   internalapi.ImageManagerService

	// The version cache of runtime daemon.
	versionCache *cache.ObjectCache

	// The directory path for seccomp profiles.
	seccompProfileRoot string

	// Container management interface for pod container.
	containerManager cm.ContainerManager

	// Internal lifecycle event handlers for container resource management.
	internalLifecycle cm.InternalContainerLifecycle

	// Manage container logs.
	logManager logs.ContainerLogManager

	// Manage RuntimeClass resources.
	runtimeClassManager *runtimeclass.Manager

	// actuatedState tracks actuated resources.
	actuatedState state.State

	// Cache last per-container error message to reduce log spam
	logReduction *logreduction.LogReduction

	// PodState provider instance
	podStateProvider podStateProvider

	// Use RuntimeDefault as the default seccomp profile for all workloads.
	seccompDefault bool

	// MemorySwapBehavior defines how swap is used
	memorySwapBehavior string

	//Function to get node allocatable resources
	getNodeAllocatable func() v1.ResourceList

	// Memory throttling factor for MemoryQoS
	// If nil, memory.high is not set (throttling is disabled).
	memoryThrottlingFactor *float64
	// Memory reservation policy for MemoryQoS memory.min behavior
	memoryReservationPolicy kubeletconfiginternal.MemoryReservationPolicy

	// Root directory used to store pod logs
	podLogsDirectory string

	// nodeIPs are the IP addresses of the node.
	nodeIPs []net.IP

	// Swap controller availability check function (Linux only)
	// Uses sync.OnceValue for lazy initialization
	getSwapControllerAvailable func() bool

	// Records first initContainer start time and last initContainer finish time
	podInitContainerTimeRecorder PodInitContainerTimeRecorder

	// podRestoreTimeout bounds the RestorePod CRI call. The CRI client's
	// default request timeout is too short for restores of large Pods.
	podRestoreTimeout time.Duration

	// restoresInFlightLock guards restoresInFlight.
	restoresInFlightLock sync.Mutex
	// restoresInFlight holds the namespace/name keys of pods that currently have
	// a sandbox restore in progress. It gates concurrent restores of the same
	// logical pod so a second attempt is rejected while the first is still
	// running. It is keyed by namespace/name rather than UID because each
	// restore attempt is admitted under a fresh pod UID.
	restoresInFlight sets.Set[string]
}

// KubeGenericRuntime is a interface contains interfaces for container runtime and streaming runtime.
type KubeGenericRuntime interface {
	kubecontainer.Runtime
	kubecontainer.StreamingRuntime
}

// NewKubeGenericRuntimeManager creates a new kubeGenericRuntimeManager
func NewKubeGenericRuntimeManager(
	ctx context.Context,
	recorder record.EventRecorderLogger,
	livenessManager proberesults.Manager,
	readinessManager proberesults.Manager,
	startupManager proberesults.Manager,
	probeLifecycle kubecontainer.ContainerProbeLifecycle,
	rootDirectory string,
	podLogsDirectory string,
	machineInfo *cadvisorapi.MachineInfo,
	podStateProvider podStateProvider,
	maxPods int32,
	osInterface kubecontainer.OSInterface,
	runtimeHelper kubecontainer.RuntimeHelper,
	insecureContainerLifecycleHTTPClient types.HTTPDoer,
	imageBackOff *flowcontrol.Backoff,
	serializeImagePulls bool,
	maxParallelImagePulls *int32,
	imagePullQPS float32,
	imagePullBurst int,
	imagePullsCredentialVerificationPolicy string,
	preloadedImagesCredentialVerificationWhitelist []string,
	imageCredentialProviderConfigPath string,
	imageCredentialProviderBinDir string,
	singleProcessOOMKill *bool,
	cpuCFSQuota bool,
	cpuCFSQuotaPeriod metav1.Duration,
	defaultPodSysctls map[string]string,
	runtimeService internalapi.RuntimeService,
	imageService internalapi.ImageManagerService,
	containerManager cm.ContainerManager,
	logManager logs.ContainerLogManager,
	runtimeClassManager *runtimeclass.Manager,
	seccompDefault bool,
	memorySwapBehavior string,
	getNodeAllocatable func() v1.ResourceList,
	memoryThrottlingFactor *float64,
	memoryReservationPolicy kubeletconfiginternal.MemoryReservationPolicy,
	podRestoreTimeout time.Duration,
	podPullingTimeRecorder images.ImagePodPullingTimeRecorder,
	tracerProvider trace.TracerProvider,
	tokenManager *token.Manager,
	getServiceAccount plugin.GetServiceAccountFunc,
	podInitContainerTimeRecorder PodInitContainerTimeRecorder,
	actuatedState state.State,
	nodeIPs []net.IP,
) (KubeGenericRuntime, []images.PostImageGCHook, error) {
	logger := klog.FromContext(ctx)

	tracer := tracerProvider.Tracer(instrumentationScope)
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		recorder:                     recorder,
		singleProcessOOMKill:         singleProcessOOMKill,
		cpuCFSQuota:                  cpuCFSQuota,
		cpuCFSQuotaPeriod:            cpuCFSQuotaPeriod,
		defaultPodSysctls:            defaultPodSysctls,
		seccompProfileRoot:           filepath.Join(rootDirectory, "seccomp"),
		livenessManager:              livenessManager,
		readinessManager:             readinessManager,
		startupManager:               startupManager,
		probeLifecycle:               probeLifecycle,
		machineInfo:                  machineInfo,
		osInterface:                  osInterface,
		runtimeHelper:                runtimeHelper,
		runtimeService:               runtimeService,
		imageService:                 imageService,
		containerManager:             containerManager,
		internalLifecycle:            containerManager.InternalContainerLifecycle(logger),
		logManager:                   logManager,
		runtimeClassManager:          runtimeClassManager,
		logReduction:                 logreduction.NewLogReduction(identicalErrorDelay),
		seccompDefault:               seccompDefault,
		memorySwapBehavior:           memorySwapBehavior,
		getNodeAllocatable:           getNodeAllocatable,
		memoryThrottlingFactor:       memoryThrottlingFactor,
		memoryReservationPolicy:      memoryReservationPolicy,
		podRestoreTimeout:            podRestoreTimeout,
		podLogsDirectory:             podLogsDirectory,
		nodeIPs:                      nodeIPs,
		podInitContainerTimeRecorder: podInitContainerTimeRecorder,
		restoresInFlight:             sets.New[string](),
	}

	// Initialize swap controller availability check with lazy evaluation
	kubeRuntimeManager.getSwapControllerAvailable = initSwapControllerAvailabilityCheck(ctx)

	typedVersion, err := kubeRuntimeManager.getTypedVersion(ctx)
	if err != nil {
		logger.Error(err, "Get runtime version failed")
		return nil, nil, err
	}

	// Only matching kubeRuntimeAPIVersion is supported now
	// TODO: Runtime API machinery is under discussion at https://github.com/kubernetes/kubernetes/issues/28642
	if typedVersion.Version != kubeRuntimeAPIVersion {
		logger.Error(err, "This runtime api version is not supported",
			"apiVersion", typedVersion.Version,
			"supportedAPIVersion", kubeRuntimeAPIVersion)
		return nil, nil, ErrVersionNotSupported
	}

	kubeRuntimeManager.runtimeName = typedVersion.RuntimeName
	logger.Info("Container runtime initialized",
		"containerRuntime", typedVersion.RuntimeName,
		"version", typedVersion.RuntimeVersion,
		"apiVersion", typedVersion.RuntimeApiVersion)

	if imageCredentialProviderConfigPath != "" || imageCredentialProviderBinDir != "" {
		if err := plugin.RegisterCredentialProviderPlugins(imageCredentialProviderConfigPath, imageCredentialProviderBinDir, tokenManager.GetServiceAccountToken, getServiceAccount); err != nil {
			logger.Error(err, "Failed to register CRI auth plugins")
			os.Exit(1)
		}
	}

	var imageGCHooks []images.PostImageGCHook
	var imagePullManager imagepullmanager.ImagePullManager = &imagepullmanager.NoopImagePullManager{}
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletEnsureSecretPulledImages) {
		imagePullCredentialsVerificationPolicy, err := imagepullmanager.NewImagePullCredentialVerificationPolicy(
			kubeletconfiginternal.ImagePullCredentialsVerificationPolicy(imagePullsCredentialVerificationPolicy),
			preloadedImagesCredentialVerificationWhitelist)

		if err != nil {
			return nil, nil, err
		}

		fsRecordAccessor, err := imagepullmanager.NewFSPullRecordsAccessor(logger, rootDirectory)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to setup the FSPullRecordsAccessor: %w", err)
		}

		var ( // variables used to determine cache/lock set sizes
			maxParallelPulls     = ptr.Deref(maxParallelImagePulls, 0)
			intentCacheSize      = max(2*maxPods, 2*maxParallelPulls)
			pullRecordsCacheSize = 5 * maxPods
		)

		memCacheRecordsAccessor := imagepullmanager.NewCachedPullRecordsAccessor(logger, fsRecordAccessor, intentCacheSize, pullRecordsCacheSize, maxParallelPulls)

		imagePullManager, err = imagepullmanager.NewImagePullManager(ctx, memCacheRecordsAccessor, imagePullCredentialsVerificationPolicy, kubeRuntimeManager, maxParallelPulls)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create image pull manager: %w", err)
		}

		imageGCHooks = append(imageGCHooks, imagePullManager.PruneUnknownRecords)
	}

	nodeKeyring := credentialprovider.NewDefaultDockerKeyring()
	kubeRuntimeManager.imagePuller = images.NewImageManager(
		kubecontainer.FilterEventRecorder(recorder),
		nodeKeyring,
		kubeRuntimeManager,
		imagePullManager,
		imageBackOff,
		serializeImagePulls,
		maxParallelImagePulls,
		imagePullQPS,
		imagePullBurst,
		podPullingTimeRecorder)
	kubeRuntimeManager.runner = lifecycle.NewHandlerRunner(insecureContainerLifecycleHTTPClient, kubecontainer.NewCommandRunner(runtimeService), kubeRuntimeManager, recorder)
	kubeRuntimeManager.containerGC = newContainerGC(runtimeService, podStateProvider, kubeRuntimeManager, tracer)
	kubeRuntimeManager.podStateProvider = podStateProvider

	kubeRuntimeManager.versionCache = cache.NewObjectCache(
		func() (interface{}, error) {
			return kubeRuntimeManager.getTypedVersion(ctx)
		},
		versionCacheTTL,
	)

	kubeRuntimeManager.actuatedState = actuatedState

	return kubeRuntimeManager, imageGCHooks, nil
}

// Type returns the type of the container runtime.
func (m *kubeGenericRuntimeManager) Type() string {
	return m.runtimeName
}

func newRuntimeVersion(version string) (*utilversion.Version, error) {
	if ver, err := utilversion.ParseSemantic(version); err == nil {
		return ver, err
	}
	return utilversion.ParseGeneric(version)
}

func (m *kubeGenericRuntimeManager) getTypedVersion(ctx context.Context) (*runtimeapi.VersionResponse, error) {
	typedVersion, err := m.runtimeService.Version(ctx, kubeRuntimeAPIVersion)
	if err != nil {
		return nil, fmt.Errorf("get remote runtime typed version failed: %v", err)
	}
	return typedVersion, nil
}

// Version returns the version information of the container runtime.
func (m *kubeGenericRuntimeManager) Version(ctx context.Context) (kubecontainer.Version, error) {
	typedVersion, err := m.getTypedVersion(ctx)
	if err != nil {
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
func (m *kubeGenericRuntimeManager) Status(ctx context.Context) (*kubecontainer.RuntimeStatus, error) {
	resp, err := m.runtimeService.Status(ctx, false)
	if err != nil {
		return nil, err
	}
	if resp.GetStatus() == nil {
		return nil, errors.New("runtime status is nil")
	}
	return toKubeRuntimeStatus(resp.GetStatus(), resp.GetRuntimeHandlers(), resp.GetFeatures()), nil
}

// GetPod fetches the current sandboxes & containers for a pod.
func (m *kubeGenericRuntimeManager) GetPod(ctx context.Context, podUID kubetypes.UID) (*kubecontainer.Pod, error) {
	pods, err := m.getPods(ctx, listOptions{podUID: podUID})
	if err != nil {
		return nil, err
	}

	pod := pods[podUID]
	if pod == nil {
		return nil, kubecontainer.ErrPodNotFound
	}
	return pod, nil
}

// GetPods returns a list of containers grouped by pods. The boolean parameter
// specifies whether the runtime returns all containers including those already
// exited and dead containers (used for garbage collection).
func (m *kubeGenericRuntimeManager) GetPods(ctx context.Context, all bool) ([]*kubecontainer.Pod, error) {
	logger := klog.FromContext(ctx)
	pods, err := m.getPods(ctx, listOptions{onlyRunningReady: !all})
	if err != nil {
		return nil, err
	}

	// Convert map to list.
	var result []*kubecontainer.Pod
	for _, pod := range pods {
		result = append(result, pod)
	}

	// There are scenarios where multiple pods are running in parallel having
	// the same name, because one of them have not been fully terminated yet.
	// To avoid unexpected behavior on container name based search (for example
	// by calling *Kubelet.findContainer() without specifying a pod ID), we now
	// return the list of pods ordered by their creation time.
	sort.SliceStable(result, func(i, j int) bool {
		return result[i].CreatedAt > result[j].CreatedAt
	})
	logger.V(4).Info("Retrieved pods from runtime", "all", all)
	return result, nil
}

func (m *kubeGenericRuntimeManager) getPods(ctx context.Context, opts listOptions) (map[kubetypes.UID]*kubecontainer.Pod, error) {
	logger := klog.FromContext(ctx)
	pods := make(map[kubetypes.UID]*kubecontainer.Pod)
	timestamp := time.Now()
	sandboxes, err := m.getSandboxes(ctx, opts)
	if err != nil {
		return nil, err
	}
	// Sort sandboxes by creation time, newest first.
	sort.Sort(podSandboxByCreatedThenID(sandboxes))
	for i := range sandboxes {
		s := sandboxes[i]
		if s.Metadata == nil {
			logger.V(4).Info("Sandbox does not have metadata", "sandbox", s)
			continue
		}
		podUID := kubetypes.UID(s.Metadata.Uid)
		if _, ok := pods[podUID]; !ok {
			pods[podUID] = &kubecontainer.Pod{
				ID:        podUID,
				Name:      s.Metadata.Name,
				Namespace: s.Metadata.Namespace,
				Timestamp: timestamp,
			}
		}
		p := pods[podUID]
		converted, err := m.sandboxToKubeContainer(s)
		if err != nil {
			logger.V(4).Info("Convert sandbox of pod failed", "runtimeName", m.runtimeName, "sandbox", s, "podUID", podUID, "err", err)
			continue
		}
		p.Sandboxes = append(p.Sandboxes, converted)
		p.CreatedAt = uint64(s.GetCreatedAt())
	}

	containers, err := m.getContainers(ctx, opts)
	if err != nil {
		return nil, err
	}
	// Sort containers: newest CreatedAt first, then by container ID for stability.
	// There are scenarios where multiple pods are running in parallel having
	// the same name, because one of them have not been fully terminated yet.
	// To avoid unexpected behavior on container name based search (for example
	// by calling *Kubelet.findContainer() without specifying a pod ID), we
	// return the list of pods ordered by their creation time.
	sort.Sort(containerByCreatedThenID(containers))
	for i := range containers {
		c := containers[i]
		if c.Metadata == nil {
			logger.V(4).Info("Container does not have metadata", "container", c)
			continue
		}

		labelledInfo := getContainerInfoFromLabels(ctx, c.Labels)
		pod, found := pods[labelledInfo.PodUID]
		if !found {
			pod = &kubecontainer.Pod{
				ID:        labelledInfo.PodUID,
				Name:      labelledInfo.PodName,
				Namespace: labelledInfo.PodNamespace,
				Timestamp: timestamp,
			}
			pods[labelledInfo.PodUID] = pod
		}

		converted, err := m.toKubeContainer(ctx, c)
		if err != nil {
			logger.V(4).Info("Convert container of pod failed", "runtimeName", m.runtimeName, "container", c, "podUID", labelledInfo.PodUID, "err", err)
			continue
		}

		pod.Containers = append(pod.Containers, converted)
	}

	return pods, nil
}

// containerKillReason explains what killed a given container
type containerKillReason string

const (
	reasonStartupProbe         containerKillReason = "StartupProbe"
	reasonLivenessProbe        containerKillReason = "LivenessProbe"
	reasonFailedPostStartHook  containerKillReason = "FailedPostStartHook"
	reasonRestartAllContainers containerKillReason = "RestartAllContainers"
	reasonUnknown              containerKillReason = "Unknown"
)

// containerToKillInfo contains necessary information to kill a container.
type containerToKillInfo struct {
	// The spec of the container.
	container *v1.Container
	// The name of the container.
	name string
	// The message indicates why the container will be killed.
	message string
	// The reason is a clearer source of info on why a container will be killed
	// TODO: replace message with reason?
	reason containerKillReason
}

// containerToUpdateInfo contains necessary information to update a container's resources.
type containerToUpdateInfo struct {
	// The spec of the container.
	container *v1.Container
	// Type of the container.
	containerType podutil.ContainerType
	// ID of the runtime container that needs resource update
	kubeContainerID kubecontainer.ContainerID
	// Desired resources for the running container
	desiredContainerResources resourceRequirements
	// Most recently configured resources on the running container
	currentContainerResources *resourceRequirements
}

// containerToRemoveInfo contains necessary information to update a container's resources.
type containerToRemoveInfo struct {
	// The ID of the container.
	containerID kubecontainer.ContainerID
	// The spec of the container.
	container *v1.Container
	// Whether to kill the container before removal.
	kill bool
}

// podActions keeps information what to do for a pod.
type podActions struct {
	// Stop all running (regular, init and ephemeral) containers and the sandbox for the pod.
	KillPod bool
	// Whether need to create a new sandbox. If needed to kill pod and create
	// a new pod sandbox, all init containers need to be purged (i.e., removed).
	CreateSandbox bool
	// The id of existing sandbox. It is used for starting containers in ContainersToStart.
	SandboxID string
	// The attempt number of creating sandboxes for the pod.
	Attempt uint32

	// InitContainersToStart keeps a list of indexes for the init containers to
	// start, where the index is the index of the specific init container in the
	// pod spec (pod.Spec.InitContainers).
	InitContainersToStart []int
	// ContainersToStart keeps a list of indexes for the containers to start,
	// where the index is the index of the specific container in the pod spec (
	// pod.Spec.Containers).
	ContainersToStart []int
	// ContainersToKill keeps a map of containers that need to be killed, note that
	// the key is the container ID of the container, while
	// the value contains necessary information to kill a container.
	ContainersToKill map[kubecontainer.ContainerID]containerToKillInfo
	// EphemeralContainersToStart is a list of indexes for the ephemeral containers to start,
	// where the index is the index of the specific container in pod.Spec.EphemeralContainers.
	EphemeralContainersToStart []int
	// ContainersToUpdate keeps a list of containers needing resource update.
	// Container resource update is applicable only for CPU and memory.
	ContainersToUpdate map[v1.ResourceName][]containerToUpdateInfo
	// VolumesToUpsize keeps a list of volumes needing size increase.
	VolumesToUpsize []v1.Volume
	// VolumesToDownsize keeps a list of volumes needing size decrease.
	VolumesToDownsize []v1.Volume
	// PodCgroupResourcesToUpdate is the set of resources (cpu, memory, and swap for a
	// pod-level swap limit) whose pod-level cgroup configuration needs to be updated.
	PodCgroupResourcesToUpdate sets.Set[v1.ResourceName]
	// ContainersToReset is a list of containers to be killed (if running) and removed from
	// runtime for RestartAllContainers. The container that triggered RestartAllContainers
	// will be reset the last.
	ContainersToReset []containerToRemoveInfo
}

// resourceRequirements summarizes the set of resources applicable to the
// running pod or container in kuberuntime context.
type resourceRequirements struct {
	memoryLimit   int64
	memoryRequest int64
	cpuLimit      int64
	cpuRequest    int64
	// swapLimit is the effective memory.swap.max of a container (KEP-5359), not
	// the raw limits.swap entry: it already folds in pod-level inheritance and
	// the LimitedSwap recalculation, so comparing two of these tells whether the
	// cgroup value must change.
	swapLimit int64
}

func (p podActions) String() string {
	return fmt.Sprintf("KillPod: %t, CreateSandbox: %t, PodCgroupResourcesToUpdate: %v, Attempt: %d, InitContainersToStart: %v, ContainersToStart: %v, EphemeralContainersToStart: %v, ContainersToUpdate: %v, VolumesToUpsize: %v, VolumesToDownsize: %v, ContainersToKill: %v, ContainersToRemove: %v",
		p.KillPod, p.CreateSandbox, p.PodCgroupResourcesToUpdate, p.Attempt, p.InitContainersToStart, p.ContainersToStart, p.EphemeralContainersToStart, p.ContainersToUpdate, p.VolumesToUpsize, p.VolumesToDownsize, p.ContainersToKill, p.ContainersToReset)
}

// containerChanged will determine whether the container has changed based on the fields that will affect the running of the container.
// Currently, there are only `image` and `name` fields.
// we don't need to consider the pod UID here, because we find the containerStatus through the pod UID.
// If the pod UID changes, we will not be able to find the containerStatus to compare against.
func containerChanged(container *v1.Container, containerStatus *kubecontainer.Status) (uint64, uint64, bool) {
	expectedHash := kubecontainer.HashContainer(container)
	return expectedHash, containerStatus.Hash, containerStatus.Hash != expectedHash
}

func shouldRestartOnFailure(pod *v1.Pod) bool {
	return pod.Spec.RestartPolicy != v1.RestartPolicyNever
}

func containerSucceeded(c *v1.Container, podStatus *kubecontainer.PodStatus) bool {
	cStatus := podStatus.FindContainerStatusByName(c.Name)
	if cStatus == nil {
		return false
	}
	// Container has exited, with an exit code of 0.
	return cStatus.State == kubecontainer.ContainerStateExited && cStatus.ExitCode == 0
}

func containerResourcesFromRequirements(podRequirements, containerRequirements *v1.ResourceRequirements) resourceRequirements {
	resources := resourceRequirements{
		memoryLimit:   containerRequirements.Limits.Memory().Value(),
		memoryRequest: containerRequirements.Requests.Memory().Value(),
		cpuLimit:      containerRequirements.Limits.Cpu().MilliValue(),
		cpuRequest:    containerRequirements.Requests.Cpu().MilliValue(),
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) {
		return resources
	}

	containerLimits := kubeutil.GetLimits(&kubeutil.ResourceOpts{PodResources: podRequirements, ContainerResources: containerRequirements})
	resources.memoryLimit = containerLimits.Memory().Value()
	resources.cpuLimit = containerLimits.Cpu().MilliValue()
	return resources
}

// unlimitedCgroupLimit is the pod cgroup memory limit or CPU quota that sets no limit.
const unlimitedCgroupLimit int64 = -1

// cgroupLimit returns the given pod cgroup limit for comparison, treating unlimited as the maximum.
func cgroupLimit(limit int64) int64 {
	if limit < 0 {
		return math.MaxInt64
	}
	return limit
}

// setResourceQuantity returns dst with the quantity of name copied from src, or removed if src
// doesn't have it. It returns nil instead of an empty list.
func setResourceQuantity(dst, src v1.ResourceList, name v1.ResourceName) v1.ResourceList {
	if qty, ok := src[name]; ok {
		if dst == nil {
			dst = v1.ResourceList{}
		}
		dst[name] = qty
	} else {
		delete(dst, name)
	}
	if len(dst) == 0 {
		return nil
	}
	return dst
}

// workloadSwapLimitsEnabled reports whether workload-specified limits.swap, and
// in-place resize of swap, are honored on this node: the WorkloadControlledSwap
// gate is on and the node can configure swap. With the gate off a LimitedSwap
// node still gets its node-derived swap limits, and this returns false.
func (m *kubeGenericRuntimeManager) workloadSwapLimitsEnabled() bool {
	return utilfeature.DefaultFeatureGate.Enabled(features.WorkloadControlledSwap) && m.nodeSwapEnabled()
}

// podSwapLimitEqual compares pod-level limits.swap. Presence matters because
// an unset pod-level swap limit means unlimited.
func podSwapLimitEqual(a, b *v1.ResourceRequirements) bool {
	var aSwap, bSwap resource.Quantity
	var aHas, bHas bool
	if a != nil {
		aSwap, aHas = a.Limits[v1.ResourceSwap]
	}
	if b != nil {
		bSwap, bHas = b.Limits[v1.ResourceSwap]
	}
	return aHas == bHas && aSwap.Equal(bSwap)
}

// actuatedPodLevelResources returns the pod-level resources to record as actuated for pod: the
// effective cpu and memory that size the pod cgroup, plus the pod-level swap limit (KEP-5359),
// which is only ever set explicitly.
func actuatedPodLevelResources(pod *v1.Pod) *v1.ResourceRequirements {
	return withPodSwapLimit(cm.EffectivePodResources(pod), pod.Spec.Resources)
}

// withPodSwapLimit returns resources with the pod-level swap limit of src, if it has one.
func withPodSwapLimit(resources, src *v1.ResourceRequirements) *v1.ResourceRequirements {
	if src == nil {
		return resources
	}
	swapLimit, ok := src.Limits[v1.ResourceSwap]
	if !ok {
		return resources
	}
	if resources == nil {
		resources = &v1.ResourceRequirements{}
	}
	if resources.Limits == nil {
		resources.Limits = v1.ResourceList{}
	}
	resources.Limits[v1.ResourceSwap] = swapLimit
	return resources
}

// actuatedSwapLimit returns the swap limit container currently runs with. It is derived from the
// actuated container resources and the actuated pod-level swap limit in the same way that
// effectiveSwapLimit derives the desired one from the pod spec.
func (m *kubeGenericRuntimeManager) actuatedSwapLimit(pod *v1.Pod, container *v1.Container, actuatedContainerResources *v1.ResourceRequirements) int64 {
	podCtx := pod
	if actuatedPodResources, _ := m.actuatedState.GetPodLevelResources(pod.UID); !podSwapLimitEqual(pod.Spec.Resources, actuatedPodResources) {
		// Shallow copies: only the pod-level swap limit differs from the pod spec.
		podCopy := *pod
		var podResources v1.ResourceRequirements
		if pod.Spec.Resources != nil {
			podResources = *pod.Spec.Resources
		}
		var actuatedLimits v1.ResourceList
		if actuatedPodResources != nil {
			actuatedLimits = actuatedPodResources.Limits
		}
		podResources.Limits = setResourceQuantity(podResources.Limits.DeepCopy(), actuatedLimits, v1.ResourceSwap)
		podCopy.Spec.Resources = &podResources
		podCtx = &podCopy
	}
	containerCtx := *container
	containerCtx.Resources = *actuatedContainerResources
	return m.effectiveSwapLimit(podCtx, &containerCtx)
}

// computePodResizeAction determines the actions required (if any) to resize the given container.
// Returns whether to keep (true) or restart (false) the container.
// TODO(vibansal): Make this function to be agnostic to whether it is dealing with a restartable init container or not (i.e. remove the argument `isRestartableInitContainer`).
func (m *kubeGenericRuntimeManager) computePodResizeAction(ctx context.Context, pod *v1.Pod, containerIdx int, initContainer bool, kubeContainerStatus *kubecontainer.Status, changes *podActions) (keepContainer bool) {
	logger := klog.FromContext(ctx)
	if resizable, _, _ := allocation.IsInPlacePodVerticalScalingAllowed(pod); !resizable {
		return true
	}

	var container v1.Container
	containerType := podutil.Containers
	if initContainer {
		container = pod.Spec.InitContainers[containerIdx]
		containerType = podutil.InitContainers
	} else {
		container = pod.Spec.Containers[containerIdx]
	}

	// Determine if the *running* container needs resource update by comparing v1.Spec.Resources (desired)
	// with v1.Status.Resources / runtime.Status.Resources (last known actual).
	// Proceed only when kubelet has accepted the resize a.k.a v1.Spec.Resources.Requests == v1.Status.AllocatedResources.
	// Skip if runtime containerID doesn't match pod.Status containerID (container is restarting)
	if kubeContainerStatus.State != kubecontainer.ContainerStateRunning {
		return true
	}

	actuatedContainerResources, found := m.actuatedState.GetContainerResources(pod.UID, container.Name)
	if !found {
		logger.Error(nil, "Missing actuated resource record", "pod", klog.KObj(pod), "container", container.Name)
		// Proceed with the zero-value actuated resources. For restart NotRequired, this may
		// result in an extra call to UpdateContainerResources, but that call should be idempotent.
		// For RestartContainer, this may trigger a container restart.
	}

	desiredResources := containerResourcesFromRequirements(pod.Spec.Resources, &container.Resources)
	// The actuated container limits already include any limits applied from pod-level limits.
	currentResources := containerResourcesFromRequirements(nil, &actuatedContainerResources)
	if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadControlledSwap) {
		desiredResources.swapLimit = m.effectiveSwapLimit(pod, &container)
		currentResources.swapLimit = m.actuatedSwapLimit(pod, &container, &actuatedContainerResources)
	}

	if currentResources == desiredResources {
		// No resize required.
		return true
	}

	determineContainerResize := func(rName v1.ResourceName, desiredValue, currentValue int64) (resize, restart bool) {
		if desiredValue == currentValue {
			return false, false
		}
		for _, policy := range container.ResizePolicy {
			if policy.ResourceName == rName {
				return true, policy.RestartPolicy == v1.RestartContainer
			}
		}
		// If a resource policy isn't set, the implicit default is NotRequired.
		// Note: `swap` is not accepted in resizePolicy today, so swap changes are
		// always NotRequired (KEP-5359 open question D6).
		return true, false
	}
	markContainerForUpdate := func(rName v1.ResourceName, desiredValue, currentValue int64) {
		cUpdateInfo := containerToUpdateInfo{
			container:                 &container,
			containerType:             containerType,
			kubeContainerID:           kubeContainerStatus.ID,
			desiredContainerResources: desiredResources,
			currentContainerResources: &currentResources,
		}
		// Order the container updates such that resource decreases are applied before increases
		switch {
		case desiredValue > currentValue: // append
			changes.ContainersToUpdate[rName] = append(changes.ContainersToUpdate[rName], cUpdateInfo)
		case desiredValue < currentValue: // prepend
			changes.ContainersToUpdate[rName] = append(changes.ContainersToUpdate[rName], containerToUpdateInfo{})
			copy(changes.ContainersToUpdate[rName][1:], changes.ContainersToUpdate[rName])
			changes.ContainersToUpdate[rName][0] = cUpdateInfo
		}
	}
	resizeMemLim, restartMemLim := determineContainerResize(v1.ResourceMemory, desiredResources.memoryLimit, currentResources.memoryLimit)
	resizeMemReq, restartMemReq := determineContainerResize(v1.ResourceMemory, desiredResources.memoryRequest, currentResources.memoryRequest)
	resizeCPULim, restartCPULim := determineContainerResize(v1.ResourceCPU, desiredResources.cpuLimit, currentResources.cpuLimit)
	resizeCPUReq, restartCPUReq := determineContainerResize(v1.ResourceCPU, desiredResources.cpuRequest, currentResources.cpuRequest)
	// A swap delta can come from a user change to limits.swap, from a pod-level
	// limit change, or from LimitedSwap recalculating after a memory request
	// change. All three are actuated the same way (own ordered pass, see
	// doPodResizeAction); admission decides which are allowed (see
	// allocation/handlers.go).
	resizeSwapLim, _ := determineContainerResize(v1.ResourceSwap, desiredResources.swapLimit, currentResources.swapLimit)
	if restartCPULim || restartCPUReq || restartMemLim || restartMemReq {
		// resize policy requires this container to restart
		changes.ContainersToKill[kubeContainerStatus.ID] = containerToKillInfo{
			name:      kubeContainerStatus.Name,
			container: &container,
			message:   fmt.Sprintf("Container %s resize requires restart", container.Name),
		}
		if initContainer {
			// Validation prevents non-restartable init containers from being marked for restart,
			// so we know this init container is restartable.
			changes.InitContainersToStart = append(changes.InitContainersToStart, containerIdx)
		} else {
			changes.ContainersToStart = append(changes.ContainersToStart, containerIdx)
		}
		return false
	} else {
		if resizeMemLim {
			markContainerForUpdate(v1.ResourceMemory, desiredResources.memoryLimit, currentResources.memoryLimit)
		} else if resizeMemReq {
			markContainerForUpdate(v1.ResourceMemory, desiredResources.memoryRequest, currentResources.memoryRequest)
		}
		if resizeCPULim {
			markContainerForUpdate(v1.ResourceCPU, desiredResources.cpuLimit, currentResources.cpuLimit)
		} else if resizeCPUReq {
			markContainerForUpdate(v1.ResourceCPU, desiredResources.cpuRequest, currentResources.cpuRequest)
		}
		if resizeSwapLim {
			markContainerForUpdate(v1.ResourceSwap, desiredResources.swapLimit, currentResources.swapLimit)
		}
	}
	return true
}

// TODO: Move InitializeActuatedPod, UpdateActuatedPodLevelResources, and IsPodResizeInProgress
// out of the runtime now that actuatedState is owned by the Kubelet.
func (m *kubeGenericRuntimeManager) InitializeActuatedPod(logger klog.Logger, allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	stored, hasPod := m.actuatedState.GetPod(allocatedPod.UID)
	// TODO: In v1.39, when the V1 checkpoint format is no longer supported, we will still need to
	// check stored.Name == "" (and preserve stored.Spec.Resources and stored.Spec.Volumes below)
	// because SyncPod calls UpdateActuatedPodLevelResources and WaitForAttachAndMount before
	// InitializeActuatedPod, which creates a nameless pod entry.
	if hasPod && !state.IsMigratedFromV1(stored) {
		return
	}
	// If there is no actuated checkpoint for the pod, seed it with the allocated pod.
	// TODO(KEP-5972): Revisit what is seeded for containers that haven't started yet (only their
	// resources, see KeepOnlyResourceValues) together with tracking the actual pod-level resources,
	// which depend on which containers are running.
	actuatedPod := *allocatedPod.DeepCopy()
	for c := range podutil.ContainerIter(&actuatedPod.Spec, podutil.InitContainers|podutil.Containers) {
		c.Resources = actuatedContainerResources(allocatedPod, c)
	}
	if hasPod {
		// TODO: Remove V1 overlay in v1.39 when the V1 checkpoint format is no longer supported,
		// while still preserving stored.Spec.Resources set by UpdateActuatedPodLevelResources.
		if updatedPod, updated := state.UpdatePodFromV1(&actuatedPod, stored); updated {
			actuatedPod = *updatedPod
		}
	}
	if hasPod && stored.Spec.Resources != nil {
		actuatedPod.Spec.Resources = withPodSwapLimit(cm.ResizableResources(stored.Spec.Resources), stored.Spec.Resources)
	} else {
		actuatedPod.Spec.Resources = actuatedPodLevelResources(&actuatedPod)
	}
	if err := m.actuatedState.SetPod(logger, &actuatedPod); err != nil {
		logger.Error(err, "Failed to initialize actuated pod resource info checkpoint", "pod", klog.KObj(&actuatedPod))
	}
	m.backfillActuatedFromV1(logger, allocatedPod, &actuatedPod, podStatus)
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) && hasPod {
		for _, vol := range stored.Spec.Volumes {
			if !allocation.VolHasMemoryBackedEmptyDir(&vol) {
				continue
			}
			effectiveLimit, err := m.runtimeHelper.EffectiveEphemeralVolumeLimit(&actuatedPod, vol.Name)
			if err != nil {
				logger.Error(err, "Failed to calculate effective emptyDir volume limit", "pod", klog.KObj(&actuatedPod), "volume", vol.Name)
				continue
			}
			if vol.EmptyDir.SizeLimit != nil && effectiveLimit != nil && vol.EmptyDir.SizeLimit.Equal(*effectiveLimit) {
				continue
			}
			actuatedEmptyDir := &v1.EmptyDirVolumeSource{
				Medium:    vol.EmptyDir.Medium,
				SizeLimit: effectiveLimit,
			}
			if err := m.actuatedState.SetEmptyDirVolumeLimit(logger, allocatedPod.UID, vol.Name, actuatedEmptyDir); err != nil {
				logger.Error(err, "Failed to initialize actuated emptyDir volume limit checkpoint", "pod", klog.KObj(&actuatedPod), "volume", vol.Name)
			}
		}
	}
}

// backfillActuatedFromV1 populates actuated container specs and mounted OCI Image volumes for
// containers that had already been created when no complete V2 actuated checkpoint exists yet
// (e.g. upgrading from a V1 checkpoint or initializing a BestEffort pod that had no V1 entry).
// It runs after SetPod rather than inside state.UpdatePodFromV1 because SetPod applies
// KeepOnlyResourceValues and requires runtime podStatus.
// TODO: Remove in v1.39 when the V1 checkpoint format is no longer supported.
func (m *kubeGenericRuntimeManager) backfillActuatedFromV1(logger klog.Logger, allocatedPod, actuatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	dynamicVolumesEnabled := utilfeature.DefaultFeatureGate.Enabled(features.DynamicNodeLocalEphemeralVolumes)
	if podStatus == nil || (!dynamicVolumesEnabled && !utilfeature.DefaultFeatureGate.Enabled(features.DynamicContainers)) {
		return
	}
	hasRegularCreated := HasAnyRegularContainerCreated(allocatedPod, podStatus)
	for specCtr, cType := range podutil.ContainerIter(&allocatedPod.Spec, podutil.AllContainers) {
		cs := podStatus.FindContainerStatusByName(specCtr.Name)
		if cs == nil && (cType != podutil.InitContainers || !hasRegularCreated) {
			continue
		}
		backfilled := *specCtr.DeepCopy()
		if actCtr := findContainerInPodSpec(&actuatedPod.Spec, specCtr.Name, cType); actCtr != nil {
			backfilled.Resources = actCtr.Resources
		} else {
			backfilled.Resources = actuatedContainerResources(allocatedPod, specCtr)
		}
		if err := m.actuatedState.SetContainer(logger, allocatedPod.UID, cType, backfilled); err != nil {
			logger.Error(err, "Failed to backfill actuated container spec from V1 checkpoint", "pod", klog.KObj(actuatedPod), "containerName", specCtr.Name)
		}
		if dynamicVolumesEnabled {
			for _, vm := range backfilled.VolumeMounts {
				volIdx := slices.IndexFunc(allocatedPod.Spec.Volumes, func(v v1.Volume) bool { return v.Name == vm.Name })
				if volIdx < 0 || allocatedPod.Spec.Volumes[volIdx].Image == nil {
					continue
				}
				if err := m.actuatedState.SetPodVolume(logger, allocatedPod.UID, allocatedPod.Spec.Volumes[volIdx]); err != nil {
					logger.Error(err, "Failed to backfill actuated image volume from V1 checkpoint", "pod", klog.KObj(actuatedPod), "volumeName", vm.Name)
				}
			}
		}
	}
}

// KeepOnlyResourceValues strips everything from pod.Spec except the resizable resources, because
// those are all SetPod initializes in the actuated state (mounted volumes are recorded via
// SetPodVolume and preserved by SetPod, and container VolumeMounts are recorded when each
// container starts).
func KeepOnlyResourceValues(pod *v1.Pod) {
	pod.Generation = 0
	spec := v1.PodSpec{
		Resources: pod.Spec.Resources,
	}
	for container, containerType := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		if !allocation.IsResizableContainer(container, containerType) {
			continue
		}
		actuated := v1.Container{
			Name:      container.Name,
			Resources: container.Resources,
		}
		if containerType == podutil.InitContainers {
			spec.InitContainers = append(spec.InitContainers, actuated)
		} else {
			spec.Containers = append(spec.Containers, actuated)
		}
	}
	pod.Spec = spec
}

// computeVolumeResizeAction determines the actions required (if any) to resize volumes.
func (m *kubeGenericRuntimeManager) computeVolumeResizeAction(ctx context.Context, pod *v1.Pod, changes *podActions) {
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
		logger := klog.FromContext(ctx)
		for _, vol := range pod.Spec.Volumes {
			if !allocation.VolHasMemoryBackedEmptyDir(&vol) {
				continue
			}
			actuatedSize, found := m.actuatedState.GetEmptyDirVolumeLimit(pod.UID, vol.Name)
			if !found {
				// If the volume is not in the actuated state, it is not mounted yet,
				// so there is nothing to resize.
				continue
			}
			desiredSize, err := m.runtimeHelper.EffectiveEphemeralVolumeLimit(pod, vol.Name)

			if err != nil || desiredSize == nil {
				// This should never happen because EffectiveEphemeralVolumeLimit defaults
				// to node allocatable memory when no pod or volume limit is specified.
				logger.Error(err, "Failed to calculate desired empty dir volume limit", "pod", klog.KObj(pod), "volume", vol.Name)
				continue
			}

			if actuatedSize == nil {
				logger.V(2).Info("Failed to read actuated empty dir volume limit", "pod", klog.KObj(pod), "volume", vol.Name)
				// If the actuation checkpoint is missing, force a sync to apply
				// the limit and write the checkpoint. We treat it as an upsize
				// (safer order) to ensure cgroup limits are expanded first.
				changes.VolumesToUpsize = append(changes.VolumesToUpsize, vol)
				continue
			}

			cmp := desiredSize.Cmp(*actuatedSize)
			if cmp > 0 {
				changes.VolumesToUpsize = append(changes.VolumesToUpsize, vol)
			} else if cmp < 0 {
				changes.VolumesToDownsize = append(changes.VolumesToDownsize, vol)
			}
		}
	}
}

func (m *kubeGenericRuntimeManager) doPodResizeAction(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, podContainerChanges podActions) *kubecontainer.SyncResult {
	logger := klog.FromContext(ctx)
	start := time.Now()
	success := false
	containerResized := len(podContainerChanges.ContainersToUpdate) > 0
	podResized := len(podContainerChanges.PodCgroupResourcesToUpdate) > 0
	defer func() {
		metrics.PodResizeDurationMilliseconds.WithLabelValues(
			strconv.FormatBool(success),
			strconv.FormatBool(containerResized),
			strconv.FormatBool(podResized),
		).Observe(float64(time.Since(start).Milliseconds()))
	}()

	resizeResult := kubecontainer.NewSyncResult(kubecontainer.ResizePodInPlace, format.Pod(pod))
	pcm := m.containerManager.NewPodContainerManager()
	enforceCPULimits := m.cpuCFSQuota
	if utilfeature.DefaultFeatureGate.Enabled(features.DisableCPUQuotaWithExclusiveCPUs) && m.containerManager.PodHasExclusiveCPUs(logger, pod) {
		enforceCPULimits = false
		logger.V(2).Info("Disabled CFS quota", "pod", klog.KObj(pod))
	}
	podResources := cm.ResourceConfigForPod(pod, enforceCPULimits, uint64((m.cpuCFSQuotaPeriod.Duration)/time.Microsecond), m.isMemoryQoSEnforced(), m.memoryReservationPolicy)
	if podResources == nil {
		logger.Error(nil, "Unable to get resource configuration", "pod", klog.KObj(pod))
		resizeResult.Fail(kubecontainer.ErrResizePodInPlace, fmt.Sprintf("unable to get resource configuration processing resize for pod %q", format.Pod(pod)))
		return resizeResult
	}
	// A pod has no effective limit for a resource if any container lacks one. Since resizes can't
	// remove container limits, an effective limit only goes away when a container without limits is
	// added, which requires DynamicContainers.
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicContainers) {
		if podResources.Memory == nil {
			podResources.Memory = new(unlimitedCgroupLimit)
		}
		if podResources.CPUQuota == nil {
			podResources.CPUQuota = new(unlimitedCgroupLimit)
		}
	}
	if m.isMemoryQoSEnforced() {
		m.applyPodLevelMemoryHigh(pod, podResources)
	}
	currentPodMemoryConfig, err := pcm.GetPodCgroupConfig(pod, v1.ResourceMemory)
	if err != nil {
		logger.Error(err, "Unable to get pod cgroup memory config", "pod", klog.KObj(pod))
		resizeResult.Fail(kubecontainer.ErrResizePodInPlace, fmt.Sprintf("unable to get pod cgroup memory config for pod %q", format.Pod(pod)))
		return resizeResult
	}
	currentPodCPUConfig, err := pcm.GetPodCgroupConfig(pod, v1.ResourceCPU)
	if err != nil {
		logger.Error(err, "Unable to get pod cgroup cpu config", "pod", klog.KObj(pod))
		resizeResult.Fail(kubecontainer.ErrResizePodInPlace, fmt.Sprintf("unable to get pod cgroup cpu config for pod %q", format.Pod(pod)))
		return resizeResult
	}

	// KEP-5359: swap is actuated in its own pass. Read the pod cgroup's current
	// memory.swap.max so pod-level increases/decreases can be ordered against
	// the container writes. swapLimitsEnabled is false when the node cannot swap, in
	// which case effectiveSwapLimit already yields no container deltas.
	swapLimitsEnabled := m.workloadSwapLimitsEnabled()
	var currentPodSwapLimit, desiredPodSwapLimit int64
	var currentPodSwapSet, desiredPodSwapSet bool
	if swapLimitsEnabled {
		currentPodSwapConfig, err := pcm.GetPodCgroupConfig(pod, v1.ResourceSwap)
		if err != nil {
			logger.Error(err, "Unable to get pod cgroup swap config", "pod", klog.KObj(pod))
			resizeResult.Fail(kubecontainer.ErrResizePodInPlace, fmt.Sprintf("unable to get pod cgroup swap config for pod %q", format.Pod(pod)))
			return resizeResult
		}
		currentPodSwapLimit, currentPodSwapSet = cm.SwapLimitFromConfig(currentPodSwapConfig)
		desiredPodSwapLimit, desiredPodSwapSet = cm.SwapLimitFromConfig(podResources)
	} else {
		podResources.Unified = cm.UnifiedWithoutSwap(podResources.Unified)
	}
	// "unset" is +inf, so adding a pod-level limit is a decrease. API validation
	// rejects removing a pod-level limit, so there is no unset-after-set case.
	podSwapResize := swapLimitsEnabled && podContainerChanges.PodCgroupResourcesToUpdate.Has(v1.ResourceSwap)
	podSwapIncreasing := podSwapResize && currentPodSwapSet && desiredPodSwapSet && desiredPodSwapLimit > currentPodSwapLimit
	podSwapDecreasing := podSwapResize && desiredPodSwapSet && (!currentPodSwapSet || desiredPodSwapLimit < currentPodSwapLimit)

	currentPodResources := podResources
	currentPodResources = mergeResourceConfig(currentPodResources, currentPodMemoryConfig)
	currentPodResources = mergeResourceConfig(currentPodResources, currentPodCPUConfig)
	if swapLimitsEnabled {
		// podResources carries the *desired* memory.swap.max in Unified; the
		// "current" view must reflect the cgroup. UnifiedWithoutSwap copies the
		// map so podResources.Unified is not mutated through the shared reference.
		currentPodResources.Unified = cm.UnifiedWithoutSwap(currentPodResources.Unified)
		if currentPodSwapSet {
			cm.SetSwapLimit(currentPodResources, &currentPodSwapLimit)
		} else {
			cm.SetSwapLimit(currentPodResources, nil)
		}
	}

	// Before proceeding with the resize, perform a best-effort check to catch potential resize
	// errors in order to avoid a partial-resize state.
	if err := m.validatePodResizeAction(ctx, pod, podStatus, currentPodResources, podResources, podContainerChanges); err != nil {
		logger.Error(err, "Allocated pod resize is not currently feasible", "pod", klog.KObj(pod))
		resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
		return resizeResult
	}

	effectivePodResources := cm.EffectivePodResources(pod)
	actuatedPodResources, found := m.actuatedState.GetPodLevelResources(pod.UID)
	if !found {
		logger.Error(nil, "Missing actuated resource record", "pod", klog.KObj(pod))
	}
	if actuatedPodResources == nil {
		// Proceed with the zero-value actuated resources.
		actuatedPodResources = &v1.ResourceRequirements{}
	}
	// updateActuatedPodLevelResources records the effective pod-level requests and/or limits of
	// resourceName as actuated.
	updateActuatedPodLevelResources := func(resourceName v1.ResourceName, requests, limits bool) error {
		if requests {
			actuatedPodResources.Requests = setResourceQuantity(actuatedPodResources.Requests, effectivePodResources.Requests, resourceName)
		}
		if limits {
			effectiveLimits := effectivePodResources.Limits
			if resourceName == v1.ResourceSwap && pod.Spec.Resources != nil {
				// The pod-level swap limit is only ever explicit.
				effectiveLimits = pod.Spec.Resources.Limits
			}
			actuatedPodResources.Limits = setResourceQuantity(actuatedPodResources.Limits, effectiveLimits, resourceName)
		}
		if err := m.actuatedState.SetPodLevelResources(logger, pod.UID, actuatedPodResources); err != nil {
			logger.Error(err, "SetPodLevelResources failed", "pod", klog.KObj(pod), "resourceName", resourceName)
			return err
		}
		return nil
	}

	setPodCgroupConfig := func(logger klog.Logger, rName v1.ResourceName, setLimitValue bool) error {
		var err error
		resizedResources := &cm.ResourceConfig{}
		switch rName {
		case v1.ResourceCPU:
			if setLimitValue {
				resizedResources.CPUPeriod = podResources.CPUPeriod
				resizedResources.CPUQuota = podResources.CPUQuota
			} else {
				resizedResources.CPUShares = podResources.CPUShares
			}
		case v1.ResourceMemory:
			// memory.high/min/low travel with the memory pass; memory.swap.max has
			// its own ordered pass (2a/2c below) and must not be written here.
			memoryUnified := cm.UnifiedWithoutSwap(podResources.Unified)
			if setLimitValue {
				resizedResources.Memory = podResources.Memory
			} else if len(memoryUnified) == 0 {
				// Memory requests are only written to cgroups by MemoryQoS.
				return nil
			}
			resizedResources.Unified = memoryUnified
		case v1.ResourceSwap:
			// Only memory.swap.max. Called only when a pod-level limit is desired.
			cm.SetSwapLimit(resizedResources, &desiredPodSwapLimit)
		}

		// Notify the runtime first. If this fails, the runtime has rejected the resize.
		mergedPodResources := mergeResourceConfig(currentPodResources, resizedResources)
		if err = m.updatePodSandboxResources(ctx, podContainerChanges.SandboxID, pod, mergedPodResources); err != nil {
			return fmt.Errorf("failed to notify runtime for UpdatePodSandboxResources (resource=%s); resize rejected: %w", rName, err)
		}

		// Actuate the pod-level cgroup change.
		err = pcm.SetPodCgroupConfig(logger, pod, resizedResources)
		if err != nil {
			logger.Error(err, "Failed to set cgroup config", "resource", rName, "pod", klog.KObj(pod))
			// If cgroup adjustment fails, notify the runtime to revert to the previous resources.
			if rollbackErr := m.updatePodSandboxResources(ctx, podContainerChanges.SandboxID, pod, currentPodResources); rollbackErr != nil {
				logger.Error(rollbackErr, "Failed to notify runtime to rollback UpdatePodSandboxResources", "resource", rName, "pod", klog.KObj(pod))
			}
			return err
		}

		// Update our tracking of the current state.
		currentPodResources = mergedPodResources
		return updateActuatedPodLevelResources(rName, !setLimitValue, setLimitValue)
	}

	// Memory, CPU and volume sizes are updated in a coordinated order:
	// 1. Volume downsize (decreases tmpfs utilization first, freeing page cache).
	// 2. Memory limits and requests (resizes memory cgroup configurations).
	// 3. Volume upsize (increases tmpfs size limits safely after cgroup memory capacity is expanded).
	// 4. CPU limits and requests (resizes cpu cgroup configurations)
	// Memory and CPU are updated separately because memory resizes may be ordered differently than CPU resizes.
	// If resize results in net pod resource increase, set pod cgroup config before resizing containers.
	// If resize results in net pod resource decrease, set pod cgroup config after resizing containers.
	// If an error occurs at any point, abort. Let future syncpod iterations retry the unfinished stuff.
	updateVolumeSize := func(volumes []v1.Volume) error {
		if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
			for _, vol := range volumes {
				if err := m.runtimeHelper.ResizeEphemeralVolume(pod, vol.Name); err != nil {
					return fmt.Errorf("failed to resize volume %s: %w", vol.Name, err)
				}
				effectiveLimit, err := m.runtimeHelper.EffectiveEphemeralVolumeLimit(pod, vol.Name)
				if err != nil {
					return fmt.Errorf("failed to calculate effective emptyDir volume limit for %s: %w", vol.Name, err)
				}
				actuatedEmptyDir := &v1.EmptyDirVolumeSource{
					Medium:    vol.EmptyDir.Medium,
					SizeLimit: effectiveLimit,
				}
				if err := m.actuatedState.SetEmptyDirVolumeLimit(logger, pod.UID, vol.Name, actuatedEmptyDir); err != nil {
					return fmt.Errorf("failed to update actuated emptyDir volume limit checkpoint for %s: %w", vol.Name, err)
				}
			}
		}
		return nil
	}

	resizeContainers := func(rName v1.ResourceName, currPodCgLimValue, newPodCgLimValue, currPodCgReqValue, newPodCgReqValue int64) error {
		var err error
		currPodCgLimValue, newPodCgLimValue = cgroupLimit(currPodCgLimValue), cgroupLimit(newPodCgLimValue)
		// At upsizing, limits should expand prior to requests in order to keep "requests <= limits".
		if newPodCgLimValue > currPodCgLimValue {
			if err = setPodCgroupConfig(logger, rName, true); err != nil {
				return err
			}
		}
		if newPodCgReqValue > currPodCgReqValue {
			if err = setPodCgroupConfig(logger, rName, false); err != nil {
				return err
			}
		}
		if len(podContainerChanges.ContainersToUpdate[rName]) > 0 {
			if err = m.updatePodContainerResources(ctx, pod, rName, podContainerChanges.ContainersToUpdate[rName]); err != nil {
				logger.Error(err, "updatePodContainerResources failed", "pod", format.Pod(pod), "resource", rName)
				return err
			}
		}

		// At downsizing, requests should shrink prior to limits in order to keep "requests <= limits".
		if newPodCgReqValue < currPodCgReqValue {
			if err = setPodCgroupConfig(logger, rName, false); err != nil {
				return err
			}
		}
		if newPodCgLimValue < currPodCgLimValue {
			if err = setPodCgroupConfig(logger, rName, true); err != nil {
				return err
			}
		}

		// Record both sides, including those that needed no cgroup write.
		if err = updateActuatedPodLevelResources(rName, true, true); err != nil {
			logger.Error(err, "Failed to update pod-level actuated resources", "resource", rName, "pod", klog.KObj(pod))
		}
		return err
	}

	// Always update the pod status once. Even if there was a resize error, the resize may have been
	// partially actuated.
	defer m.runtimeHelper.RequestPodReinspect(pod.UID)

	// 1. Volume Downsize
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
		if err := updateVolumeSize(podContainerChanges.VolumesToDownsize); err != nil {
			resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
			return resizeResult
		}
	}

	// 2. Memory family: swap and memory are independent cgroup v2 limits, but
	// reclaim triggered by a tighter memory.max (or memory.high under MemoryQoS)
	// needs swap headroom to make progress. Every intermediate state is kept at
	// least as permissive as both endpoints:
	//   2a. swap increases   (pod cgroup up, then containers)
	//   2b. memory           (existing: pod up -> containers -> pod down)
	//   2c. swap decreases   (containers, then pod cgroup down)
	// If a step fails, SyncPod retries from the top; a partially applied state
	// (more swap than the final value) is safe.
	var swapIncreases, swapDecreases []containerToUpdateInfo
	for _, cUpdate := range podContainerChanges.ContainersToUpdate[v1.ResourceSwap] {
		if cUpdate.desiredContainerResources.swapLimit > cUpdate.currentContainerResources.swapLimit {
			swapIncreases = append(swapIncreases, cUpdate)
		} else {
			swapDecreases = append(swapDecreases, cUpdate)
		}
	}
	// 2a. Swap increases
	if swapLimitsEnabled && (podSwapIncreasing || len(swapIncreases) > 0) {
		if podSwapIncreasing {
			if err := setPodCgroupConfig(logger, v1.ResourceSwap, true); err != nil {
				resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
				return resizeResult
			}
		}
		if len(swapIncreases) > 0 {
			if err := m.updatePodContainerResources(ctx, pod, v1.ResourceSwap, swapIncreases); err != nil {
				logger.Error(err, "updatePodContainerResources failed", "pod", format.Pod(pod), "resource", v1.ResourceSwap)
				resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
				return resizeResult
			}
		}
	}

	// 2b. Memory cgroups
	if len(podContainerChanges.ContainersToUpdate[v1.ResourceMemory]) > 0 || podContainerChanges.PodCgroupResourcesToUpdate.Has(v1.ResourceMemory) {
		if podResources.Memory == nil {
			// Default pod memory limit to the current memory limit if unset to prevent it from updating.
			podResources.Memory = currentPodMemoryConfig.Memory
		}
		if errResize := resizeContainers(v1.ResourceMemory, int64(*currentPodMemoryConfig.Memory), *podResources.Memory,
			actuatedPodResources.Requests.Memory().Value(), effectivePodResources.Requests.Memory().Value()); errResize != nil {
			resizeResult.Fail(kubecontainer.ErrResizePodInPlace, errResize.Error())
			return resizeResult
		}
	}

	// 2c. Swap decreases
	if swapLimitsEnabled && (podSwapDecreasing || len(swapDecreases) > 0) {
		if len(swapDecreases) > 0 {
			if err := m.updatePodContainerResources(ctx, pod, v1.ResourceSwap, swapDecreases); err != nil {
				logger.Error(err, "updatePodContainerResources failed", "pod", format.Pod(pod), "resource", v1.ResourceSwap)
				resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
				return resizeResult
			}
		}
		if podSwapDecreasing {
			if err := setPodCgroupConfig(logger, v1.ResourceSwap, true); err != nil {
				resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
				return resizeResult
			}
		}
	}
	if podSwapResize {
		// Record the limit, including when it needed no cgroup write.
		if err := updateActuatedPodLevelResources(v1.ResourceSwap, false, true); err != nil {
			logger.Error(err, "Failed to update pod-level actuated resources", "resource", v1.ResourceSwap, "pod", klog.KObj(pod))
		}
	}

	// 3. Volume upsize
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
		if err := updateVolumeSize(podContainerChanges.VolumesToUpsize); err != nil {
			resizeResult.Fail(kubecontainer.ErrResizePodInPlace, err.Error())
			return resizeResult
		}
	}

	// 4. CPU cgroups
	if len(podContainerChanges.ContainersToUpdate[v1.ResourceCPU]) > 0 || podContainerChanges.PodCgroupResourcesToUpdate.Has(v1.ResourceCPU) {
		if podResources.CPUShares == nil {
			// This shouldn't happen: ResourceConfigForPod always returns a non-nil value for CPUShares.
			logger.Error(nil, "podResources.CPUShares is nil", "pod", pod.Name)
			resizeResult.Fail(kubecontainer.ErrResizePodInPlace, fmt.Sprintf("podResources.CPUShares is nil for pod %s", pod.Name))
			return resizeResult
		}

		// Default pod CPUQuota to the current CPUQuota if no limit is set to prevent the pod limit
		// from updating.
		if podResources.CPUQuota == nil {
			podResources.CPUQuota = currentPodCPUConfig.CPUQuota
		}
		if errResize := resizeContainers(v1.ResourceCPU, *currentPodCPUConfig.CPUQuota, *podResources.CPUQuota,
			int64(*currentPodCPUConfig.CPUShares), int64(*podResources.CPUShares)); errResize != nil {
			resizeResult.Fail(kubecontainer.ErrResizePodInPlace, errResize.Error())
			return resizeResult
		}
	}

	success = true
	return resizeResult
}

// validatePodResizeAction checks whether the proposed resize actions are currently viable.
func (m *kubeGenericRuntimeManager) validatePodResizeAction(
	ctx context.Context,
	pod *v1.Pod,
	podStatus *kubecontainer.PodStatus,
	currentPodResources, desiredPodResources *cm.ResourceConfig,
	podContainerChanges podActions,
) error {
	memoryChanging := len(podContainerChanges.ContainersToUpdate[v1.ResourceMemory]) > 0 || podContainerChanges.PodCgroupResourcesToUpdate.Has(v1.ResourceMemory)
	// Swap limits only change where the kubelet honors workload swap limits.
	swapChanging := m.workloadSwapLimitsEnabled() &&
		(len(podContainerChanges.ContainersToUpdate[v1.ResourceSwap]) > 0 || podContainerChanges.PodCgroupResourcesToUpdate.Has(v1.ResourceSwap))
	if memoryChanging || swapChanging {
		return m.validateMemoryResizeAction(ctx, pod, podStatus, currentPodResources, desiredPodResources, podContainerChanges)
	}

	return nil
}

// extraNodeAllocatableMemoryLimit returns the memory that getMemoryLimit adds to the container's
// memory limit for its extra node allocatable resources.
func extraNodeAllocatableMemoryLimit(pod *v1.Pod, container *v1.Container) int64 {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRANodeAllocatableResources) || container.Resources.Limits.Memory().IsZero() {
		return 0
	}
	limits := resourcehelper.GetExtraNodeAllocatableResourcesLimits(pod, container.Name)
	return limits.Memory().Value()
}

// validateMemoryResizeAction defers a resize while any memory-family limit
// would be set at or below current usage. Memory: the kernel would reclaim and
// possibly OOM-kill. Swap (KEP-5359): the kernel accepts memory.swap.max below
// memory.swap.current without reclaim, but the workload would then be one
// memory.max touch away from OOM with no swap-out possible, and status would
// report a limit the cgroup is already over. The swap check applies equally
// to user-declared and LimitedSwap-recalculated limits.
func (m *kubeGenericRuntimeManager) validateMemoryResizeAction(
	ctx context.Context,
	pod *v1.Pod,
	podStatus *kubecontainer.PodStatus,
	currentPodResources, desiredPodResources *cm.ResourceConfig,
	podContainerChanges podActions,
) error {
	// Determine which memory limits are decreasing.
	podLimitDecreasing := desiredPodResources.Memory != nil &&
		(currentPodResources.Memory == nil || // Pod memory limit added
			cgroupLimit(*desiredPodResources.Memory) < cgroupLimit(*currentPodResources.Memory)) // Pod memory limit decreasing

	decreasingContainerLimits := map[string]int64{} // Map of container name to desired memory limit.
	for _, cUpdate := range podContainerChanges.ContainersToUpdate[v1.ResourceMemory] {
		if cUpdate.desiredContainerResources.memoryLimit != 0 {
			if cUpdate.currentContainerResources == nil || cUpdate.currentContainerResources.memoryLimit == 0 || // Limit added
				cUpdate.desiredContainerResources.memoryLimit < cUpdate.currentContainerResources.memoryLimit { // Limit decreasing
				decreasingContainerLimits[cUpdate.container.Name] = cUpdate.desiredContainerResources.memoryLimit + extraNodeAllocatableMemoryLimit(pod, cUpdate.container)
			}
		}
	}

	// Determine which swap limits are decreasing. "unset" (max) counts as +inf,
	// so adding a pod-level swap limit is a decrease.
	swapLimitsEnabled := m.workloadSwapLimitsEnabled()
	podSwapDecreasing := false
	var desiredPodSwap int64
	if swapLimitsEnabled && podContainerChanges.PodCgroupResourcesToUpdate.Has(v1.ResourceSwap) {
		currentSwap, currentSet := cm.SwapLimitFromConfig(currentPodResources)
		desiredSwap, desiredSet := cm.SwapLimitFromConfig(desiredPodResources)
		if desiredSet && (!currentSet || desiredSwap < currentSwap) {
			podSwapDecreasing = true
			desiredPodSwap = desiredSwap
		}
	}
	decreasingContainerSwap := map[string]int64{} // Map of container name to desired swap limit.
	for _, cUpdate := range podContainerChanges.ContainersToUpdate[v1.ResourceSwap] {
		if cUpdate.currentContainerResources == nil || cUpdate.desiredContainerResources.swapLimit < cUpdate.currentContainerResources.swapLimit {
			decreasingContainerSwap[cUpdate.container.Name] = cUpdate.desiredContainerResources.swapLimit
		}
	}

	if !podLimitDecreasing && len(decreasingContainerLimits) == 0 && !podSwapDecreasing && len(decreasingContainerSwap) == 0 {
		// No memory or swap limits are decreasing: nothing else to check here.
		return nil
	}

	// Check whether any of the new memory limits are below current memory usage.
	podUsageStats, err := m.runtimeHelper.PodCPUAndMemoryStats(ctx, pod, podStatus)
	if err != nil {
		return fmt.Errorf("unable to read memory usage for pod %q", format.Pod(pod))
	}
	if missingSwapUsage(podUsageStats, podSwapDecreasing, decreasingContainerSwap) {
		// Some runtimes leave swap out of CRI stats. Only pay for the extra
		// stats read when a swap limit is actually being lowered.
		if swapStats, err := m.runtimeHelper.PodSwapStats(ctx, pod, podStatus); err == nil {
			mergeSwapUsage(podUsageStats, swapStats)
		}
	}

	var errs []error
	if podLimitDecreasing {
		if podUsageStats.Memory == nil || podUsageStats.Memory.UsageBytes == nil {
			errs = append(errs, fmt.Errorf("missing pod memory usage"))
		} else if *podUsageStats.Memory.UsageBytes >= uint64(*desiredPodResources.Memory) {
			errs = append(errs, fmt.Errorf("attempting to set pod memory limit (%d) below current usage (%d)",
				*desiredPodResources.Memory, *podUsageStats.Memory.UsageBytes))
		}
	}
	// Swap usage 0 under a limit of 0 is fine (no swap in use), hence ">" not ">=".
	if podSwapDecreasing {
		if podUsageStats.Swap == nil || podUsageStats.Swap.SwapUsageBytes == nil {
			errs = append(errs, fmt.Errorf("missing pod swap usage"))
		} else if *podUsageStats.Swap.SwapUsageBytes > uint64(desiredPodSwap) {
			errs = append(errs, fmt.Errorf("attempting to set pod swap limit (%d) below current swap usage (%d)",
				desiredPodSwap, *podUsageStats.Swap.SwapUsageBytes))
		}
	}
	for _, cStats := range podUsageStats.Containers {
		if desiredLimit, ok := decreasingContainerLimits[cStats.Name]; ok {
			if cStats.Memory == nil || cStats.Memory.UsageBytes == nil {
				errs = append(errs, fmt.Errorf("missing container %q memory usage", cStats.Name))
			} else if *cStats.Memory.UsageBytes >= uint64(desiredLimit) {
				errs = append(errs, fmt.Errorf("attempting to set container %q memory limit (%d) below current usage (%d)",
					cStats.Name, desiredLimit, *cStats.Memory.UsageBytes))
			}
		}
		if desiredSwap, ok := decreasingContainerSwap[cStats.Name]; ok {
			if cStats.Swap == nil || cStats.Swap.SwapUsageBytes == nil {
				errs = append(errs, fmt.Errorf("missing container %q swap usage", cStats.Name))
			} else if *cStats.Swap.SwapUsageBytes > uint64(desiredSwap) {
				errs = append(errs, fmt.Errorf("attempting to set container %q swap limit (%d) below current swap usage (%d)",
					cStats.Name, desiredSwap, *cStats.Swap.SwapUsageBytes))
			}
		}
	}

	if len(errs) > 0 {
		agg := utilerrors.NewAggregate(errs)
		if podSwapDecreasing || len(decreasingContainerSwap) > 0 {
			return fmt.Errorf("cannot decrease memory or swap limits: %w", agg)
		}
		return fmt.Errorf("cannot decrease memory limits: %w", agg)
	}

	return nil
}

// missingSwapUsage reports whether a swap limit is being lowered for the pod or
// a container whose swap usage is absent from stats.
func missingSwapUsage(stats *statsapi.PodStats, podSwapDecreasing bool, decreasingContainerSwap map[string]int64) bool {
	if podSwapDecreasing && (stats.Swap == nil || stats.Swap.SwapUsageBytes == nil) {
		return true
	}
	if len(decreasingContainerSwap) == 0 {
		return false
	}
	found := map[string]bool{}
	for _, cs := range stats.Containers {
		if cs.Swap != nil && cs.Swap.SwapUsageBytes != nil {
			found[cs.Name] = true
		}
	}
	for name := range decreasingContainerSwap {
		if !found[name] {
			return true
		}
	}
	return false
}

// mergeSwapUsage fills swap usage missing from stats with the values in swapStats.
func mergeSwapUsage(stats, swapStats *statsapi.PodStats) {
	if swapStats == nil {
		return
	}
	if stats.Swap == nil || stats.Swap.SwapUsageBytes == nil {
		stats.Swap = swapStats.Swap
	}
	for _, fb := range swapStats.Containers {
		for i := range stats.Containers {
			if stats.Containers[i].Name != fb.Name {
				continue
			}
			if stats.Containers[i].Swap == nil || stats.Containers[i].Swap.SwapUsageBytes == nil {
				stats.Containers[i].Swap = fb.Swap
			}
			break
		}
	}
}

func (m *kubeGenericRuntimeManager) updatePodContainerResources(ctx context.Context, pod *v1.Pod, resourceName v1.ResourceName, containersToUpdate []containerToUpdateInfo) error {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Updating container resources", "pod", klog.KObj(pod))

	for _, cInfo := range containersToUpdate {
		container := cInfo.container.DeepCopy()
		// Each UpdateContainerResources call carries the full LinuxContainerResources,
		// so the synthetic container holds the most recently configured value for
		// every resource except the one being updated.
		switch resourceName {
		case v1.ResourceMemory:
			container.Resources.Limits = v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(cInfo.currentContainerResources.cpuLimit, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(cInfo.desiredContainerResources.memoryLimit, resource.BinarySI),
			}
			container.Resources.Requests = v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(cInfo.currentContainerResources.cpuRequest, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(cInfo.desiredContainerResources.memoryRequest, resource.BinarySI),
			}
		case v1.ResourceCPU:
			container.Resources.Limits = v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(cInfo.desiredContainerResources.cpuLimit, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(cInfo.currentContainerResources.memoryLimit, resource.BinarySI),
			}
			container.Resources.Requests = v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(cInfo.desiredContainerResources.cpuRequest, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(cInfo.currentContainerResources.memoryRequest, resource.BinarySI),
			}
		case v1.ResourceSwap:
			container.Resources.Limits = v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(cInfo.currentContainerResources.cpuLimit, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(cInfo.currentContainerResources.memoryLimit, resource.BinarySI),
			}
			container.Resources.Requests = v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(cInfo.currentContainerResources.cpuRequest, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(cInfo.currentContainerResources.memoryRequest, resource.BinarySI),
			}
		}
		actuated := container
		if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadControlledSwap) {
			// KEP-5359: pin the swap limit explicitly. generateLinuxContainerResources
			// re-derives memory.swap.max from the container it is handed; without
			// this a cpu/memory update would reset swap to the mode default (0 under
			// WorkloadControlledSwap) or recompute LimitedSwap from the synthetic
			// memory request. The explicit value takes precedence in
			// effectiveSwapLimit, so the number written is exactly the one chosen in
			// computePodResizeAction.
			swapLimit := cInfo.currentContainerResources.swapLimit
			if resourceName == v1.ResourceSwap {
				swapLimit = cInfo.desiredContainerResources.swapLimit
			}
			container.Resources.Limits[v1.ResourceSwap] = *resource.NewQuantity(swapLimit, resource.BinarySI)
			// The pin is only for the CRI request. The actuated checkpoint records
			// limits.swap only for a container that declares it, and only once the
			// swap pass has applied it. A container that inherits its swap limit
			// (pod-level, LimitedSwap, or the node default) must not get one, or it
			// would stop following later pod-level or LimitedSwap changes.
			actuated = container.DeepCopy()
			delete(actuated.Resources.Limits, v1.ResourceSwap)
			if resourceName == v1.ResourceSwap {
				if _, declared := cInfo.container.Resources.Limits[v1.ResourceSwap]; declared {
					actuated.Resources.Limits[v1.ResourceSwap] = container.Resources.Limits[v1.ResourceSwap]
				}
			} else if previous, found := m.actuatedState.GetContainerResources(pod.UID, container.Name); found {
				// cpu and memory passes leave the recorded swap limit as it was.
				if q, recorded := previous.Limits[v1.ResourceSwap]; recorded {
					actuated.Resources.Limits[v1.ResourceSwap] = q
				}
			}
		}
		if err := m.updateContainerResourcesAndRecord(ctx, pod, container, actuated, cInfo.kubeContainerID, cInfo.containerType); err != nil {
			// Log error and abort as container updates need to succeed in the order determined by computePodResizeAction.
			// The recovery path is for SyncPod to keep retrying at later times until it succeeds.
			logger.Error(err, "updateContainerResources failed", "container", container.Name, "cID", cInfo.kubeContainerID,
				"pod", format.Pod(pod), "resourceName", resourceName)
			return err
		}

		// If UpdateContainerResources is error-free, it means desired values for 'resourceName' was accepted by runtime.
		// So we update currentContainerResources for 'resourceName', which is our view of most recently configured resources.
		// Note: We can't rely on GetPodStatus as runtime may lag in actuating the resource values it just accepted.
		switch resourceName {
		case v1.ResourceMemory:
			cInfo.currentContainerResources.memoryLimit = cInfo.desiredContainerResources.memoryLimit
			cInfo.currentContainerResources.memoryRequest = cInfo.desiredContainerResources.memoryRequest
		case v1.ResourceCPU:
			cInfo.currentContainerResources.cpuLimit = cInfo.desiredContainerResources.cpuLimit
			cInfo.currentContainerResources.cpuRequest = cInfo.desiredContainerResources.cpuRequest
		case v1.ResourceSwap:
			cInfo.currentContainerResources.swapLimit = cInfo.desiredContainerResources.swapLimit
			m.warnSwapLimitAbovePodLimit(ctx, pod, cInfo.container)
		}
	}
	return nil
}

func findContainerInPodSpec(spec *v1.PodSpec, name string, containerType podutil.ContainerType) *v1.Container {
	for c := range podutil.ContainerIter(spec, containerType) {
		if c.Name == name {
			return c
		}
	}
	return nil
}

func hasSubPathVolumeMount(mounts []v1.VolumeMount) bool {
	return slices.ContainsFunc(mounts, func(vm v1.VolumeMount) bool {
		return vm.SubPath != "" || vm.SubPathExpr != ""
	})
}

func isVolumeMountedInPodSpec(spec *v1.PodSpec, volumeName string) bool {
	for c := range podutil.ContainerIter(spec, podutil.AllContainers) {
		if slices.ContainsFunc(c.VolumeMounts, func(vm v1.VolumeMount) bool { return vm.Name == volumeName }) {
			return true
		}
	}
	return false
}

// containerVolumesReady reports whether every volume mounted by container is ready on the node.
func containerVolumesReady(pod *v1.Pod, container *v1.Container, actuatedPod *v1.Pod) (string, bool) {
	if actuatedPod == nil || !utilfeature.DefaultFeatureGate.Enabled(features.DynamicNodeLocalEphemeralVolumes) {
		return "", true
	}
	for _, vm := range container.VolumeMounts {
		volIdx := slices.IndexFunc(pod.Spec.Volumes, func(v v1.Volume) bool { return v.Name == vm.Name })
		if volIdx < 0 {
			return vm.Name, false
		}
		// Non-dynamic volume sources are never deferred by WaitForAttachAndMount, and PVC-backed
		// volumes have nil volumeSpec.Volume so they are not recorded in actuatedPod.Spec.Volumes.
		// Image volumes pull their OCI image in startContainer after WaitForAttachAndMount ensures
		// their plugin entry exists in actualStateOfWorld.
		if !podutil.IsSupportedDynamicVolumeSource(&pod.Spec.Volumes[volIdx].VolumeSource) || pod.Spec.Volumes[volIdx].Image != nil {
			continue
		}
		if isVolumeMountedInPodSpec(&actuatedPod.Spec, vm.Name) ||
			slices.ContainsFunc(actuatedPod.Spec.Volumes, func(v v1.Volume) bool { return v.Name == vm.Name }) {
			continue
		}
		return vm.Name, false
	}
	return "", true
}

// reconcileNonRunningActuatedContainers clears actuated container VolumeMounts and VolumeDevices
// for removed containers once they are no longer running, so dynamically removed volumes can
// unmount from the host and removed containers' terminated statuses drop stale VolumeMounts.
func (m *kubeGenericRuntimeManager) reconcileNonRunningActuatedContainers(logger klog.Logger, pod *v1.Pod, podStatus *kubecontainer.PodStatus, actuatedPod *v1.Pod) *v1.Pod {
	if actuatedPod == nil || podStatus == nil {
		return actuatedPod
	}
	updated := false
	for actCtr, cType := range podutil.ContainerIter(&actuatedPod.Spec, podutil.AllContainers) {
		if len(actCtr.VolumeMounts) == 0 && len(actCtr.VolumeDevices) == 0 {
			continue
		}
		if podStatus.HasRunningOrUnknownContainer(actCtr.Name) {
			continue
		}
		specCtr := findContainerInPodSpec(&pod.Spec, actCtr.Name, cType)
		if specCtr != nil {
			continue
		}
		if hasSubPathVolumeMount(actCtr.VolumeMounts) && m.runtimeHelper != nil {
			if err := m.runtimeHelper.CleanContainerSubPaths(pod.UID, actCtr.Name); err != nil {
				logger.Error(err, "Failed to clean container subPath mounts for stopped container", "pod", klog.KObj(pod), "containerName", actCtr.Name)
				continue
			}
		}
		cleared := *actCtr
		cleared.VolumeMounts = nil
		cleared.VolumeDevices = nil
		if err := m.actuatedState.SetContainer(logger, pod.UID, cType, cleared); err != nil {
			logger.Error(err, "Failed to clear actuated volume mounts for stopped container", "pod", klog.KObj(pod), "containerName", actCtr.Name)
		} else {
			updated = true
		}
	}
	if updated {
		if refreshed, ok := m.actuatedState.GetPodReadOnly(pod.UID); ok {
			return refreshed
		}
	}
	return actuatedPod
}

// computePodActions checks whether the pod spec has changed and returns the changes if true.
func (m *kubeGenericRuntimeManager) computePodActions(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, restartAllContainers bool) podActions {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Syncing Pod", "pod", klog.KObj(pod))

	createPodSandbox, attempt, sandboxID, _ := runtimeutil.PodSandboxChanged(logger, pod, podStatus)
	changes := podActions{
		KillPod:           createPodSandbox,
		CreateSandbox:     createPodSandbox,
		SandboxID:         sandboxID,
		Attempt:           attempt,
		ContainersToStart: []int{},
		ContainersToKill:  make(map[kubecontainer.ContainerID]containerToKillInfo),
	}

	// Needs to kill and remove all containers in reverse order when the pod is marked for RestartAllContainers.
	if utilfeature.DefaultFeatureGate.Enabled(features.RestartAllContainersOnContainerExits) && restartAllContainers {
		logger.V(3).Info("Pod marked for RestartAllContainers", "pod", klog.KObj(pod))
		// Kill and remove containers in reverse order. Source containers (which exited and triggered
		// RestartAllContainers) are removed last.
		sourceInitContainers, targetInitContainers := m.getContainersToReset(pod.Spec.InitContainers, podStatus)
		sourceContainers, targetContainers := m.getContainersToReset(pod.Spec.Containers, podStatus)
		changes.ContainersToReset = append(changes.ContainersToReset, targetContainers...)
		changes.ContainersToReset = append(changes.ContainersToReset, targetInitContainers...)
		changes.ContainersToReset = append(changes.ContainersToReset, sourceContainers...)
		changes.ContainersToReset = append(changes.ContainersToReset, sourceInitContainers...)
		// Containers removed from the pod spec are killed, but never restarted.
		m.killRemovedContainers(logger, pod, podStatus, &changes)
		return changes
	}

	var actuatedPod *v1.Pod
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicNodeLocalEphemeralVolumes) || utilfeature.DefaultFeatureGate.Enabled(features.DynamicContainers) {
		if p, ok := m.actuatedState.GetPodReadOnly(pod.UID); ok && p != nil && p.Name != "" {
			actuatedPod = m.reconcileNonRunningActuatedContainers(logger, pod, podStatus, p)
		}
	}

	// If we need to (re-)create the pod sandbox, everything will need to be
	// killed and recreated, and init containers should be purged.
	if createPodSandbox {
		if !shouldRestartOnFailure(pod) && attempt != 0 && len(podStatus.ContainerStatuses) != 0 {
			// Should not restart the pod, just return.
			// we should not create a sandbox, and just kill the pod if it is already done.
			// if all containers are done and should not be started, there is no need to create a new sandbox.
			// this stops confusing logs on pods whose containers all have exit codes, but we recreate a sandbox before terminating it.
			//
			// If ContainerStatuses is empty, we assume that we've never
			// successfully created any containers. In this case, we should
			// retry creating the sandbox.
			changes.CreateSandbox = false
			return changes
		}

		// Get the containers to start, excluding the ones that succeeded if RestartPolicy is OnFailure.
		var containersToStart []int
		for idx, c := range pod.Spec.Containers {
			runOnce := pod.Spec.RestartPolicy == v1.RestartPolicyOnFailure
			if utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules) {
				if c.RestartPolicy != nil {
					runOnce = *c.RestartPolicy == v1.ContainerRestartPolicyOnFailure
				}
			}
			if runOnce && containerSucceeded(&c, podStatus) {
				continue
			}
			if utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules) {
				if c.RestartPolicy != nil && *c.RestartPolicy == v1.ContainerRestartPolicyOnFailure && containerSucceeded(&c, podStatus) {
					continue
				}
			}
			containersToStart = append(containersToStart, idx)
		}

		// We should not create a sandbox, and just kill the pod if initialization
		// is done and there is no container to start.
		if len(containersToStart) == 0 {
			hasInitialized := false

			// If there is any regular container, it means all init containers have
			// been initialized.
			hasInitialized = HasAnyRegularContainerCreated(pod, podStatus)

			if hasInitialized {
				changes.CreateSandbox = false
				return changes
			}
		}

		// If we are creating a pod sandbox, we should restart from the initial
		// state.
		if len(pod.Spec.InitContainers) != 0 {
			// Pod has init containers, return the first one.
			changes.InitContainersToStart = []int{0}

			return changes
		}
		var readyContainersToStart []int
		for _, idx := range containersToStart {
			c := &pod.Spec.Containers[idx]
			if unreadyVol, ready := containerVolumesReady(pod, c, actuatedPod); !ready {
				logger.Info("Deferring container start during sandbox creation until dynamic volume is ready",
					"pod", klog.KObj(pod), "containerName", c.Name, "volumeName", unreadyVol)
				continue
			}
			readyContainersToStart = append(readyContainersToStart, idx)
		}
		changes.ContainersToStart = readyContainersToStart
		return changes
	}

	// Ephemeral containers may be started even if initialization is not yet complete.
	for i := range pod.Spec.EphemeralContainers {
		c := (*v1.Container)(&pod.Spec.EphemeralContainers[i].EphemeralContainerCommon)

		// Ephemeral Containers are never restarted
		if podStatus.FindContainerStatusByName(c.Name) == nil {
			if volName, ready := containerVolumesReady(pod, c, actuatedPod); !ready {
				logger.Info("Deferring ephemeral container start until dynamic volume is ready",
					"pod", klog.KObj(pod), "containerName", c.Name, "volumeName", volName)
				continue
			}
			changes.EphemeralContainersToStart = append(changes.EphemeralContainersToStart, i)
		}
	}

	if resizable, _, _ := allocation.IsInPlacePodVerticalScalingAllowed(pod); resizable {
		changes.ContainersToUpdate = make(map[v1.ResourceName][]containerToUpdateInfo)
	}

	// Check initialization progress.
	// TODO: Remove this code path as logically it is the subset of the next
	// code path.
	hasInitialized := m.computeInitContainerActions(ctx, pod, podStatus, &changes)
	if !changes.KillPod {
		changes.PodCgroupResourcesToUpdate = m.computePodLevelResourcesResizeAction(ctx, pod)
	}
	if changes.KillPod || !hasInitialized {
		// Initialization failed or still in progress. Skip inspecting non-init
		// containers.
		return changes
	}

	// Number of running containers to keep.
	keepCount := 0
	hasDeferredContainerStart := false
	// check the status of containers.
	for idx, container := range pod.Spec.Containers {
		containerStatus := podStatus.FindContainerStatusByName(container.Name)

		// Call internal container post-stop lifecycle hook for any non-running container so that any
		// allocated cpus are released immediately. If the container is restarted, cpus will be re-allocated
		// to it.
		if containerStatus != nil && containerStatus.State != kubecontainer.ContainerStateRunning {
			if err := m.internalLifecycle.PostStopContainer(logger, containerStatus.ID.ID); err != nil {
				logger.Error(err, "Internal container post-stop lifecycle hook failed for container in pod with error",
					"containerName", container.Name, "pod", klog.KObj(pod))
			}
		}

		// If container does not exist, or is not running, check whether we
		// need to restart it.
		if containerStatus == nil || containerStatus.State != kubecontainer.ContainerStateRunning {
			if kubecontainer.ShouldContainerBeRestarted(logger, &container, pod, podStatus) {
				if containerStatus != nil && containerStatus.State == kubecontainer.ContainerStateUnknown {
					// If container is in unknown state, we don't know whether it
					// is actually running or not, always try killing it before
					// restart to avoid having 2 running instances of the same container.
					changes.ContainersToKill[containerStatus.ID] = containerToKillInfo{
						name:      containerStatus.Name,
						container: &pod.Spec.Containers[idx],
						message: fmt.Sprintf("Container is in %q state, try killing it before restart",
							containerStatus.State),
						reason: reasonUnknown,
					}
				}
				if unreadyVolName, ready := containerVolumesReady(pod, &container, actuatedPod); !ready {
					logger.Info("Deferring container start until dynamic volume is ready",
						"pod", klog.KObj(pod), "containerName", container.Name, "volumeName", unreadyVolName)
					hasDeferredContainerStart = true
					continue
				}
				logger.V(3).Info("Container of pod is not in the desired state and shall be started", "containerName", container.Name, "pod", klog.KObj(pod))
				changes.ContainersToStart = append(changes.ContainersToStart, idx)
			}
			continue
		}
		// The container is running, but kill the container if any of the following condition is met.
		var message string
		var reason containerKillReason
		restart := shouldRestartOnFailure(pod)
		if utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules) {
			// For probe failures, use container-level restart policy only. Container-level restart
			// rules are not evaluated because the container is still running.
			if container.RestartPolicy != nil {
				restart = *container.RestartPolicy != v1.ContainerRestartPolicyNever
			}
		}
		if _, _, changed := containerChanged(&container, containerStatus); changed {
			message = fmt.Sprintf("Container %s definition changed", container.Name)
			// Restart regardless of the restart policy because the container
			// spec changed.
			restart = true
		} else if liveness, found := m.livenessManager.Get(containerStatus.ID); found && liveness == proberesults.Failure {
			// If the container failed the liveness probe, we should kill it.
			message = fmt.Sprintf("Container %s failed liveness probe", container.Name)
			reason = reasonLivenessProbe
		} else if startup, found := m.startupManager.Get(containerStatus.ID); found && startup == proberesults.Failure {
			// If the container failed the startup probe, we should kill it.
			message = fmt.Sprintf("Container %s failed startup probe", container.Name)
			reason = reasonStartupProbe
		} else if !m.computePodResizeAction(ctx, pod, idx, false, containerStatus, &changes) {
			// computePodResizeAction updates 'changes' if resize policy requires restarting this container
			continue
		} else {
			// Keep the container.
			keepCount++
			continue
		}

		// We need to kill the container, but if we also want to restart the
		// container afterwards, make the intent clear in the message. Also do
		// not kill the entire pod since we expect container to be running eventually.
		if restart {
			message = fmt.Sprintf("%s, will be restarted", message)
			changes.ContainersToStart = append(changes.ContainersToStart, idx)
		}

		changes.ContainersToKill[containerStatus.ID] = containerToKillInfo{
			name:      containerStatus.Name,
			container: &pod.Spec.Containers[idx],
			message:   message,
			reason:    reason,
		}
		logger.V(2).Info("Message for Container of pod", "containerName", container.Name, "containerStatusID", containerStatus.ID, "pod", klog.KObj(pod), "containerMessage", message)
	}

	m.killRemovedContainers(logger, pod, podStatus, &changes)
	m.dropRemovedActuatedContainers(logger, pod, podStatus)

	// Check for volume resize actions
	m.computeVolumeResizeAction(ctx, pod, &changes)

	if keepCount == 0 && len(changes.ContainersToStart) == 0 && !hasDeferredContainerStart {
		changes.KillPod = true
		// To prevent the restartable init containers to keep pod alive, we should
		// not restart them.
		changes.InitContainersToStart = nil
		changes.PodCgroupResourcesToUpdate = nil
	}

	return changes
}

// killRemovedContainers adds the running containers that were removed from the pod spec to the
// containers to kill. A removed container must not keep running. Only regular containers can be
// removed from the spec, and API validation doesn't let an added container reuse the name of a
// removed one while the pod reports its status, which it does while the removed container has
// runtime containers. So a container that isn't in the pod spec is a regular container that was
// removed from it.
//
// TODO: The CPU and memory managers' removeStaleState works from the API pods, so they release the
// exclusive CPUs and memory of a removed container while it still runs (until it is killed, or
// while its removal isn't allocated yet). Have them keep the allocations of running containers.
func (m *kubeGenericRuntimeManager) killRemovedContainers(logger klog.Logger, pod *v1.Pod, podStatus *kubecontainer.PodStatus, changes *podActions) {
	for _, containerStatus := range podStatus.ContainerStatuses {
		if kubecontainer.GetContainerSpec(pod, containerStatus.Name) != nil {
			continue
		}
		if containerStatus.State != kubecontainer.ContainerStateRunning {
			// As for containers in the spec, drop the topology hints of the container as
			// soon as it stops rather than when it is garbage collected.
			if err := m.internalLifecycle.PostStopContainer(logger, containerStatus.ID.ID); err != nil {
				logger.Error(err, "Internal container post-stop lifecycle hook failed for removed container in pod with error",
					"containerName", containerStatus.Name, "pod", klog.KObj(pod))
			}
		}
		if containerStatus.State != kubecontainer.ContainerStateRunning && containerStatus.State != kubecontainer.ContainerStateUnknown {
			continue
		}
		message := fmt.Sprintf("Container %s removed from pod spec", containerStatus.Name)
		changes.ContainersToKill[containerStatus.ID] = containerToKillInfo{
			name:    containerStatus.Name,
			message: message,
		}
		logger.V(2).Info("Message for Container of pod", "containerName", containerStatus.Name, "containerStatusID", containerStatus.ID, "pod", klog.KObj(pod), "containerMessage", message)
	}
}

// dropRemovedActuatedContainers drops the actuated records of containers removed from the pod spec that
// have no runtime container left (it was pruned, garbage collected, or never created). A record
// thus lasts as long as the status of the removed container, which is generated from its runtime
// container.
func (m *kubeGenericRuntimeManager) dropRemovedActuatedContainers(logger klog.Logger, pod *v1.Pod, podStatus *kubecontainer.PodStatus) {
	actuated, ok := m.actuatedState.GetPod(pod.UID)
	if !ok {
		return
	}
	var forget []string
	for pair := range podutil.ContainerPairs(pod, actuated, podutil.Containers) {
		if pair.New == nil && podStatus.FindContainerStatusByName(pair.Old.Name) == nil {
			if hasSubPathVolumeMount(pair.Old.VolumeMounts) && m.runtimeHelper != nil {
				if err := m.runtimeHelper.CleanContainerSubPaths(pod.UID, pair.Old.Name); err != nil {
					logger.Error(err, "Failed to clean container subPath mounts for dropped container", "pod", klog.KObj(pod), "containerName", pair.Old.Name)
					continue
				}
			}
			forget = append(forget, pair.Old.Name)
		}
	}
	if len(forget) == 0 {
		return
	}
	if err := m.actuatedState.RemoveContainers(logger, pod.UID, podutil.Containers, forget...); err != nil {
		logger.Error(err, "Failed to drop actuated records of containers removed from the pod spec", "pod", klog.KObj(pod), "containers", forget)
	}
}

// pruneRemovedContainers removes the runtime containers of containers removed from the pod spec
// beyond the newest maxRemovedContainers (see removedContainersToPrune), and returns a result for
// each. Once all the runtime containers of a removed container are removed, its actuated record is
// dropped too, so that a container added later under the same name doesn't find it.
func (m *kubeGenericRuntimeManager) pruneRemovedContainers(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus) []*kubecontainer.SyncResult {
	containerIDs := removedContainersToPrune(pod, podStatus)
	if len(containerIDs) == 0 {
		return nil
	}
	logger := klog.FromContext(ctx)
	var results []*kubecontainer.SyncResult
	removed := sets.New[kubecontainer.ContainerID]()
	for _, containerID := range containerIDs {
		logger.V(3).Info("Removing container removed from pod spec", "containerID", containerID, "pod", klog.KObj(pod))
		result := kubecontainer.NewSyncResult(kubecontainer.RemoveContainer, containerID.ID)
		results = append(results, result)
		if err := m.removeContainer(ctx, containerID.ID, false); err != nil {
			result.Fail(kubecontainer.ErrRemoveContainer, err.Error())
			logger.Error(err, "Failed to remove container removed from pod spec", "containerID", containerID, "pod", klog.KObj(pod))
			continue
		}
		removed.Insert(containerID)
	}
	gone := sets.New[string]()
	remaining := sets.New[string]()
	for _, s := range podStatus.ContainerStatuses {
		if removed.Has(s.ID) {
			gone.Insert(s.Name)
		} else {
			remaining.Insert(s.Name)
		}
	}
	if names := sets.List(gone.Difference(remaining)); len(names) > 0 {
		var cleaned []string
		for _, name := range names {
			if m.runtimeHelper != nil {
				if err := m.runtimeHelper.CleanContainerSubPaths(pod.UID, name); err != nil {
					logger.Error(err, "Failed to clean container subPath mounts for pruned container", "pod", klog.KObj(pod), "containerName", name)
					continue
				}
			}
			cleaned = append(cleaned, name)
		}
		if len(cleaned) > 0 {
			if err := m.actuatedState.RemoveContainers(logger, pod.UID, podutil.Containers, cleaned...); err != nil {
				logger.Error(err, "Failed to drop actuated records of pruned containers", "pod", klog.KObj(pod), "containers", cleaned)
			}
		}
	}
	return results
}

// maxRemovedContainers is the number of containers removed from the pod spec
// whose runtime containers, and so statuses and logs, are kept. It bounds the
// pod status size for pods that churn containers.
const maxRemovedContainers = 10

// removedContainersToPrune returns the runtime containers of the containers
// removed from the pod spec that finished longest ago, beyond the newest
// maxRemovedContainers removed containers. Removed containers that are still
// running count toward the limit but are never pruned.
//
// The status of a removed container is generated from its runtime container,
// so pruning the runtime container drops the status too. Keeping the two in
// sync matters: the API allows a name to be reused once its status is gone,
// and a leftover runtime container would be mistaken for the new container.
func removedContainersToPrune(pod *v1.Pod, podStatus *kubecontainer.PodStatus) []kubecontainer.ContainerID {
	// The most recent runtime container for each removed container name.
	latest := map[string]*kubecontainer.Status{}
	for _, s := range podStatus.ContainerStatuses {
		if kubecontainer.GetContainerSpec(pod, s.Name) != nil {
			continue
		}
		if l, ok := latest[s.Name]; !ok || s.CreatedAt.After(l.CreatedAt) {
			latest[s.Name] = s
		}
	}
	if len(latest) <= maxRemovedContainers {
		return nil
	}

	var stopped []*kubecontainer.Status
	for _, s := range latest {
		if s.State != kubecontainer.ContainerStateRunning && s.State != kubecontainer.ContainerStateUnknown {
			stopped = append(stopped, s)
		}
	}
	// Oldest finished first; names break ties for determinism.
	sort.Slice(stopped, func(i, j int) bool {
		if !stopped[i].FinishedAt.Equal(stopped[j].FinishedAt) {
			return stopped[i].FinishedAt.Before(stopped[j].FinishedAt)
		}
		return stopped[i].Name < stopped[j].Name
	})
	toPrune := min(len(latest)-maxRemovedContainers, len(stopped))
	names := sets.New[string]()
	for _, s := range stopped[:toPrune] {
		names.Insert(s.Name)
	}

	var ids []kubecontainer.ContainerID
	for _, s := range podStatus.ContainerStatuses {
		if names.Has(s.Name) && kubecontainer.GetContainerSpec(pod, s.Name) == nil {
			ids = append(ids, s.ID)
		}
	}
	return ids
}

// getContainersToReset returns container info about the containers to remove from the runtime.
// The first list are the containers that triggered the RestartAllContainers; the second list
// are the containers that are victim of the RestartAllContainers.
func (m *kubeGenericRuntimeManager) getContainersToReset(containers []v1.Container, podStatus *kubecontainer.PodStatus) (sources []containerToRemoveInfo, targets []containerToRemoveInfo) {
	for idx, c := range containers {
		// podStatus.FindContainerStatusByName cannot be used because there can be multiple container
		// statuses per container, and RestartAllContainers require all container to be purged from runtime.
		for _, containerStatus := range podStatus.ContainerStatuses {
			if containerStatus.Name != c.Name {
				continue
			}
			info := containerToRemoveInfo{
				containerID: containerStatus.ID,
				container:   &containers[idx],
			}
			if containerStatus.State == kubecontainer.ContainerStateExited {
				exitCode := containerStatus.ExitCode
				rule, ok := podutil.FindMatchingContainerRestartRule(c, int32(exitCode))
				if ok && rule.Action == v1.ContainerRestartRuleActionRestartAllContainers {
					sources = append(sources, info)
				} else {
					targets = append(targets, info)
				}
			} else {
				info.kill = true
				targets = append(targets, info)
			}
		}
	}
	return
}

func (m *kubeGenericRuntimeManager) computePodLevelResourcesResizeAction(ctx context.Context, pod *v1.Pod) sets.Set[v1.ResourceName] {
	if resizable, _, _ := allocation.IsInPlacePodVerticalScalingAllowed(pod); !resizable {
		return nil
	}

	actuatedPodLevelResources, found := m.actuatedState.GetPodLevelResources(pod.UID)
	if !found || actuatedPodLevelResources == nil {
		klog.FromContext(ctx).Error(nil, "Missing actuated pod level resource record", "pod", klog.KObj(pod))
		return nil
	}

	toUpdate := podCgroupResourcesToUpdate(cm.EffectivePodResources(pod), actuatedPodLevelResources)
	// KEP-5359: the pod-level swap limit is a pod cgroup limit of its own.
	if m.workloadSwapLimitsEnabled() && !podSwapLimitEqual(pod.Spec.Resources, actuatedPodLevelResources) {
		if toUpdate == nil {
			toUpdate = sets.New[v1.ResourceName]()
		}
		toUpdate.Insert(v1.ResourceSwap)
	}
	return toUpdate
}

// podCgroupResourcesToUpdate returns the resources (cpu, memory) whose desired pod cgroup requests
// or limits differ from the actuated ones.
func podCgroupResourcesToUpdate(desired, actuated *v1.ResourceRequirements) sets.Set[v1.ResourceName] {
	var toUpdate sets.Set[v1.ResourceName]
	for _, rName := range []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory} {
		if !desired.Requests[rName].Equal(actuated.Requests[rName]) || !desired.Limits[rName].Equal(actuated.Limits[rName]) {
			if toUpdate == nil {
				toUpdate = sets.New[v1.ResourceName]()
			}
			toUpdate.Insert(rName)
		}
	}
	return toUpdate
}

// SyncPod syncs the running pod into the desired pod by executing following steps:
//
//  1. Compute sandbox and container changes.
//  2. Kill pod sandbox if necessary.
//  3. Kill any containers that should not be running.
//  4. Create sandbox if necessary.
//  5. Invoke OnPodSandboxReady to notify Kubelet to update pod status.
//  6. Resize pod & running containers
//  7. Create ephemeral containers.
//  8. Create init containers.
//  9. Create normal containers.
func (m *kubeGenericRuntimeManager) SyncPod(ctx context.Context, pod *v1.Pod, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret, backOff *flowcontrol.Backoff, restartAllContainers bool) (result kubecontainer.PodSyncResult) {
	defer func() {
		// Observe SyncResult error metrics, excluding expected backoff errors.
		if result.SyncError != nil && !kubecontainer.IsBackoffError(result.SyncError) {
			metrics.SyncResultErrorsTotal.WithLabelValues(metrics.SyncResultActionSyncPod).Inc()
		}
		for _, r := range result.SyncResults {
			if r.Error != nil && !kubecontainer.IsBackoffError(r.Error) {
				metrics.SyncResultErrorsTotal.WithLabelValues(string(r.Action)).Inc()
			}
		}
	}()
	logger := klog.FromContext(ctx)
	// Step 1: Compute sandbox and container changes.
	podContainerChanges := m.computePodActions(ctx, pod, podStatus, restartAllContainers)
	logger.V(3).Info("computePodActions got for pod", "podActions", podContainerChanges, "pod", klog.KObj(pod))
	if podContainerChanges.CreateSandbox {
		ref, err := ref.GetReference(legacyscheme.Scheme, pod)
		if err != nil {
			logger.Error(err, "Couldn't make a ref to pod", "pod", klog.KObj(pod))
		}
		if podContainerChanges.SandboxID != "" {
			m.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeNormal, events.SandboxChanged, "Pod sandbox changed, it will be killed and re-created.")
		} else {
			logger.V(4).Info("SyncPod received new pod, will create a sandbox for it", "pod", klog.KObj(pod))
		}
	}

	// Step 2: Kill the pod if the sandbox has changed.
	if podContainerChanges.KillPod {
		if podContainerChanges.CreateSandbox {
			logger.V(4).Info("Stopping PodSandbox for pod, will start new one", "pod", klog.KObj(pod))
		} else {
			logger.V(4).Info("Stopping PodSandbox for pod, because all other containers are dead", "pod", klog.KObj(pod))
		}

		killResult := m.killPodWithSyncResult(ctx, pod, kubecontainer.ConvertPodStatusToRunningPod(m.runtimeName, podStatus), nil)
		result.AddPodSyncResult(killResult)
		if killResult.Error() != nil {
			logger.Error(killResult.Error(), "killPodWithSyncResult failed")
			return
		}

		if podContainerChanges.CreateSandbox {
			m.purgeInitContainers(ctx, pod, podStatus)
		}
	} else {
		// Step 3: kill any running containers in this pod which are not to keep.
		for containerID, containerInfo := range podContainerChanges.ContainersToKill {
			logger.V(3).Info("Killing unwanted container for pod", "containerName", containerInfo.name, "containerID", containerID, "pod", klog.KObj(pod))
		}
		// Termination ordering (e.g. for sidecars) only applies when the whole pod is
		// shutting down. Here we are only killing individual containers, so no
		// ordering is enforced between them.
		// TODO: Killing a container removed from the pod spec blocks the sync for its whole grace
		// period, which delays other changes to the pod, and its deletion. Kill removed containers
		// without blocking the sync.
		result.AddSyncResult(m.killContainers(ctx, pod, podContainerChanges.ContainersToKill, nil, nil)...)
		if result.Error() != nil {
			return
		}

		// Failures are retried on the next sync, and don't block the rest of this one.
		result.AddSyncResult(m.pruneRemovedContainers(ctx, pod, podStatus)...)

		// Removes the containers if they are marked for removal (for in-place restart)
		if utilfeature.DefaultFeatureGate.Enabled(features.RestartAllContainersOnContainerExits) {
			for _, containerInfo := range podContainerChanges.ContainersToReset {
				cName := containerInfo.container.Name
				logger.V(3).Info("Removing container before pod restarts", "containerName", cName, "containerID", containerInfo.containerID, "pod", klog.KObj(pod))
				removeContainerResult := kubecontainer.NewSyncResult(kubecontainer.RemoveContainer, cName)
				result.AddSyncResult(removeContainerResult)
				if containerInfo.kill {
					logger.V(3).Info("Killing container before removal", "containerName", cName, "containerID", containerInfo.containerID, "pod", klog.KObj(pod))
					// Killing containers without grace period.
					var gracePeriod int64 = 0
					if err := m.killContainer(ctx, pod, containerInfo.containerID, cName, "killing", reasonRestartAllContainers, &gracePeriod, nil); err != nil {
						removeContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
						logger.Error(err, "killContainer for pod failed", "containerName", cName, "containerID", containerInfo.containerID, "pod", klog.KObj(pod))
						return
					}
				}
				// The logs of removed containers will be preserved until the pod is deleted and GC is triggered.
				if err := m.removeContainer(ctx, containerInfo.containerID.ID, true); err != nil {
					removeContainerResult.Fail(kubecontainer.ErrRemoveContainer, err.Error())
					logger.Error(err, "removeContainer for pod failed", "containerName", cName, "containerID", containerInfo.containerID, "pod", klog.KObj(pod))
					return
				}
			}
		}
	}

	// Keep terminated init containers fairly aggressively controlled
	// This is an optimization because container removals are typically handled
	// by container garbage collector.
	m.pruneInitContainersBeforeStart(ctx, pod, podStatus)

	// We pass the value of the PRIMARY podIP and list of podIPs down to
	// generatePodSandboxConfig and generateContainerConfig, which in turn
	// passes it to various other functions, in order to facilitate functionality
	// that requires this value (hosts file and downward API) and avoid races determining
	// the pod IP in cases where a container requires restart but the
	// podIP isn't in the status manager yet. The list of podIPs is used to
	// generate the hosts file.
	//
	// We default to the IPs in the passed-in pod status, and overwrite them if the
	// sandbox needs to be (re)started.
	var podIPs []string
	if podStatus != nil {
		podIPs = podStatus.IPs
	}

	// Step 4: Create a sandbox for the pod if necessary.
	podSandboxID := podContainerChanges.SandboxID
	restorePod, restoreSandbox := podSandboxRestore(pod, podStatus)

	if podContainerChanges.CreateSandbox {
		var sandboxStatus *runtimeapi.PodSandboxStatus

		logger.V(4).Info("Creating PodSandbox for pod", "pod", klog.KObj(pod))
		metrics.StartedPodsTotal.Inc()
		if utilfeature.DefaultFeatureGate.Enabled(features.UserNamespacesSupport) && pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
			metrics.StartedUserNamespacedPodsTotal.Inc()
			// Failures in user namespace creation could happen at any point in the pod lifecycle,
			// but usually will be caught in container creation.
			// To avoid specifically handling each error case, loop through the result after the sync finishes
			defer func() {
				// catch unhandled errors
				for _, res := range result.SyncResults {
					if res.Error != nil {
						metrics.StartedUserNamespacedPodsErrorsTotal.Inc()
						return
					}
				}
				// catch handled error
				if result.SyncError != nil {
					metrics.StartedUserNamespacedPodsErrorsTotal.Inc()
				}
			}()
		}
		createSandboxResult := kubecontainer.NewSyncResult(podSandboxSyncAction(restoreSandbox), format.Pod(pod))
		result.AddSyncResult(createSandboxResult)

		// ConvertPodSysctlsVariableToDotsSeparator converts sysctl variable
		// in the Pod.Spec.SecurityContext.Sysctls slice into a dot as a separator.
		// runc uses the dot as the separator to verify whether the sysctl variable
		// is correct in a separate namespace, so when using the slash as the sysctl
		// variable separator, runc returns an error: "sysctl is not in a separate kernel namespace"
		// and the podSandBox cannot be successfully created. Therefore, before calling runc,
		// we need to convert the sysctl variable, the dot is used as a separator to separate the kernel namespace.
		// When runc supports slash as sysctl separator, this function can no longer be used.
		sysctl.ConvertPodSysctlsVariableToDotsSeparator(pod.Spec.SecurityContext)

		// Prepare resources allocated by the Dynamic Resource Allocation feature for the pod
		if err := m.runtimeHelper.PrepareDynamicResources(ctx, pod); err != nil {
			reason := dynamicResourcesFailureReason(restoreSandbox)
			if restoreSandbox {
				createSandboxResult.Fail(&kubecontainer.RestoreError{Reason: reason, Err: err}, err.Error())
			}
			ref, referr := ref.GetReference(legacyscheme.Scheme, pod)
			if referr != nil {
				logger.Error(referr, "Couldn't make a ref to pod", "pod", klog.KObj(pod))
				return
			}
			m.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeWarning, reason, "Failed to prepare dynamic resources: %v", err)
			logger.Error(err, "Failed to prepare dynamic resources", "pod", klog.KObj(pod))
			return
		}

		var started bool
		if restoreSandbox {
			podSandboxID, sandboxStatus, started = m.restorePodSandboxForSync(ctx, pod, podContainerChanges.Attempt, pullSecrets, createSandboxResult)
		} else {
			podSandboxID, started = m.createPodSandboxForSync(ctx, pod, podContainerChanges.Attempt, createSandboxResult)
		}
		if !started {
			return
		}

		if sandboxStatus == nil {
			resp, err := m.runtimeService.PodSandboxStatus(ctx, podSandboxID, false)
			if err != nil {
				ref, referr := ref.GetReference(legacyscheme.Scheme, pod)
				if referr != nil {
					logger.Error(referr, "Couldn't make a ref to pod", "pod", klog.KObj(pod))
				}
				m.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeWarning, events.FailedStatusPodSandBox, "Unable to get pod sandbox status: %v", err)
				logger.Error(err, "Failed to get pod sandbox status; Skipping pod", "pod", klog.KObj(pod))
				result.Fail(err)
				return
			}
			if resp.GetStatus() == nil {
				result.Fail(errors.New("pod sandbox status is nil"))
				return
			}

			sandboxStatus = resp.GetStatus()
		}

		// If we ever allow updating a pod from non-host-network to
		// host-network, we may use a stale IP.
		if !kubecontainer.IsHostNetworkPod(pod) {
			// Overwrite the podIPs passed in the pod status, since we just started the pod sandbox.
			podIPs = m.determinePodSandboxIPs(ctx, pod.Namespace, pod.Name, sandboxStatus)
			logger.V(4).Info("Determined the ip for pod after sandbox changed", "IPs", podIPs, "pod", klog.KObj(pod))
		}

		// A runtime that reported the pod_network_isolated capability but
		// attached the sandbox to the pod network anyway must not get any
		// container started in it: the pod asked for isolation. Stop the
		// sandbox and fail; the API server also rejects a status with IPs.
		if kubecontainer.IsNetworkIsolatedPod(pod) && len(podIPs) > 0 {
			msg := fmt.Sprintf("runtime attached the sandbox of a pod with podNetwork %q to the pod network (IPs %v)", v1.PodNetworkIsolated, podIPs)
			if err := m.runtimeService.StopPodSandbox(ctx, podSandboxID); err != nil && !crierror.IsNotFound(err) {
				logger.Error(err, "Failed to stop sandbox with unexpected IPs", "pod", klog.KObj(pod), "podSandboxID", podSandboxID)
			}
			metrics.StartedPodsErrorsTotal.Inc()
			createSandboxResult.Fail(kubecontainer.ErrCreatePodSandbox, msg)
			logger.Error(nil, "CreatePodSandbox for pod produced a sandbox with IPs for a network isolated pod", "pod", klog.KObj(pod), "podSandboxID", podSandboxID, "IPs", podIPs)
			ref, referr := ref.GetReference(legacyscheme.Scheme, pod)
			if referr != nil {
				logger.Error(referr, "Couldn't make a ref to pod", "pod", klog.KObj(pod))
			}
			m.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeWarning, events.FailedCreatePodSandBox, "Failed to create pod sandbox: %s", msg)
			return
		}

		// Step 5: invoke the sandbox ready callback before image pulling .
		// At this point, dynamic resources are prepared (at PrepareDynamicResources() call above)
		// and volumes are already mounted (at the kubelet SyncPod() level), so,
		// all requirements (sandbox, networking, volumes) are met to set `PodReadyToStartContainers=True` condition.
		logger.V(4).Info("Pod sandbox and network ready, invoking callback", "pod", klog.KObj(pod), "podIPs", podIPs)
		if err := m.runtimeHelper.OnPodSandboxReady(ctx, pod); err != nil {
			// log the error but continue the pod creation process, to retain the existing behaviour
			logger.Error(err, "Failed to invoke sandbox ready callback, continuing with pod creation", "pod", klog.KObj(pod))
		}
	}

	// the start containers routines depend on pod ip(as in primary pod ip)
	// instead of trying to figure out if we have 0 < len(podIPs)
	// everytime, we short circuit it here
	podIP := ""
	if len(podIPs) != 0 {
		podIP = podIPs[0]
	}
	if restorePod && !restoreSandbox {
		if err := m.updateRestoredPodHosts(ctx, pod, podIPs); err != nil {
			result.Fail(err)
			return
		}
	}

	// Step 6: Resize pod & running containers (if necessary).
	if resizable, _, _ := allocation.IsInPlacePodVerticalScalingAllowed(pod); resizable {
		updateVolumes := utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) &&
			(len(podContainerChanges.VolumesToUpsize) > 0 || len(podContainerChanges.VolumesToDownsize) > 0)
		if len(podContainerChanges.ContainersToUpdate) > 0 || len(podContainerChanges.PodCgroupResourcesToUpdate) > 0 || updateVolumes {
			result.AddSyncResult(m.doPodResizeAction(ctx, pod, podStatus, podContainerChanges))
		}
	}

	// Helper containing boilerplate common to starting all types of containers.
	// typeName is a description used to describe this type of container in log messages,
	// currently: "container", "init container" or "ephemeral container"
	// metricLabel is the label used to describe this type of container in monitoring metrics.
	// currently: "container", "init_container" or "ephemeral_container"
	startWithInitState := func(ctx context.Context, typeName, metricLabel string, spec *startSpec, podSandboxConfig *runtimeapi.PodSandboxConfig, imageVolumePullResults imageVolumePulls) error {
		startContainerResult := kubecontainer.NewSyncResult(kubecontainer.StartContainer, spec.container.Name)
		result.AddSyncResult(startContainerResult)

		isInBackOff, msg, err := m.doBackOff(ctx, pod, spec.container, podStatus, backOff)
		if isInBackOff {
			startContainerResult.Fail(err, msg)
			logger.V(4).Info("Backing Off restarting container in pod", "containerType", typeName, "container", spec.container.Name, "pod", klog.KObj(pod))
			return err
		}

		countFailure := countContainerStart(pod, spec.container, metricLabel)
		logger.V(4).Info("Creating container in pod", "containerType", typeName, "container", spec.container.Name, "pod", klog.KObj(pod))

		// We fail late here to populate the "ErrImagePull" and "ImagePullBackOff" correctly to the end user.
		imageVolumes, err := m.toKubeContainerImageVolumes(ctx, imageVolumePullResults, spec.container, pod, startContainerResult)
		if err != nil {
			return err
		}

		// NOTE (aramase) podIPs are populated for single stack and dual stack clusters. Send only podIPs.
		msg, err = m.startContainer(ctx, podSandboxID, podSandboxConfig, spec, pod, podStatus, pullSecrets, podIP, podIPs, imageVolumes)
		incrementImageVolumeMetrics(err, msg, spec.container, imageVolumes)
		if err != nil {
			// startContainer() returns well-defined error codes that have reasonable cardinality for metrics and are
			// useful to cluster administrators to distinguish "server errors" from "user errors".
			countFailure(err)
			startContainerResult.Fail(err, msg)
			// known errors that are logged in other places are logged at higher levels here to avoid
			// repetitive log spam
			switch {
			case errors.Is(err, images.ErrImagePullBackOff), errors.Is(err, kubecontainer.ErrSecurityProfilePullBackOff):
				logger.V(3).Info("Container start failed in pod", "containerType", typeName, "container", spec.container.Name, "pod", klog.KObj(pod), "containerMessage", msg, "err", err)
			default:
				utilruntime.HandleError(fmt.Errorf("%v %v start failed in pod %v: %w: %s", typeName, spec.container.Name, format.Pod(pod), err, msg))
			}
			return err
		}
		if typeName == "init container" {
			// Don't measure restartable init containers (sidecars)
			if !podutil.IsRestartableInitContainer(spec.container) {
				if m.podInitContainerTimeRecorder != nil {
					m.podInitContainerTimeRecorder.RecordInitContainerStarted(pod.UID, time.Now())
				}
			}
		}

		return nil
	}

	// State required only if we need to start any containers.
	var (
		podSandboxConfig       *runtimeapi.PodSandboxConfig
		imageVolumePullResults imageVolumePulls
	)

	// lazyStart initializes state required to start containers and returns a start function.
	lazyStart := func() func(ctx context.Context, typeName, metricLabel string, spec *startSpec) error {
		var err error

		if podSandboxConfig == nil {
			podSandboxConfig, err = m.generatePodSandboxConfig(ctx, pod, podContainerChanges.Attempt)
			if err != nil {
				logger.Error(err, "GeneratePodSandboxConfig for pod failed", "pod", klog.KObj(pod))
				result.Fail(fmt.Errorf("GeneratePodSandboxConfig for pod %q failed: %w", format.Pod(pod), err))
				return nil
			}
		}

		if imageVolumePullResults == nil {
			imageVolumePullResults, err = m.getImageVolumes(ctx, pod, podSandboxConfig, pullSecrets)
			if err != nil {
				// Don't start any containers until we have all image volumes.
				logger.Error(err, "Get image volumes for pod failed", "pod", klog.KObj(pod))
				result.Fail(err)
				return nil
			}
		}

		return func(ctx context.Context, typeName, metricLabel string, spec *startSpec) error {
			return startWithInitState(ctx, typeName, metricLabel, spec, podSandboxConfig, imageVolumePullResults)
		}
	}

	// When a pod is restored from a checkpoint, restorePodSandbox already
	// started its containers. Skip container creation and startup.
	if restoreSandbox {
		logger.V(4).Info("Pod was restored from checkpoint, skipping container startup", "pod", klog.KObj(pod))
		return result
	}

	// Step 7: start ephemeral containers
	// These are started "prior" to init containers to allow running ephemeral containers even when there
	// are errors starting an init container. In practice init containers will start first since ephemeral
	// containers cannot be specified on pod creation.
	for _, idx := range podContainerChanges.EphemeralContainersToStart {
		start := lazyStart()
		if start == nil {
			return
		}
		// Errors are recorded in result inside start; a failed ephemeral
		// container start does not abort the sync.
		_ = start(ctx, "ephemeral container", metrics.EphemeralContainer, ephemeralContainerStartSpec(&pod.Spec.EphemeralContainers[idx]))
	}

	// Step 8: start init containers.
	for _, idx := range podContainerChanges.InitContainersToStart {
		start := lazyStart()
		if start == nil {
			return
		}

		container := &pod.Spec.InitContainers[idx]
		// Start the next init container.
		if err := start(ctx, "init container", metrics.InitContainer, initContainerStartSpec(container)); err != nil {
			if podutil.IsRestartableInitContainer(container) {
				logger.V(4).Info("Failed to start the restartable init container for the pod, skipping", "initContainerName", container.Name, "pod", klog.KObj(pod))
				continue
			}
			logger.V(4).Info("Failed to initialize the pod, as the init container failed to start, aborting", "initContainerName", container.Name, "pod", klog.KObj(pod))
			return
		}

		// Successfully started the container; clear the entry in the failure
		logger.V(4).Info("Completed init container for pod", "containerName", container.Name, "pod", klog.KObj(pod))
	}

	for _, cs := range podStatus.ContainerStatuses {
		// Check if this is an init container
		for _, init := range pod.Spec.InitContainers {
			// Don't measure restartable init containers (sidecars)
			if podutil.IsRestartableInitContainer(&init) {
				continue
			}
			if cs.Name == init.Name && cs.State == kubecontainer.ContainerStateExited && !cs.FinishedAt.IsZero() {
				if m.podInitContainerTimeRecorder != nil {
					m.podInitContainerTimeRecorder.RecordInitContainerFinished(pod.UID, cs.FinishedAt)
				}
			}
		}
	}

	// Step 9: start containers in podContainerChanges.ContainersToStart.
	for _, idx := range podContainerChanges.ContainersToStart {
		start := lazyStart()
		if start == nil {
			return
		}

		// Errors are recorded in result inside start; remaining containers
		// are still attempted.
		_ = start(ctx, "container", metrics.Container, containerStartSpec(&pod.Spec.Containers[idx]))
	}

	return result
}

// podSandboxRestore reports whether pod restores from a checkpoint and whether
// this sync restores its sandbox. Restore is only performed for the pod's
// first-ever sandbox creation: it replaces the initial createPodSandbox with a
// restore from the checkpoint's captured state. Once any sandbox has existed for
// this pod, a later sandbox loss is handled by the normal restart path
// (createPodSandbox honoring the RestartPolicy), not by re-restoring the stale
// memory image.
func podSandboxRestore(pod *v1.Pod, podStatus *kubecontainer.PodStatus) (restorePod, restoreSandbox bool) {
	_, restored := podutil.GetPodCondition(&pod.Status, v1.PodRestored)
	restorePod = utilfeature.DefaultFeatureGate.Enabled(features.PodLevelCheckpointRestore) &&
		pod.Spec.RestoreFrom != nil && pod.Spec.RestoreFrom.Name != "" &&
		(restored == nil || restored.Status == v1.ConditionUnknown)
	restoreSandbox = restorePod && (podStatus == nil || len(podStatus.SandboxStatuses) == 0)
	return restorePod, restoreSandbox
}

// podSandboxSyncAction returns the sync action that starts the Pod sandbox.
func podSandboxSyncAction(restore bool) kubecontainer.SyncAction {
	if restore {
		return kubecontainer.RestorePodSandbox
	}
	return kubecontainer.CreatePodSandbox
}

// dynamicResourcesFailureReason returns the event reason for a DRA preparation
// failure. A restore reports it as a retryable preparation error, because
// nothing was restored yet.
func dynamicResourcesFailureReason(restore bool) string {
	if restore {
		return events.RestorePreparationFailed
	}
	return events.FailedPrepareDynamicResources
}

// createPodSandboxForSync creates the Pod sandbox and reports a failure in
// sandboxResult. It returns false when SyncPod must stop.
func (m *kubeGenericRuntimeManager) createPodSandboxForSync(ctx context.Context, pod *v1.Pod, attempt uint32, sandboxResult *kubecontainer.SyncResult) (string, bool) {
	logger := klog.FromContext(ctx)
	podSandboxID, msg, err := m.createPodSandbox(ctx, pod, attempt)
	if err != nil {
		// createPodSandbox can return an error from CNI, CSI,
		// or CRI if the Pod has been deleted while the POD is
		// being created. If the pod has been deleted then it's
		// not a real error.
		//
		// SyncPod can still be running when we get here, which
		// means the PodWorker has not acked the deletion.
		if m.podStateProvider.IsPodTerminationRequested(pod.UID) {
			logger.V(4).Info("Pod was deleted and sandbox failed to be created", "pod", klog.KObj(pod), "podUID", pod.UID)
			return "", false
		}
		metrics.StartedPodsErrorsTotal.Inc()
		sandboxResult.Fail(kubecontainer.ErrCreatePodSandbox, msg)
		logger.Error(err, "CreatePodSandbox for pod failed", "pod", klog.KObj(pod))
		ref, referr := ref.GetReference(legacyscheme.Scheme, pod)
		if referr != nil {
			logger.Error(referr, "Couldn't make a ref to pod", "pod", klog.KObj(pod))
		}
		m.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeWarning, events.FailedCreatePodSandBox, "Failed to create pod sandbox: %v", err)
		return "", false
	}
	logger.V(4).Info("Created PodSandbox for pod", "podSandboxID", podSandboxID, "pod", klog.KObj(pod))
	return podSandboxID, true
}

// restorePodSandboxForSync restores the Pod sandbox from its checkpoint and
// reports a failure in sandboxResult with the restore reason, so the kubelet can
// tell retryable preparation failures from terminal ones. It returns false when
// SyncPod must stop.
func (m *kubeGenericRuntimeManager) restorePodSandboxForSync(ctx context.Context, pod *v1.Pod, attempt uint32, pullSecrets []v1.Secret, sandboxResult *kubecontainer.SyncResult) (string, *runtimeapi.PodSandboxStatus, bool) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Restoring PodSandbox from checkpoint for pod", "pod", klog.KObj(pod), "checkpointName", pod.Spec.RestoreFrom.Name)
	restoredSandbox, msg, err := m.restorePodSandbox(ctx, pod, attempt, pullSecrets)
	if err != nil {
		if kubecontainer.RestoreErrorReason(err) == "" {
			err = &kubecontainer.RestoreError{Reason: events.RestoreFailed, Err: err}
		}
		// As with createPodSandbox, an error after the Pod was deleted is not
		// a real error. Termination removes anything the restore created.
		if m.podStateProvider.IsPodTerminationRequested(pod.UID) {
			logger.V(4).Info("Pod was deleted and sandbox failed to be restored", "pod", klog.KObj(pod), "podUID", pod.UID)
			return "", nil, false
		}
		metrics.StartedPodsErrorsTotal.Inc()
		// Preserve the restore reason so kubelet can retry lock contention.
		sandboxResult.Fail(fmt.Errorf("%w: %w", kubecontainer.ErrCreatePodSandbox, err), msg)
		logger.Error(err, "RestorePodSandbox for pod failed", "pod", klog.KObj(pod), "checkpointName", pod.Spec.RestoreFrom.Name)
		ref, referr := ref.GetReference(legacyscheme.Scheme, pod)
		if referr != nil {
			logger.Error(referr, "Couldn't make a ref to pod", "pod", klog.KObj(pod))
		}
		// Surface the specific restore failure reason (CheckpointNotReady,
		// CheckpointWrongNode, PodSpecMismatch, RestoreInProgress) so operators
		// can distinguish failure causes.
		m.recorder.WithLogger(logger).Eventf(ref, v1.EventTypeWarning, kubecontainer.RestoreErrorReason(err), "Failed to restore pod sandbox from checkpoint: %v", err)
		return "", nil, false
	}
	logger.V(4).Info("Restored PodSandbox for pod", "podSandboxID", restoredSandbox.id, "pod", klog.KObj(pod), "checkpointName", pod.Spec.RestoreFrom.Name)
	return restoredSandbox.id, restoredSandbox.status, true
}

// updateRestoredPodHosts lets an existing restored sandbox retry hosts
// bookkeeping without repeating RestorePod or starting its containers again.
func (m *kubeGenericRuntimeManager) updateRestoredPodHosts(ctx context.Context, pod *v1.Pod, podIPs []string) error {
	if !kubecontainer.IsHostNetworkPod(pod) && len(podIPs) == 0 {
		return nil
	}
	if err := m.runtimeHelper.UpdatePodRestoreHosts(ctx, pod, podIPs); err != nil {
		klog.FromContext(ctx).Error(err, "Failed to update restored pod hosts file", "pod", klog.KObj(pod), "podIPs", podIPs)
		return &kubecontainer.RestoreError{Reason: events.RestorePostProcessingFailed, Err: err}
	}
	return nil
}

// incrementImageVolumeMetrics increments the image volume mount metrics
// depending on the provided error and the usage of the image volume mount
// within the container.
func incrementImageVolumeMetrics(err error, msg string, container *v1.Container, imageVolumes kubecontainer.ImageVolumes) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ImageVolume) {
		return
	}

	metrics.ImageVolumeRequestedTotal.Add(float64(len(imageVolumes)))

	for _, m := range container.VolumeMounts {
		if _, exists := imageVolumes[m.Name]; exists {
			if errors.Is(err, ErrCreateContainer) && strings.HasPrefix(msg, crierror.ErrImageVolumeMountFailed.Error()) {
				metrics.ImageVolumeMountedErrorsTotal.Inc()
			} else {
				metrics.ImageVolumeMountedSucceedTotal.Inc()
			}
		}
	}
}

// imageVolumePulls are the pull results for each image volume name.
type imageVolumePulls = map[string]imageVolumePullResult

// imageVolumePullResult is a pull result for a single image volume.
// If spec is nil, then err and msg should be set.
// If err is nil, then spec should be set.
type imageVolumePullResult struct {
	spec *runtimeapi.ImageSpec
	err  error
	msg  string
}

func (m *kubeGenericRuntimeManager) toKubeContainerImageVolumes(ctx context.Context, imageVolumePullResults imageVolumePulls, container *v1.Container, pod *v1.Pod, syncResult *kubecontainer.SyncResult) (kubecontainer.ImageVolumes, error) {
	if len(imageVolumePullResults) == 0 {
		return nil, nil
	}

	imageVolumes := kubecontainer.ImageVolumes{}
	var (
		lastErr error
		lastMsg string
	)
	for _, v := range container.VolumeMounts {
		res, ok := imageVolumePullResults[v.Name]
		if !ok {
			continue
		}

		if res.err != nil {
			s, _ := grpcstatus.FromError(res.err)
			m.recordContainerEvent(ctx, pod, container, "", v1.EventTypeWarning, events.FailedToCreateContainer, "Error: %v", s.Message())
			lastErr = res.err
			lastMsg = res.msg
			continue
		}

		imageVolumes[v.Name] = res.spec
	}

	if lastErr != nil {
		syncResult.Fail(lastErr, lastMsg)
		return nil, lastErr
	}

	return imageVolumes, nil
}

func (m *kubeGenericRuntimeManager) getImageVolumes(ctx context.Context, pod *v1.Pod, podSandboxConfig *runtimeapi.PodSandboxConfig, pullSecrets []v1.Secret) (imageVolumePulls, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ImageVolume) {
		return nil, nil
	}
	runtimeHandler, err := m.getPodRuntimeHandler(pod)
	if err != nil {
		klog.FromContext(ctx).Error(err, "Failed to get pod runtime handler", "pod", klog.KObj(pod))
		return nil, err
	}
	return m.getImageVolumesWithRuntimeHandler(ctx, pod, podSandboxConfig, pullSecrets, runtimeHandler)
}

func (m *kubeGenericRuntimeManager) getImageVolumesWithRuntimeHandler(ctx context.Context, pod *v1.Pod, podSandboxConfig *runtimeapi.PodSandboxConfig, pullSecrets []v1.Secret, runtimeHandler string) (imageVolumePulls, error) {
	logger := klog.FromContext(ctx)
	if !utilfeature.DefaultFeatureGate.Enabled(features.ImageVolume) {
		return nil, nil
	}
	// Image-volume specs go directly into the container config, so apply the
	// same gate used by image-service request conversion.
	podRuntimeHandler := ""
	if utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClassInImageCriAPI) {
		podRuntimeHandler = runtimeHandler
	}

	dynamicVolumesEnabled := utilfeature.DefaultFeatureGate.Enabled(features.DynamicNodeLocalEphemeralVolumes)
	res := make(imageVolumePulls)
	for _, volume := range pod.Spec.Volumes {
		if volume.Image == nil || (dynamicVolumesEnabled && !isVolumeMountedInPodSpec(&pod.Spec, volume.Name)) {
			continue
		}

		objectRef, _ := ref.GetReference(legacyscheme.Scheme, pod) // objectRef can be nil, no error check required
		ref, msg, err := m.imagePuller.EnsureImageExists(
			ctx, objectRef, pod, volume.Image.Reference, pullSecrets, podSandboxConfig, podRuntimeHandler, volume.Image.PullPolicy,
		)
		if err != nil {
			logger.Error(err, "Failed to ensure image", "pod", klog.KObj(pod), "volumeName", volume.Name)
			res[volume.Name] = imageVolumePullResult{err: err, msg: msg}
			continue
		}

		logger.V(4).Info("Pulled image", "ref", ref, "pod", klog.KObj(pod), "volumeName", volume.Name)
		if dynamicVolumesEnabled && m.actuatedState != nil {
			if setErr := m.actuatedState.SetPodVolume(logger, pod.UID, volume); setErr != nil {
				logger.Error(setErr, "Failed to record image volume in actuated state", "pod", klog.KObj(pod), "volumeName", volume.Name)
			}
		}
		res[volume.Name] = imageVolumePullResult{spec: &runtimeapi.ImageSpec{
			Image:              ref,
			UserSpecifiedImage: volume.Image.Reference,
			RuntimeHandler:     podRuntimeHandler,
			Annotations:        pod.Annotations,
		}}
	}

	return res, nil
}

// If a container is still in backoff, the function will return a brief backoff error and
// a detailed error message.
func (m *kubeGenericRuntimeManager) doBackOff(ctx context.Context, pod *v1.Pod, container *v1.Container, podStatus *kubecontainer.PodStatus, backOff *flowcontrol.Backoff) (bool, string, error) {
	logger := klog.FromContext(ctx)
	var cStatus *kubecontainer.Status
	for _, c := range podStatus.ContainerStatuses {
		if c.Name == container.Name && c.State == kubecontainer.ContainerStateExited {
			cStatus = c
			break
		}
	}

	if cStatus == nil {
		return false, "", nil
	}

	logger.V(3).Info("Checking backoff for container in pod", "containerName", container.Name, "pod", klog.KObj(pod))
	// Use the finished time of the latest exited container as the start point to calculate whether to do back-off.
	ts := cStatus.FinishedAt
	// backOff requires a unique key to identify the container.
	key := GetBackoffKey(pod, container)
	if backOff.IsInBackOffSince(key, ts) {
		if containerRef, err := kubecontainer.GenerateContainerRef(pod, container); err == nil {
			m.recorder.WithLogger(logger).Eventf(containerRef, v1.EventTypeWarning, events.BackOffStartContainer,
				"Back-off restarting failed container %s in pod %s", container.Name, format.Pod(pod))
		}
		backoff := backOff.Get(key)
		err := fmt.Errorf("back-off %s restarting failed container=%s pod=%s", backoff, container.Name, format.Pod(pod))
		logger.V(3).Info("Back-off restarting failed container", "err", err.Error())
		return true, err.Error(), kubecontainer.NewBackoffError(kubecontainer.ErrCrashLoopBackOff, ts.Add(backoff))
	}

	backOff.Next(key, ts)
	return false, "", nil
}

// KillPod kills all the containers of a pod. pod must not be nil.
// gracePeriodOverride if specified allows the caller to override the pod default grace period.
// only hard kill paths are allowed to specify a gracePeriodOverride in the kubelet in order to not corrupt user data.
// it is useful when doing SIGKILL for hard eviction scenarios, or max grace period during soft eviction scenarios.
func (m *kubeGenericRuntimeManager) KillPod(ctx context.Context, pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) error {
	result := m.killPodWithSyncResult(ctx, pod, runningPod, gracePeriodOverride)
	return result.Error()
}

// killPodWithSyncResult kills a runningPod and returns SyncResult.
// pod must not be nil. For a pod the kubelet has no spec for (e.g. an orphaned
// pod found after a kubelet restart), callers should pass runningPod.ToAPIPod().
func (m *kubeGenericRuntimeManager) killPodWithSyncResult(ctx context.Context, pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) (result kubecontainer.PodSyncResult) {
	logger := klog.FromContext(ctx)
	result.AddSyncResult(m.killRunningPodContainers(ctx, pod, runningPod, gracePeriodOverride)...)

	// stop sandbox, the sandbox will be removed in GarbageCollect
	killSandboxResult := kubecontainer.NewSyncResult(kubecontainer.KillPodSandbox, runningPod.ID)
	result.AddSyncResult(killSandboxResult)
	// Stop all sandboxes belongs to same pod
	for _, podSandbox := range runningPod.Sandboxes {
		if err := m.runtimeService.StopPodSandbox(ctx, podSandbox.ID.ID); err != nil && !crierror.IsNotFound(err) {
			killSandboxResult.Fail(kubecontainer.ErrKillPodSandbox, err.Error())
			logger.Error(nil, "Failed to stop sandbox", "podSandboxID", podSandbox.ID)
		}
	}

	return
}

func (m *kubeGenericRuntimeManager) GeneratePodStatus(ctx context.Context, event *runtimeapi.ContainerEventResponse) *kubecontainer.PodStatus {
	podUID := kubetypes.UID(event.PodSandboxStatus.Metadata.Uid)
	podIPs := m.determinePodSandboxIPs(ctx, event.PodSandboxStatus.Metadata.Namespace, event.PodSandboxStatus.Metadata.Name, event.PodSandboxStatus)

	kubeContainerStatuses := []*kubecontainer.Status{}
	for _, status := range event.ContainersStatuses {
		kubeContainerStatuses = append(kubeContainerStatuses, m.convertToKubeContainerStatus(ctx, podUID, status))
	}

	sort.Sort(containerStatusByCreated(kubeContainerStatuses))

	return &kubecontainer.PodStatus{
		ID:                kubetypes.UID(event.PodSandboxStatus.Metadata.Uid),
		Name:              event.PodSandboxStatus.Metadata.Name,
		Namespace:         event.PodSandboxStatus.Metadata.Namespace,
		IPs:               podIPs,
		SandboxStatuses:   []*runtimeapi.PodSandboxStatus{event.PodSandboxStatus},
		ContainerStatuses: kubeContainerStatuses,
	}
}

// GetPodStatus retrieves the status of the pod, including the
// information of all containers in the pod that are visible in Runtime.
func (m *kubeGenericRuntimeManager) GetPodStatus(ctx context.Context, pod *kubecontainer.Pod) (*kubecontainer.PodStatus, error) {
	logger := klog.FromContext(ctx)
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
	podFullName := format.PodDesc(pod.Name, pod.Namespace, pod.ID)
	logger = logger.WithValues("pod", podFullName)
	ctx = klog.NewContext(ctx, logger)

	podSandboxIDs := make([]string, len(pod.Sandboxes))
	for i, s := range pod.Sandboxes {
		podSandboxIDs[i] = s.ID.ID
	}

	sandboxStatuses := []*runtimeapi.PodSandboxStatus{}
	containerStatuses := []*kubecontainer.Status{}
	activeContainerStatuses := []*kubecontainer.Status{}

	// Since we rely on the sandbox & container IDs in the kubecontainer.Pod, we must be
	// conservative and use it's timestamp as the status timestamp. Otherwise, we risk reporting a
	// newer PodStatus timestamp that is missing Sandboxes or Containers that should have been
	// present at that time.
	timestamp := pod.Timestamp

	podIPs := []string{}
	var activePodSandboxID string
	for idx, podSandboxID := range podSandboxIDs {
		resp, err := m.runtimeService.PodSandboxStatus(ctx, podSandboxID, false)
		// Between List (ListPodSandbox) and check (PodSandboxStatus) another thread might remove a container, and that is normal.
		// The previous call (getSandboxIDByPodUID) never fails due to a pod sandbox not existing.
		// Therefore, this method should not either, but instead act as if the previous call failed,
		// which means the error should be ignored.
		if crierror.IsNotFound(err) {
			continue
		}
		if err != nil {
			logger.Error(err, "PodSandboxStatus of sandbox for pod", "podSandboxID", podSandboxID)
			return nil, err
		}
		if resp.GetStatus() == nil {
			return nil, errors.New("pod sandbox status is nil")

		}
		sandboxStatuses = append(sandboxStatuses, resp.Status)
		// Only get pod IP from latest sandbox
		if idx == 0 && resp.Status.State == runtimeapi.PodSandboxState_SANDBOX_READY {
			podIPs = m.determinePodSandboxIPs(ctx, pod.Namespace, pod.Name, resp.Status)
			activePodSandboxID = podSandboxID
		}

		if idx == 0 && utilfeature.DefaultFeatureGate.Enabled(features.EventedPLEG) {
			if resp.Timestamp == 0 {
				// If the Evented PLEG is enabled in the kubelet, but not in the runtime
				// then the pod status we get will not have the timestamp set.
				// e.g. CI job 'pull-kubernetes-e2e-gce-alpha-features' will runs with
				// features gate enabled, which includes Evented PLEG, but uses the
				// runtime without Evented PLEG support.
				logger.V(4).Info("Runtime does not set pod status timestamp")
				containerStatuses, activeContainerStatuses, err = m.getPodContainerStatuses(ctx, pod, activePodSandboxID)
				if err != nil {
					if m.logReduction.ShouldMessageBePrinted(err.Error(), podFullName) {
						logger.Error(err, "getPodContainerStatuses for pod failed")
					}
					return nil, err
				}
			} else {
				// Get the statuses of all containers visible to the pod and
				// timestamp from sandboxStatus.
				timestamp = time.Unix(0, resp.Timestamp)
				for _, cs := range resp.ContainersStatuses {
					cStatus := m.convertToKubeContainerStatus(ctx, pod.ID, cs)
					containerStatuses = append(containerStatuses, cStatus)
				}
			}
		}
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.EventedPLEG) {
		// Get statuses of all containers visible in the pod.
		var err error
		containerStatuses, activeContainerStatuses, err = m.getPodContainerStatuses(ctx, pod, activePodSandboxID)
		if err != nil {
			if m.logReduction.ShouldMessageBePrinted(err.Error(), podFullName) {
				logger.Error(err, "getPodContainerStatuses for pod failed")
			}
			return nil, err
		}
	}

	m.logReduction.ClearID(podFullName)
	return &kubecontainer.PodStatus{
		ID:                      pod.ID,
		Name:                    pod.Name,
		Namespace:               pod.Namespace,
		IPs:                     podIPs,
		SandboxStatuses:         sandboxStatuses,
		ContainerStatuses:       containerStatuses,
		ActiveContainerStatuses: activeContainerStatuses,
		TimeStamp:               timestamp,
	}, nil
}

func (m *kubeGenericRuntimeManager) GetContainerStatus(ctx context.Context, podUID kubetypes.UID, id kubecontainer.ContainerID) (*kubecontainer.Status, error) {
	resp, err := m.runtimeService.ContainerStatus(ctx, id.ID, false)
	if err != nil {
		return nil, fmt.Errorf("runtime container status: %w", err)
	}

	status := resp.GetStatus()
	if status == nil {
		return nil, remote.ErrContainerStatusNil
	}

	return m.convertToKubeContainerStatus(ctx, podUID, status), nil
}

// GarbageCollect removes dead containers using the specified container gc policy.
func (m *kubeGenericRuntimeManager) GarbageCollect(ctx context.Context, gcPolicy kubecontainer.GCPolicy, allSourcesReady bool, evictNonDeletedPods bool) error {
	logger := klog.FromContext(ctx)
	// Remove terminated pods from the actuated state once all their volumes are unmounted.
	// If a pod has been removed from podManager while its inline volumes are still mounted on
	// disk, keeping its actuated state entry until VolumeManager unmounts those volumes ensures
	// reconstructVolume still has the pod metadata and volume spec if the kubelet restarts before
	// unmount completes.
	for _, uid := range m.actuatedState.GetPodUIDs() {
		if !m.podStateProvider.ShouldPodContentBeRemoved(uid) {
			continue
		}
		vols, _ := m.actuatedState.GetPodVolumes(uid)
		needsUnmount := slices.ContainsFunc(vols, reconciler.VolumeRequiresUnmount)
		if needsUnmount {
			continue
		}
		if err := m.actuatedState.RemovePod(logger, uid); err != nil {
			// No need to act on the error beyond logging it here.
			logger.Error(err, "Failed to remove pod from actuated state", "podUID", uid)
		}
	}

	return m.containerGC.GarbageCollect(ctx, gcPolicy, allSourcesReady, evictNonDeletedPods)
}

// UpdatePodCIDR is just a passthrough method to update the runtimeConfig of the shim
// with the podCIDR supplied by the kubelet.
func (m *kubeGenericRuntimeManager) UpdatePodCIDR(ctx context.Context, podCIDR string) error {
	logger := klog.FromContext(ctx)
	// TODO(#35531): do we really want to write a method on this manager for each
	// field of the config?
	logger.Info("Updating runtime config through cri with podcidr", "CIDR", podCIDR)
	return m.runtimeService.UpdateRuntimeConfig(ctx,
		&runtimeapi.RuntimeConfig{
			NetworkConfig: &runtimeapi.NetworkConfig{
				PodCidr: podCIDR,
			},
		})
}

func (m *kubeGenericRuntimeManager) CheckpointContainer(ctx context.Context, options *runtimeapi.CheckpointContainerRequest) error {
	return m.runtimeService.CheckpointContainer(ctx, options)
}

func (m *kubeGenericRuntimeManager) CheckpointPod(ctx context.Context, options *runtimeapi.CheckpointPodRequest) error {
	return m.runtimeService.CheckpointPod(ctx, options)
}

func (m *kubeGenericRuntimeManager) ListMetricDescriptors(ctx context.Context) ([]*runtimeapi.MetricDescriptor, error) {
	return m.runtimeService.ListMetricDescriptors(ctx)
}

func (m *kubeGenericRuntimeManager) ListPodSandboxMetrics(ctx context.Context) ([]*runtimeapi.PodSandboxMetrics, error) {
	return m.runtimeService.ListPodSandboxMetrics(ctx)
}

func (m *kubeGenericRuntimeManager) UpdateActuatedPodLevelResources(logger klog.Logger, actuatedPod *v1.Pod) error {
	return m.actuatedState.SetPodLevelResources(logger, actuatedPod.UID, actuatedPodLevelResources(actuatedPod))
}

// isPodResizeInProgress checks whether the actuated resizable resources differ from
// the resources allocated for:
// * any running containers - Specifically, the following differences are ignored:
// - Non-resizable containers: non-restartable init containers, ephemeral containers
// - Non-resizable resources: only CPU & memory are resizable
// - Non-running containers: they will be sized correctly when (re)started
// * any running containers removed from the pod spec, which run with resources that are no longer
// allocated.
// * the pod cgroup.
// * any running pod if InPlacePodVerticalScalingMemoryBackedVolumes is enabled.
func (m *kubeGenericRuntimeManager) IsPodResizeInProgress(allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	if m.isContainerResourceResizeInProgress(allocatedPod, podStatus) {
		return true
	}

	if m.isMemoryBackedVolumeResizeInProgress(allocatedPod) {
		return true
	}

	return m.isPodLevelResourcesResizeInProgress(allocatedPod, podStatus)
}

func (m *kubeGenericRuntimeManager) isMemoryBackedVolumeResizeInProgress(allocatedPod *v1.Pod) bool {
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
		for _, volume := range allocatedPod.Spec.Volumes {
			if !allocation.VolHasMemoryBackedEmptyDir(&volume) {
				continue
			}

			allocatedLimit, _ := m.runtimeHelper.EffectiveEphemeralVolumeLimit(allocatedPod, volume.Name)
			actuatedLimit, found := m.actuatedState.GetEmptyDirVolumeLimit(allocatedPod.UID, volume.Name)
			if !found {
				// No actuation checkpoint yet.
				continue
			}

			hasDesiredLimit := allocatedLimit != nil && !allocatedLimit.IsZero()
			hasActuatedLimit := actuatedLimit != nil && !actuatedLimit.IsZero()
			if !hasDesiredLimit && !hasActuatedLimit {
				// nil and 0 are considered equivalent for sizeLimit
				continue
			}
			if hasDesiredLimit != hasActuatedLimit || !allocatedLimit.Equal(*actuatedLimit) {
				return true
			}
		}
	}
	return false
}

func (m *kubeGenericRuntimeManager) isContainerResourceResizeInProgress(allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	if m.isRemovedContainerRunningWithResources(allocatedPod, podStatus) {
		return true
	}
	return !podutil.VisitContainers(&allocatedPod.Spec, podutil.InitContainers|podutil.Containers,
		func(allocatedContainer *v1.Container, containerType podutil.ContainerType) (shouldContinue bool) {
			if !isResizableContainer(allocatedContainer, containerType) {
				return true
			}

			containerStatus := podStatus.FindContainerStatusByName(allocatedContainer.Name)
			if containerStatus == nil || containerStatus.State != kubecontainer.ContainerStateRunning {
				// If the container isn't running, it doesn't need to be resized.
				return true
			}

			actuatedResources, _ := m.actuatedState.GetContainerResources(allocatedPod.UID, allocatedContainer.Name)
			allocatedResources := allocatedContainer.Resources
			if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) {
				allocatedResources.Limits = kubeutil.GetLimits(&kubeutil.ResourceOpts{PodResources: allocatedPod.Spec.Resources, ContainerResources: &allocatedContainer.Resources})
			}

			if utilfeature.DefaultFeatureGate.Enabled(features.WorkloadControlledSwap) {
				// Compare effective swap, not raw map entries: a container without
				// limits.swap derives its value from the pod-level limit or LimitedSwap.
				allocatedSwap := m.effectiveSwapLimit(allocatedPod, allocatedContainer)
				actuatedSwap := m.actuatedSwapLimit(allocatedPod, allocatedContainer, &actuatedResources)
				if allocatedSwap != actuatedSwap {
					return false
				}
			}

			return allocatedResources.Requests[v1.ResourceCPU].Equal(actuatedResources.Requests[v1.ResourceCPU]) &&
				allocatedResources.Limits[v1.ResourceCPU].Equal(actuatedResources.Limits[v1.ResourceCPU]) &&
				allocatedResources.Requests[v1.ResourceMemory].Equal(actuatedResources.Requests[v1.ResourceMemory]) &&
				allocatedResources.Limits[v1.ResourceMemory].Equal(actuatedResources.Limits[v1.ResourceMemory])
		})
}

// isRemovedContainerRunningWithResources returns whether a container removed from the pod spec is
// still running with CPU or memory resources, which the pod no longer allocates.
func (m *kubeGenericRuntimeManager) isRemovedContainerRunningWithResources(allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	if podStatus == nil {
		return false
	}
	for _, cs := range podStatus.ContainerStatuses {
		if cs.State != kubecontainer.ContainerStateRunning || kubecontainer.GetContainerSpec(allocatedPod, cs.Name) != nil {
			continue
		}
		resources, ok := m.actuatedState.GetContainerResources(allocatedPod.UID, cs.Name)
		if !ok {
			continue
		}
		for _, list := range []v1.ResourceList{resources.Requests, resources.Limits} {
			if !list.Cpu().IsZero() || !list.Memory().IsZero() {
				return true
			}
		}
	}
	return false
}

func (m *kubeGenericRuntimeManager) isPodLevelResourcesResizeInProgress(allocatedPod *v1.Pod, podStatus *kubecontainer.PodStatus) bool {
	if resizable, _, _ := allocation.IsInPlacePodVerticalScalingAllowed(allocatedPod); !resizable {
		return false
	}

	// Without running containers, the pod cgroup doesn't need to be resized.
	if podStatus == nil || len(podStatus.GetRunningContainerStatuses()) == 0 {
		return false
	}

	actuatedPodResources, found := m.actuatedState.GetPodLevelResources(allocatedPod.UID)
	if !found || actuatedPodResources == nil {
		return false
	}

	if podCgroupResourcesToUpdate(cm.EffectivePodResources(allocatedPod), actuatedPodResources).Len() > 0 {
		return true
	}
	// Pod-level swap is only actuated when the node manages swap; comparing it
	// otherwise would leave the resize in progress forever.
	return m.workloadSwapLimitsEnabled() && !podSwapLimitEqual(actuatedPodResources, allocatedPod.Spec.Resources)
}

func isResizableContainer(container *v1.Container, containerType podutil.ContainerType) bool {
	switch containerType {
	case podutil.InitContainers:
		return podutil.IsRestartableInitContainer(container)
	case podutil.Containers:
		return true
	default:
		return false
	}
}
