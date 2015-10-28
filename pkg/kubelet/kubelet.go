/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

// Note: if you change code in this file, you might need to change code in
// contrib/mesos/pkg/executor/.

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/envvars"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/rkt"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubeletTypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeletUtil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utilErrors "k8s.io/kubernetes/pkg/util/errors"
	kubeio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/procfs"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/third_party/golang/expansion"
)

const (
	// Max amount of time to wait for the container runtime to come up.
	maxWaitForContainerRuntime = 5 * time.Minute

	// nodeStatusUpdateRetry specifies how many times kubelet retries when posting node status failed.
	nodeStatusUpdateRetry = 5

	// Location of container logs.
	containerLogsDir = "/var/log/containers"

	// max backoff period
	maxContainerBackOff = 300 * time.Second

	// Capacity of the channel for storing pods to kill. A small number should
	// suffice because a goroutine is dedicated to check the channel and does
	// not block on anything else.
	podKillingChannelCapacity = 50

	// system default DNS resolver configuration
	ResolvConfDefault = "/etc/resolv.conf"

	// Minimum period for performing global cleanup tasks, i.e., housekeeping
	// will not be performed more than once per housekeepingMinimumPeriod.
	housekeepingMinimumPeriod = time.Second * 2

	etcHostsPath = "/etc/hosts"
)

var (
	// ErrContainerNotFound returned when a container in the given pod with the
	// given container name was not found, amongst those managed by the kubelet.
	ErrContainerNotFound = errors.New("no matching container")
)

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {
	HandlePodAdditions(pods []*api.Pod)
	HandlePodUpdates(pods []*api.Pod)
	HandlePodDeletions(pods []*api.Pod)
	HandlePodSyncs(pods []*api.Pod)
	HandlePodCleanups() error
}

type SourcesReadyFn func() bool

// Wait for the container runtime to be up with a timeout.
func waitUntilRuntimeIsUp(cr kubecontainer.Runtime, timeout time.Duration) error {
	var err error = nil
	waitStart := time.Now()
	for time.Since(waitStart) < timeout {
		_, err = cr.Version()
		if err == nil {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return err
}

// New creates a new Kubelet for use in main
func NewMainKubelet(
	hostname string,
	nodeName string,
	dockerClient dockertools.DockerInterface,
	kubeClient client.Interface,
	rootDirectory string,
	podInfraContainerImage string,
	resyncInterval time.Duration,
	pullQPS float32,
	pullBurst int,
	eventQPS float32,
	eventBurst int,
	containerGCPolicy ContainerGCPolicy,
	sourcesReady SourcesReadyFn,
	registerNode bool,
	standaloneMode bool,
	clusterDomain string,
	clusterDNS net.IP,
	masterServiceNamespace string,
	volumePlugins []volume.VolumePlugin,
	networkPlugins []network.NetworkPlugin,
	networkPluginName string,
	streamingConnectionIdleTimeout time.Duration,
	recorder record.EventRecorder,
	cadvisorInterface cadvisor.Interface,
	imageGCPolicy ImageGCPolicy,
	diskSpacePolicy DiskSpacePolicy,
	cloud cloudprovider.Interface,
	nodeStatusUpdateFrequency time.Duration,
	resourceContainer string,
	osInterface kubecontainer.OSInterface,
	cgroupRoot string,
	containerRuntime string,
	rktPath string,
	rktStage1Image string,
	mounter mount.Interface,
	writer kubeio.Writer,
	dockerDaemonContainer string,
	systemContainer string,
	configureCBR0 bool,
	podCIDR string,
	pods int,
	dockerExecHandler dockertools.ExecHandler,
	resolverConfig string,
	cpuCFSQuota bool,
	daemonEndpoints *api.NodeDaemonEndpoints,
	serializeImagePulls bool,
) (*Kubelet, error) {
	if rootDirectory == "" {
		return nil, fmt.Errorf("invalid root directory %q", rootDirectory)
	}
	if resyncInterval <= 0 {
		return nil, fmt.Errorf("invalid sync frequency %d", resyncInterval)
	}
	if systemContainer != "" && cgroupRoot == "" {
		return nil, fmt.Errorf("invalid configuration: system container was specified and cgroup root was not specified")
	}
	dockerClient = dockertools.NewInstrumentedDockerInterface(dockerClient)

	serviceStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	if kubeClient != nil {
		// TODO: cache.NewListWatchFromClient is limited as it takes a client implementation rather
		// than an interface. There is no way to construct a list+watcher using resource name.
		listWatch := &cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.Services(api.NamespaceAll).List(labels.Everything())
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.Services(api.NamespaceAll).Watch(labels.Everything(), fields.Everything(), resourceVersion)
			},
		}
		cache.NewReflector(listWatch, &api.Service{}, serviceStore, 0).Run()
	}
	serviceLister := &cache.StoreToServiceLister{Store: serviceStore}

	nodeStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	if kubeClient != nil {
		// TODO: cache.NewListWatchFromClient is limited as it takes a client implementation rather
		// than an interface. There is no way to construct a list+watcher using resource name.
		fieldSelector := fields.Set{client.ObjectNameField: nodeName}.AsSelector()
		listWatch := &cache.ListWatch{
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.Nodes().List(labels.Everything(), fieldSelector)
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.Nodes().Watch(labels.Everything(), fieldSelector, resourceVersion)
			},
		}
		cache.NewReflector(listWatch, &api.Node{}, nodeStore, 0).Run()
	}
	nodeLister := &cache.StoreToNodeLister{Store: nodeStore}

	// TODO: get the real node object of ourself,
	// and use the real node name and UID.
	// TODO: what is namespace for node?
	nodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeName),
		Namespace: "",
	}

	containerGC, err := newContainerGC(dockerClient, containerGCPolicy)
	if err != nil {
		return nil, err
	}
	imageManager, err := newImageManager(dockerClient, cadvisorInterface, recorder, nodeRef, imageGCPolicy)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize image manager: %v", err)
	}
	diskSpaceManager, err := newDiskSpaceManager(cadvisorInterface, diskSpacePolicy)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize disk manager: %v", err)
	}
	statusManager := status.NewManager(kubeClient)
	readinessManager := kubecontainer.NewReadinessManager()
	containerRefManager := kubecontainer.NewRefManager()

	volumeManager := newVolumeManager()

	oomWatcher := NewOOMWatcher(cadvisorInterface, recorder)

	klet := &Kubelet{
		hostname:                       hostname,
		nodeName:                       nodeName,
		dockerClient:                   dockerClient,
		kubeClient:                     kubeClient,
		rootDirectory:                  rootDirectory,
		resyncInterval:                 resyncInterval,
		containerRefManager:            containerRefManager,
		readinessManager:               readinessManager,
		httpClient:                     &http.Client{},
		sourcesReady:                   sourcesReady,
		registerNode:                   registerNode,
		standaloneMode:                 standaloneMode,
		clusterDomain:                  clusterDomain,
		clusterDNS:                     clusterDNS,
		serviceLister:                  serviceLister,
		nodeLister:                     nodeLister,
		runtimeMutex:                   sync.Mutex{},
		runtimeUpThreshold:             maxWaitForContainerRuntime,
		lastTimestampRuntimeUp:         time.Time{},
		masterServiceNamespace:         masterServiceNamespace,
		streamingConnectionIdleTimeout: streamingConnectionIdleTimeout,
		recorder:                       recorder,
		cadvisor:                       cadvisorInterface,
		containerGC:                    containerGC,
		imageManager:                   imageManager,
		diskSpaceManager:               diskSpaceManager,
		statusManager:                  statusManager,
		volumeManager:                  volumeManager,
		cloud:                          cloud,
		nodeRef:                        nodeRef,
		nodeStatusUpdateFrequency:      nodeStatusUpdateFrequency,
		resourceContainer:              resourceContainer,
		os:                             osInterface,
		oomWatcher:                     oomWatcher,
		cgroupRoot:                     cgroupRoot,
		mounter:                        mounter,
		writer:                         writer,
		configureCBR0:                  configureCBR0,
		podCIDR:                        podCIDR,
		pods:                           pods,
		syncLoopMonitor:                util.AtomicValue{},
		resolverConfig:                 resolverConfig,
		cpuCFSQuota:                    cpuCFSQuota,
		daemonEndpoints:                daemonEndpoints,
	}

	if plug, err := network.InitNetworkPlugin(networkPlugins, networkPluginName, &networkHost{klet}); err != nil {
		return nil, err
	} else {
		klet.networkPlugin = plug
	}

	machineInfo, err := klet.GetCachedMachineInfo()
	if err != nil {
		return nil, err
	}

	oomAdjuster := oom.NewOomAdjuster()
	procFs := procfs.NewProcFs()

	// Initialize the runtime.
	switch containerRuntime {
	case "docker":
		// Only supported one for now, continue.
		klet.containerRuntime = dockertools.NewDockerManager(
			dockerClient,
			recorder,
			readinessManager,
			containerRefManager,
			machineInfo,
			podInfraContainerImage,
			pullQPS,
			pullBurst,
			containerLogsDir,
			osInterface,
			klet.networkPlugin,
			klet,
			klet.httpClient,
			dockerExecHandler,
			oomAdjuster,
			procFs,
			klet.cpuCFSQuota,
			serializeImagePulls,
		)

	case "rkt":
		conf := &rkt.Config{
			Path:               rktPath,
			Stage1Image:        rktStage1Image,
			InsecureSkipVerify: true,
		}
		rktRuntime, err := rkt.New(
			conf,
			klet,
			recorder,
			containerRefManager,
			readinessManager,
			klet.volumeManager,
			serializeImagePulls,
		)
		if err != nil {
			return nil, err
		}
		klet.containerRuntime = rktRuntime

		// No Docker daemon to put in a container.
		dockerDaemonContainer = ""
	default:
		return nil, fmt.Errorf("unsupported container runtime %q specified", containerRuntime)
	}

	// Setup container manager, can fail if the devices hierarchy is not mounted
	// (it is required by Docker however).
	containerManager, err := newContainerManager(mounter, cadvisorInterface, dockerDaemonContainer, systemContainer, resourceContainer)
	if err != nil {
		return nil, fmt.Errorf("failed to create the Container Manager: %v", err)
	}
	klet.containerManager = containerManager

	go util.Until(klet.syncNetworkStatus, 30*time.Second, util.NeverStop)
	if klet.kubeClient != nil {
		// Start syncing node status immediately, this may set up things the runtime needs to run.
		go util.Until(klet.syncNodeStatus, klet.nodeStatusUpdateFrequency, util.NeverStop)
	}

	// Wait for the runtime to be up with a timeout.
	if err := waitUntilRuntimeIsUp(klet.containerRuntime, maxWaitForContainerRuntime); err != nil {
		return nil, fmt.Errorf("timed out waiting for %q to come up: %v", containerRuntime, err)
	}
	klet.lastTimestampRuntimeUp = time.Now()

	klet.runner = klet.containerRuntime
	klet.podManager = newBasicPodManager(klet.kubeClient)

	runtimeCache, err := kubecontainer.NewRuntimeCache(klet.containerRuntime)
	if err != nil {
		return nil, err
	}
	klet.runtimeCache = runtimeCache
	klet.podWorkers = newPodWorkers(runtimeCache, klet.syncPod, recorder)

	metrics.Register(runtimeCache)

	if err = klet.setupDataDirs(); err != nil {
		return nil, err
	}
	if err = klet.volumePluginMgr.InitPlugins(volumePlugins, &volumeHost{klet}); err != nil {
		return nil, err
	}

	// If the container logs directory does not exist, create it.
	if _, err := os.Stat(containerLogsDir); err != nil {
		if err := osInterface.Mkdir(containerLogsDir, 0755); err != nil {
			glog.Errorf("Failed to create directory %q: %v", containerLogsDir, err)
		}
	}

	klet.backOff = util.NewBackOff(resyncInterval, maxContainerBackOff)
	klet.podKillingCh = make(chan *kubecontainer.Pod, podKillingChannelCapacity)

	return klet, nil
}

type serviceLister interface {
	List() (api.ServiceList, error)
}

type nodeLister interface {
	List() (machines api.NodeList, err error)
	GetNodeInfo(id string) (*api.Node, error)
}

// Kubelet is the main kubelet implementation.
type Kubelet struct {
	hostname       string
	nodeName       string
	dockerClient   dockertools.DockerInterface
	runtimeCache   kubecontainer.RuntimeCache
	kubeClient     client.Interface
	rootDirectory  string
	podWorkers     PodWorkers
	resyncInterval time.Duration
	resyncTicker   *time.Ticker
	sourcesReady   SourcesReadyFn

	podManager podManager

	// Needed to report events for containers belonging to deleted/modified pods.
	// Tracks references for reporting events
	containerRefManager *kubecontainer.RefManager

	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
	// Optional, defaults to simple Docker implementation
	runner kubecontainer.ContainerCommandRunner
	// Optional, client for http requests, defaults to empty client
	httpClient kubeletTypes.HttpGetter

	// cAdvisor used for container information.
	cadvisor cadvisor.Interface

	// Set to true to have the node register itself with the apiserver.
	registerNode bool
	// for internal book keeping; access only from within registerWithApiserver
	registrationCompleted bool

	// Set to true if the kubelet is in standalone mode (i.e. setup without an apiserver)
	standaloneMode bool

	// If non-empty, use this for container DNS search.
	clusterDomain string

	// If non-nil, use this for container DNS server.
	clusterDNS net.IP

	masterServiceNamespace string
	serviceLister          serviceLister
	nodeLister             nodeLister

	// Last timestamp when runtime responded on ping.
	// Mutex is used to protect this value.
	runtimeMutex           sync.Mutex
	runtimeUpThreshold     time.Duration
	lastTimestampRuntimeUp time.Time

	// Network Status information
	networkConfigMutex sync.Mutex
	networkConfigured  bool

	// Volume plugins.
	volumePluginMgr volume.VolumePluginMgr

	// Network plugin.
	networkPlugin network.NetworkPlugin

	// Container readiness state manager.
	readinessManager *kubecontainer.ReadinessManager

	// How long to keep idle streaming command execution/port forwarding
	// connections open before terminating them
	streamingConnectionIdleTimeout time.Duration

	// The EventRecorder to use
	recorder record.EventRecorder

	// Policy for handling garbage collection of dead containers.
	containerGC containerGC

	// Manager for images.
	imageManager imageManager

	// Diskspace manager.
	diskSpaceManager diskSpaceManager

	// Cached MachineInfo returned by cadvisor.
	machineInfo *cadvisorApi.MachineInfo

	// Syncs pods statuses with apiserver; also used as a cache of statuses.
	statusManager status.Manager

	// Manager for the volume maps for the pods.
	volumeManager *volumeManager

	//Cloud provider interface
	cloud cloudprovider.Interface

	// Reference to this node.
	nodeRef *api.ObjectReference

	// Container runtime.
	containerRuntime kubecontainer.Runtime

	// nodeStatusUpdateFrequency specifies how often kubelet posts node status to master.
	// Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod
	// in nodecontroller. There are several constraints:
	// 1. nodeMonitorGracePeriod must be N times more than nodeStatusUpdateFrequency, where
	//    N means number of retries allowed for kubelet to post node status. It is pointless
	//    to make nodeMonitorGracePeriod be less than nodeStatusUpdateFrequency, since there
	//    will only be fresh values from Kubelet at an interval of nodeStatusUpdateFrequency.
	//    The constant must be less than podEvictionTimeout.
	// 2. nodeStatusUpdateFrequency needs to be large enough for kubelet to generate node
	//    status. Kubelet may fail to update node status reliably if the value is too small,
	//    as it takes time to gather all necessary node information.
	nodeStatusUpdateFrequency time.Duration

	// The name of the resource-only container to run the Kubelet in (empty for no container).
	// Name must be absolute.
	resourceContainer string

	os kubecontainer.OSInterface

	// Watcher of out of memory events.
	oomWatcher OOMWatcher

	// If non-empty, pass this to the container runtime as the root cgroup.
	cgroupRoot string

	// Mounter to use for volumes.
	mounter mount.Interface

	// Writer interface to use for volumes.
	writer kubeio.Writer

	// Manager of non-Runtime containers.
	containerManager containerManager

	// Whether or not kubelet should take responsibility for keeping cbr0 in
	// the correct state.
	configureCBR0 bool
	podCIDR       string

	// Number of Pods which can be run by this Kubelet
	pods int

	// Monitor Kubelet's sync loop
	syncLoopMonitor util.AtomicValue

	// Container restart Backoff
	backOff *util.Backoff

	// Channel for sending pods to kill.
	podKillingCh chan *kubecontainer.Pod

	// The configuration file used as the base to generate the container's
	// DNS resolver configuration file. This can be used in conjunction with
	// clusterDomain and clusterDNS.
	resolverConfig string

	// Optionally shape the bandwidth of a pod
	shaper bandwidth.BandwidthShaper

	// True if container cpu limits should be enforced via cgroup CFS quota
	cpuCFSQuota bool

	// Information about the ports which are opened by daemons on Node running this Kubelet server.
	daemonEndpoints *api.NodeDaemonEndpoints
}

// getRootDir returns the full path to the directory under which kubelet can
// store data.  These functions are useful to pass interfaces to other modules
// that may need to know where to write data without getting a whole kubelet
// instance.
func (kl *Kubelet) getRootDir() string {
	return kl.rootDirectory
}

// getPodsDir returns the full path to the directory under which pod
// directories are created.
func (kl *Kubelet) getPodsDir() string {
	return path.Join(kl.getRootDir(), "pods")
}

// getPluginsDir returns the full path to the directory under which plugin
// directories are created.  Plugins can use these directories for data that
// they need to persist.  Plugins should create subdirectories under this named
// after their own names.
func (kl *Kubelet) getPluginsDir() string {
	return path.Join(kl.getRootDir(), "plugins")
}

// getPluginDir returns a data directory name for a given plugin name.
// Plugins can use these directories to store data that they need to persist.
// For per-pod plugin data, see getPodPluginDir.
func (kl *Kubelet) getPluginDir(pluginName string) string {
	return path.Join(kl.getPluginsDir(), pluginName)
}

// getPodDir returns the full path to the per-pod data directory for the
// specified pod.  This directory may not exist if the pod does not exist.
func (kl *Kubelet) getPodDir(podUID types.UID) string {
	// Backwards compat.  The "old" stuff should be removed before 1.0
	// release.  The thinking here is this:
	//     !old && !new = use new
	//     !old && new  = use new
	//     old && !new  = use old
	//     old && new   = use new (but warn)
	oldPath := path.Join(kl.getRootDir(), string(podUID))
	oldExists := dirExists(oldPath)
	newPath := path.Join(kl.getPodsDir(), string(podUID))
	newExists := dirExists(newPath)
	if oldExists && !newExists {
		return oldPath
	}
	if oldExists {
		glog.Warningf("Data dir for pod %q exists in both old and new form, using new", podUID)
	}
	return newPath
}

// getPodVolumesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumesDir(podUID types.UID) string {
	return path.Join(kl.getPodDir(podUID), "volumes")
}

// getPodVolumeDir returns the full path to the directory which represents the
// named volume under the named plugin for specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return path.Join(kl.getPodVolumesDir(podUID), pluginName, volumeName)
}

// getPodPluginsDir returns the full path to the per-pod data directory under
// which plugins may store data for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) getPodPluginsDir(podUID types.UID) string {
	return path.Join(kl.getPodDir(podUID), "plugins")
}

// getPodPluginDir returns a data directory name for a given plugin name for a
// given pod UID.  Plugins can use these directories to store data that they
// need to persist.  For non-per-pod plugin data, see getPluginDir.
func (kl *Kubelet) getPodPluginDir(podUID types.UID, pluginName string) string {
	return path.Join(kl.getPodPluginsDir(podUID), pluginName)
}

// getPodContainerDir returns the full path to the per-pod data directory under
// which container data is held for the specified pod.  This directory may not
// exist if the pod or container does not exist.
func (kl *Kubelet) getPodContainerDir(podUID types.UID, ctrName string) string {
	// Backwards compat.  The "old" stuff should be removed before 1.0
	// release.  The thinking here is this:
	//     !old && !new = use new
	//     !old && new  = use new
	//     old && !new  = use old
	//     old && new   = use new (but warn)
	oldPath := path.Join(kl.getPodDir(podUID), ctrName)
	oldExists := dirExists(oldPath)
	newPath := path.Join(kl.getPodDir(podUID), "containers", ctrName)
	newExists := dirExists(newPath)
	if oldExists && !newExists {
		return oldPath
	}
	if oldExists {
		glog.Warningf("Data dir for pod %q, container %q exists in both old and new form, using new", podUID, ctrName)
	}
	return newPath
}

func dirExists(path string) bool {
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return s.IsDir()
}

func (kl *Kubelet) setupDataDirs() error {
	kl.rootDirectory = path.Clean(kl.rootDirectory)
	if err := os.MkdirAll(kl.getRootDir(), 0750); err != nil {
		return fmt.Errorf("error creating root directory: %v", err)
	}
	if err := os.MkdirAll(kl.getPodsDir(), 0750); err != nil {
		return fmt.Errorf("error creating pods directory: %v", err)
	}
	if err := os.MkdirAll(kl.getPluginsDir(), 0750); err != nil {
		return fmt.Errorf("error creating plugins directory: %v", err)
	}
	return nil
}

// Get a list of pods that have data directories.
func (kl *Kubelet) listPodsFromDisk() ([]types.UID, error) {
	podInfos, err := ioutil.ReadDir(kl.getPodsDir())
	if err != nil {
		return nil, err
	}
	pods := []types.UID{}
	for i := range podInfos {
		if podInfos[i].IsDir() {
			pods = append(pods, types.UID(podInfos[i].Name()))
		}
	}
	return pods, nil
}

func (kl *Kubelet) GetNode() (*api.Node, error) {
	if kl.standaloneMode {
		return nil, errors.New("no node entry for kubelet in standalone mode")
	}
	return kl.nodeLister.GetNodeInfo(kl.nodeName)
}

// Starts garbage collection threads.
func (kl *Kubelet) StartGarbageCollection() {
	go util.Until(func() {
		if err := kl.containerGC.GarbageCollect(); err != nil {
			glog.Errorf("Container garbage collection failed: %v", err)
		}
	}, time.Minute, util.NeverStop)

	go util.Until(func() {
		if err := kl.imageManager.GarbageCollect(); err != nil {
			glog.Errorf("Image garbage collection failed: %v", err)
		}
	}, 5*time.Minute, util.NeverStop)
}

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.kubeClient == nil {
		glog.Warning("No api server defined - no node status update will be sent.")
	}

	// Move Kubelet to a container.
	if kl.resourceContainer != "" {
		// Fixme: I need to reside inside ContainerManager interface.
		err := util.RunInResourceContainer(kl.resourceContainer)
		if err != nil {
			glog.Warningf("Failed to move Kubelet to container %q: %v", kl.resourceContainer, err)
		}
		glog.Infof("Running in container %q", kl.resourceContainer)
	}

	if err := kl.imageManager.Start(); err != nil {
		kl.recorder.Eventf(kl.nodeRef, "KubeletSetupFailed", "Failed to start ImageManager %v", err)
		glog.Errorf("Failed to start ImageManager, images may not be garbage collected: %v", err)
	}

	if err := kl.cadvisor.Start(); err != nil {
		kl.recorder.Eventf(kl.nodeRef, "KubeletSetupFailed", "Failed to start CAdvisor %v", err)
		glog.Errorf("Failed to start CAdvisor, system may not be properly monitored: %v", err)
	}

	if err := kl.containerManager.Start(); err != nil {
		kl.recorder.Eventf(kl.nodeRef, "KubeletSetupFailed", "Failed to start ContainerManager %v", err)
		glog.Errorf("Failed to start ContainerManager, system may not be properly isolated: %v", err)
	}

	if err := kl.oomWatcher.Start(kl.nodeRef); err != nil {
		kl.recorder.Eventf(kl.nodeRef, "KubeletSetupFailed", "Failed to start OOM watcher %v", err)
		glog.Errorf("Failed to start OOM watching: %v", err)
	}

	go util.Until(kl.updateRuntimeUp, 5*time.Second, util.NeverStop)

	// Start a goroutine responsible for killing pods (that are not properly
	// handled by pod workers).
	go util.Until(kl.podKiller, 1*time.Second, util.NeverStop)

	// Run the system oom watcher forever.
	kl.statusManager.Start()
	kl.syncLoop(updates, kl)
}

func (kl *Kubelet) initialNodeStatus() (*api.Node, error) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:   kl.nodeName,
			Labels: map[string]string{"kubernetes.io/hostname": kl.hostname},
		},
	}
	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return nil, fmt.Errorf("failed to get instances from cloud provider")
		}

		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO: ExternalID is deprecated, we'll have to drop this code
		externalID, err := instances.ExternalID(kl.nodeName)
		if err != nil {
			return nil, fmt.Errorf("failed to get external ID from cloud provider: %v", err)
		}
		node.Spec.ExternalID = externalID

		// TODO: We can't assume that the node has credentials to talk to the
		// cloudprovider from arbitrary nodes. At most, we should talk to a
		// local metadata server here.
		node.Spec.ProviderID, err = cloudprovider.GetInstanceProviderID(kl.cloud, kl.nodeName)
		if err != nil {
			return nil, err
		}
	} else {
		node.Spec.ExternalID = kl.hostname
	}
	if err := kl.setNodeStatus(node); err != nil {
		return nil, err
	}
	return node, nil
}

// registerWithApiserver registers the node with the cluster master. It is safe
// to call multiple times, but not concurrently (kl.registrationCompleted is
// not locked).
func (kl *Kubelet) registerWithApiserver() {
	if kl.registrationCompleted {
		return
	}
	step := 100 * time.Millisecond
	for {
		time.Sleep(step)
		step = step * 2
		if step >= 7*time.Second {
			step = 7 * time.Second
		}

		node, err := kl.initialNodeStatus()
		if err != nil {
			glog.Errorf("Unable to construct api.Node object for kubelet: %v", err)
			continue
		}
		glog.V(2).Infof("Attempting to register node %s", node.Name)
		if _, err := kl.kubeClient.Nodes().Create(node); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				glog.V(2).Infof("Unable to register %s with the apiserver: %v", node.Name, err)
				continue
			}
			currentNode, err := kl.kubeClient.Nodes().Get(kl.nodeName)
			if err != nil {
				glog.Errorf("error getting node %q: %v", kl.nodeName, err)
				continue
			}
			if currentNode == nil {
				glog.Errorf("no node instance returned for %q", kl.nodeName)
				continue
			}
			if currentNode.Spec.ExternalID == node.Spec.ExternalID {
				glog.Infof("Node %s was previously registered", node.Name)
				kl.registrationCompleted = true
				return
			}
			glog.Errorf(
				"Previously %q had externalID %q; now it is %q; will delete and recreate.",
				kl.nodeName, node.Spec.ExternalID, currentNode.Spec.ExternalID,
			)
			if err := kl.kubeClient.Nodes().Delete(node.Name); err != nil {
				glog.Errorf("Unable to delete old node: %v", err)
			} else {
				glog.Errorf("Deleted old node object %q", kl.nodeName)
			}
			continue
		}
		glog.Infof("Successfully registered node %s", node.Name)
		kl.registrationCompleted = true
		return
	}
}

// syncNodeStatus should be called periodically from a goroutine.
// It synchronizes node status to master, registering the kubelet first if
// necessary.
func (kl *Kubelet) syncNodeStatus() {
	if kl.kubeClient == nil {
		return
	}
	if kl.registerNode {
		// This will exit immediately if it doesn't need to do anything.
		kl.registerWithApiserver()
	}
	if err := kl.updateNodeStatus(); err != nil {
		glog.Errorf("Unable to update node status: %v", err)
	}
}

func makeMounts(pod *api.Pod, podDir string, container *api.Container, podVolumes kubecontainer.VolumeMap) ([]kubecontainer.Mount, error) {
	// Kubernetes only mounts on /etc/hosts if :
	// - container does not use hostNetwork and
	// - container is not a infrastructure(pause) container
	// - container is not already mounting on /etc/hosts
	// When the pause container is being created, its IP is still unknown. Hence, PodIP will not have been set.
	mountEtcHostsFile := !pod.Spec.HostNetwork && len(pod.Status.PodIP) > 0
	glog.V(4).Infof("Will create hosts mount for container:%q, podIP:%s: %v", container.Name, pod.Status.PodIP, mountEtcHostsFile)
	mounts := []kubecontainer.Mount{}
	for _, mount := range container.VolumeMounts {
		mountEtcHostsFile = mountEtcHostsFile && (mount.MountPath != etcHostsPath)
		vol, ok := podVolumes[mount.Name]
		if !ok {
			glog.Warningf("Mount cannot be satisified for container %q, because the volume is missing: %q", container.Name, mount)
			continue
		}
		mounts = append(mounts, kubecontainer.Mount{
			Name:          mount.Name,
			ContainerPath: mount.MountPath,
			HostPath:      vol.GetPath(),
			ReadOnly:      mount.ReadOnly,
		})
	}
	if mountEtcHostsFile {
		hostsMount, err := makeHostsMount(podDir, pod.Status.PodIP, pod.Name)
		if err != nil {
			return nil, err
		}
		mounts = append(mounts, *hostsMount)
	}
	return mounts, nil
}

func makeHostsMount(podDir, podIP, podName string) (*kubecontainer.Mount, error) {
	hostsFilePath := path.Join(podDir, "etc-hosts")
	if err := ensureHostsFile(hostsFilePath, podIP, podName); err != nil {
		return nil, err
	}
	return &kubecontainer.Mount{
		Name:          "k8s-managed-etc-hosts",
		ContainerPath: etcHostsPath,
		HostPath:      hostsFilePath,
		ReadOnly:      false,
	}, nil
}

func ensureHostsFile(fileName string, hostIP, hostName string) error {
	if _, err := os.Stat(fileName); os.IsExist(err) {
		glog.V(4).Infof("kubernetes-managed etc-hosts file exits. Will not be recreated: %q", fileName)
		return nil
	}
	var buffer bytes.Buffer
	buffer.WriteString("# Kubernetes-managed hosts file.\n")
	buffer.WriteString("127.0.0.1\tlocalhost\n")                      // ipv4 localhost
	buffer.WriteString("::1\tlocalhost ip6-localhost ip6-loopback\n") // ipv6 localhost
	buffer.WriteString("fe00::0\tip6-localnet\n")
	buffer.WriteString("fe00::0\tip6-mcastprefix\n")
	buffer.WriteString("fe00::1\tip6-allnodes\n")
	buffer.WriteString("fe00::2\tip6-allrouters\n")
	buffer.WriteString(fmt.Sprintf("%s\t%s\n", hostIP, hostName))
	return ioutil.WriteFile(fileName, buffer.Bytes(), 0644)
}

func makePortMappings(container *api.Container) (ports []kubecontainer.PortMapping) {
	names := make(map[string]struct{})
	for _, p := range container.Ports {
		pm := kubecontainer.PortMapping{
			HostPort:      p.HostPort,
			ContainerPort: p.ContainerPort,
			Protocol:      p.Protocol,
			HostIP:        p.HostIP,
		}

		// We need to create some default port name if it's not specified, since
		// this is necessary for rkt.
		// http://issue.k8s.io/7710
		if p.Name == "" {
			pm.Name = fmt.Sprintf("%s-%s:%d", container.Name, p.Protocol, p.ContainerPort)
		} else {
			pm.Name = fmt.Sprintf("%s-%s", container.Name, p.Name)
		}

		// Protect against exposing the same protocol-port more than once in a container.
		if _, ok := names[pm.Name]; ok {
			glog.Warningf("Port name conflicted, %q is defined more than once", pm.Name)
			continue
		}
		ports = append(ports, pm)
		names[pm.Name] = struct{}{}
	}
	return
}

// GenerateRunContainerOptions generates the RunContainerOptions, which can be used by
// the container runtime to set parameters for launching a container.
func (kl *Kubelet) GenerateRunContainerOptions(pod *api.Pod, container *api.Container) (*kubecontainer.RunContainerOptions, error) {
	var err error
	opts := &kubecontainer.RunContainerOptions{CgroupParent: kl.cgroupRoot}

	vol, ok := kl.volumeManager.GetVolumes(pod.UID)
	if !ok {
		return nil, fmt.Errorf("impossible: cannot find the mounted volumes for pod %q", kubecontainer.GetPodFullName(pod))
	}

	opts.PortMappings = makePortMappings(container)
	opts.Mounts, err = makeMounts(pod, kl.getPodDir(pod.UID), container, vol)
	if err != nil {
		return nil, err
	}
	opts.Envs, err = kl.makeEnvironmentVariables(pod, container)
	if err != nil {
		return nil, err
	}

	if len(container.TerminationMessagePath) != 0 {
		p := kl.getPodContainerDir(pod.UID, container.Name)
		if err := os.MkdirAll(p, 0750); err != nil {
			glog.Errorf("Error on creating %q: %v", p, err)
		} else {
			opts.PodContainerDir = p
		}
	}

	opts.DNS, opts.DNSSearch, err = kl.getClusterDNS(pod)
	if err != nil {
		return nil, err
	}

	return opts, nil
}

var masterServices = sets.NewString("kubernetes")

// getServiceEnvVarMap makes a map[string]string of env vars for services a pod in namespace ns should see
func (kl *Kubelet) getServiceEnvVarMap(ns string) (map[string]string, error) {
	var (
		serviceMap = make(map[string]api.Service)
		m          = make(map[string]string)
	)

	// Get all service resources from the master (via a cache),
	// and populate them into service environment variables.
	if kl.serviceLister == nil {
		// Kubelets without masters (e.g. plain GCE ContainerVM) don't set env vars.
		return m, nil
	}
	services, err := kl.serviceLister.List()
	if err != nil {
		return m, fmt.Errorf("failed to list services when setting up env vars.")
	}

	// project the services in namespace ns onto the master services
	for _, service := range services.Items {
		// ignore services where ClusterIP is "None" or empty
		if !api.IsServiceIPSet(&service) {
			continue
		}
		serviceName := service.Name

		switch service.Namespace {
		// for the case whether the master service namespace is the namespace the pod
		// is in, the pod should receive all the services in the namespace.
		//
		// ordering of the case clauses below enforces this
		case ns:
			serviceMap[serviceName] = service
		case kl.masterServiceNamespace:
			if masterServices.Has(serviceName) {
				if _, exists := serviceMap[serviceName]; !exists {
					serviceMap[serviceName] = service
				}
			}
		}
	}
	services.Items = []api.Service{}
	for _, service := range serviceMap {
		services.Items = append(services.Items, service)
	}

	for _, e := range envvars.FromServices(&services) {
		m[e.Name] = e.Value
	}
	return m, nil
}

// Make the service environment variables for a pod in the given namespace.
func (kl *Kubelet) makeEnvironmentVariables(pod *api.Pod, container *api.Container) ([]kubecontainer.EnvVar, error) {
	var result []kubecontainer.EnvVar
	// Note:  These are added to the docker.Config, but are not included in the checksum computed
	// by dockertools.BuildDockerName(...).  That way, we can still determine whether an
	// api.Container is already running by its hash. (We don't want to restart a container just
	// because some service changed.)
	//
	// Note that there is a race between Kubelet seeing the pod and kubelet seeing the service.
	// To avoid this users can: (1) wait between starting a service and starting; or (2) detect
	// missing service env var and exit and be restarted; or (3) use DNS instead of env vars
	// and keep trying to resolve the DNS name of the service (recommended).
	serviceEnv, err := kl.getServiceEnvVarMap(pod.Namespace)
	if err != nil {
		return result, err
	}

	// Determine the final values of variables:
	//
	// 1.  Determine the final value of each variable:
	//     a.  If the variable's Value is set, expand the `$(var)` references to other
	//         variables in the .Value field; the sources of variables are the declared
	//         variables of the container and the service environment variables
	//     b.  If a source is defined for an environment variable, resolve the source
	// 2.  Create the container's environment in the order variables are declared
	// 3.  Add remaining service environment vars

	tmpEnv := make(map[string]string)
	mappingFunc := expansion.MappingFuncFor(tmpEnv, serviceEnv)
	for _, envVar := range container.Env {
		// Accesses apiserver+Pods.
		// So, the master may set service env vars, or kubelet may.  In case both are doing
		// it, we delete the key from the kubelet-generated ones so we don't have duplicate
		// env vars.
		// TODO: remove this net line once all platforms use apiserver+Pods.
		delete(serviceEnv, envVar.Name)

		runtimeVal := envVar.Value
		if runtimeVal != "" {
			// Step 1a: expand variable references
			runtimeVal = expansion.Expand(runtimeVal, mappingFunc)
		} else if envVar.ValueFrom != nil && envVar.ValueFrom.FieldRef != nil {
			// Step 1b: resolve alternate env var sources
			runtimeVal, err = kl.podFieldSelectorRuntimeValue(envVar.ValueFrom.FieldRef, pod)
			if err != nil {
				return result, err
			}
		}

		tmpEnv[envVar.Name] = runtimeVal
		result = append(result, kubecontainer.EnvVar{Name: envVar.Name, Value: tmpEnv[envVar.Name]})
	}

	// Append remaining service env vars.
	for k, v := range serviceEnv {
		result = append(result, kubecontainer.EnvVar{Name: k, Value: v})
	}
	return result, nil
}

func (kl *Kubelet) podFieldSelectorRuntimeValue(fs *api.ObjectFieldSelector, pod *api.Pod) (string, error) {
	internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(fs.APIVersion, "Pod", fs.FieldPath, "")
	if err != nil {
		return "", err
	}
	switch internalFieldPath {
	case "status.podIP":
		return pod.Status.PodIP, nil
	}
	return fieldpath.ExtractFieldPathAsString(pod, internalFieldPath)
}

// getClusterDNS returns a list of the DNS servers and a list of the DNS search
// domains of the cluster.
func (kl *Kubelet) getClusterDNS(pod *api.Pod) ([]string, []string, error) {
	var hostDNS, hostSearch []string
	// Get host DNS settings and append them to cluster DNS settings.
	if kl.resolverConfig != "" {
		f, err := os.Open(kl.resolverConfig)
		if err != nil {
			return nil, nil, err
		}
		defer f.Close()

		hostDNS, hostSearch, err = kl.parseResolvConf(f)
		if err != nil {
			return nil, nil, err
		}
	}
	if pod.Spec.DNSPolicy != api.DNSClusterFirst {
		// When the kubelet --resolv-conf flag is set to the empty string, use
		// DNS settings that override the docker default (which is to use
		// /etc/resolv.conf) and effectivly disable DNS lookups. According to
		// the bind documentation, the behavior of the DNS client library when
		// "nameservers" are not specified is to "use the nameserver on the
		// local machine". A nameserver setting of localhost is equivalent to
		// this documented behavior.
		if kl.resolverConfig == "" {
			hostDNS = []string{"127.0.0.1"}
			hostSearch = []string{"."}
		}
		return hostDNS, hostSearch, nil
	}
	var dns, dnsSearch []string

	if kl.clusterDNS != nil {
		dns = append([]string{kl.clusterDNS.String()}, hostDNS...)
	} else {
		dns = hostDNS
	}
	if kl.clusterDomain != "" {
		nsSvcDomain := fmt.Sprintf("%s.svc.%s", pod.Namespace, kl.clusterDomain)
		svcDomain := fmt.Sprintf("svc.%s", kl.clusterDomain)
		dnsSearch = append([]string{nsSvcDomain, svcDomain, kl.clusterDomain}, hostSearch...)
	} else {
		dnsSearch = hostSearch
	}
	return dns, dnsSearch, nil
}

// Returns the list of DNS servers and DNS search domains.
func (kl *Kubelet) parseResolvConf(reader io.Reader) (nameservers []string, searches []string, err error) {
	var scrubber dnsScrubber
	if kl.cloud != nil {
		scrubber = kl.cloud
	}
	return parseResolvConf(reader, scrubber)
}

// A helper for testing.
type dnsScrubber interface {
	ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string)
}

func parseResolvConf(reader io.Reader, dnsScrubber dnsScrubber) (nameservers []string, searches []string, err error) {
	file, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, nil, err
	}

	// Lines of the form "nameserver 1.2.3.4" accumulate.
	nameservers = []string{}

	// Lines of the form "search example.com" overrule - last one wins.
	searches = []string{}

	lines := strings.Split(string(file), "\n")
	for l := range lines {
		trimmed := strings.TrimSpace(lines[l])
		if strings.HasPrefix(trimmed, "#") {
			continue
		}
		fields := strings.Fields(trimmed)
		if len(fields) == 0 {
			continue
		}
		if fields[0] == "nameserver" {
			nameservers = append(nameservers, fields[1:]...)
		}
		if fields[0] == "search" {
			searches = fields[1:]
		}
	}

	// Give the cloud-provider a chance to post-process DNS settings.
	if dnsScrubber != nil {
		nameservers, searches = dnsScrubber.ScrubDNS(nameservers, searches)
	}
	return nameservers, searches, nil
}

// Kill all running containers in a pod (includes the pod infra container).
func (kl *Kubelet) killPod(pod *api.Pod, runningPod kubecontainer.Pod) error {
	return kl.containerRuntime.KillPod(pod, runningPod)
}

type empty struct{}

// makePodDataDirs creates the dirs for the pod datas.
func (kl *Kubelet) makePodDataDirs(pod *api.Pod) error {
	uid := pod.UID
	if err := os.Mkdir(kl.getPodDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	if err := os.Mkdir(kl.getPodVolumesDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	if err := os.Mkdir(kl.getPodPluginsDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	return nil
}

func (kl *Kubelet) syncPod(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod, updateType SyncPodType) error {
	podFullName := kubecontainer.GetPodFullName(pod)
	uid := pod.UID
	start := time.Now()
	var firstSeenTime time.Time
	if firstSeenTimeStr, ok := pod.Annotations[ConfigFirstSeenAnnotationKey]; !ok {
		glog.V(3).Infof("First seen time not recorded for pod %q", pod.UID)
	} else {
		firstSeenTime = kubeletTypes.ConvertToTimestamp(firstSeenTimeStr).Get()
	}

	// Before returning, regenerate status and store it in the cache.
	defer func() {
		if isStaticPod(pod) && mirrorPod == nil {
			// No need to cache the status because the mirror pod does not
			// exist yet.
			return
		}
		status, err := kl.generatePodStatus(pod)
		if err != nil {
			glog.Errorf("Unable to generate status for pod with name %q and uid %q info with error(%v)", podFullName, uid, err)
		} else {
			podToUpdate := pod
			if mirrorPod != nil {
				podToUpdate = mirrorPod
			}
			existingStatus, ok := kl.statusManager.GetPodStatus(podToUpdate.UID)
			if !ok || existingStatus.Phase == api.PodPending && status.Phase == api.PodRunning &&
				!firstSeenTime.IsZero() {
				metrics.PodStartLatency.Observe(metrics.SinceInMicroseconds(firstSeenTime))
			}
			kl.statusManager.SetPodStatus(podToUpdate, status)
		}
	}()

	// Kill pods we can't run.
	if err := canRunPod(pod); err != nil || pod.DeletionTimestamp != nil {
		if err := kl.killPod(pod, runningPod); err != nil {
			util.HandleError(err)
		}
		return err
	}

	if err := kl.makePodDataDirs(pod); err != nil {
		glog.Errorf("Unable to make pod data directories for pod %q (uid %q): %v", podFullName, uid, err)
		return err
	}

	// Starting phase:
	ref, err := api.GetReference(pod)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %q: '%v'", podFullName, err)
	}

	// Mount volumes.
	podVolumes, err := kl.mountExternalVolumes(pod)
	if err != nil {
		if ref != nil {
			kl.recorder.Eventf(ref, "FailedMount", "Unable to mount volumes for pod %q: %v", podFullName, err)
		}
		glog.Errorf("Unable to mount volumes for pod %q: %v; skipping pod", podFullName, err)
		return err
	}
	kl.volumeManager.SetVolumes(pod.UID, podVolumes)

	// The kubelet is the source of truth for pod status. It ignores the status sent from
	// the apiserver and regenerates status for every pod update, incrementally updating
	// the status it received at pod creation time.
	//
	// The container runtime needs 2 pieces of information from the status to sync a pod:
	// The terminated state of containers (to restart them) and the podIp (for liveness probes).
	// New pods don't have either, so we skip the expensive status generation step.
	//
	// If we end up here with a create event for an already running pod, it could result in a
	// restart of its containers. This cannot happen unless the kubelet restarts, because the
	// delete before the second create would cancel this pod worker.
	//
	// If the kubelet restarts, we have a bunch of running containers for which we get create
	// events. This is ok, because the pod status for these will include the podIp and terminated
	// status. Any race conditions here effectively boils down to -- the pod worker didn't sync
	// state of a newly started container with the apiserver before the kubelet restarted, so
	// it's OK to pretend like the kubelet started them after it restarted.
	//
	// Also note that deletes currently have an updateType of `create` set in UpdatePods.
	// This, again, does not matter because deletes are not processed by this method.

	var podStatus api.PodStatus
	if updateType == SyncPodCreate {
		// This is the first time we are syncing the pod. Record the latency
		// since kubelet first saw the pod if firstSeenTime is set.
		if !firstSeenTime.IsZero() {
			metrics.PodWorkerStartLatency.Observe(metrics.SinceInMicroseconds(firstSeenTime))
		}

		podStatus = pod.Status
		podStatus.StartTime = &unversioned.Time{Time: start}
		kl.statusManager.SetPodStatus(pod, podStatus)
		glog.V(3).Infof("Not generating pod status for new pod %q", podFullName)
	} else {
		var err error
		podStatus, err = kl.generatePodStatus(pod)
		if err != nil {
			glog.Errorf("Unable to get status for pod %q (uid %q): %v", podFullName, uid, err)
			return err
		}
	}

	pullSecrets, err := kl.getPullSecretsForPod(pod)
	if err != nil {
		glog.Errorf("Unable to get pull secrets for pod %q (uid %q): %v", podFullName, uid, err)
		return err
	}

	err = kl.containerRuntime.SyncPod(pod, runningPod, podStatus, pullSecrets, kl.backOff)
	if err != nil {
		return err
	}

	ingress, egress, err := extractBandwidthResources(pod)
	if err != nil {
		return err
	}
	if egress != nil || ingress != nil {
		if pod.Spec.HostNetwork {
			kl.recorder.Event(pod, "HostNetworkNotSupported", "Bandwidth shaping is not currently supported on the host network")
		} else if kl.shaper != nil {
			status, found := kl.statusManager.GetPodStatus(pod.UID)
			if !found {
				statusPtr, err := kl.containerRuntime.GetPodStatus(pod)
				if err != nil {
					glog.Errorf("Error getting pod for bandwidth shaping")
					return err
				}
				status = *statusPtr
			}
			if len(status.PodIP) > 0 {
				err = kl.shaper.ReconcileCIDR(fmt.Sprintf("%s/32", status.PodIP), egress, ingress)
			}
		} else {
			kl.recorder.Event(pod, "NilShaper", "Pod requests bandwidth shaping, but the shaper is undefined")
		}
	}

	if isStaticPod(pod) {
		if mirrorPod != nil && !kl.podManager.IsMirrorPodOf(mirrorPod, pod) {
			// The mirror pod is semantically different from the static pod. Remove
			// it. The mirror pod will get recreated later.
			glog.Errorf("Deleting mirror pod %q because it is outdated", podFullName)
			if err := kl.podManager.DeleteMirrorPod(podFullName); err != nil {
				glog.Errorf("Failed deleting mirror pod %q: %v", podFullName, err)
			}
		}
		if mirrorPod == nil {
			glog.V(3).Infof("Creating a mirror pod %q", podFullName)
			if err := kl.podManager.CreateMirrorPod(pod); err != nil {
				glog.Errorf("Failed creating a mirror pod %q: %v", podFullName, err)
			}
		}
	}
	return nil
}

// getPullSecretsForPod inspects the Pod and retrieves the referenced pull secrets
// TODO duplicate secrets are being retrieved multiple times and there is no cache.  Creating and using a secret manager interface will make this easier to address.
func (kl *Kubelet) getPullSecretsForPod(pod *api.Pod) ([]api.Secret, error) {
	pullSecrets := []api.Secret{}

	for _, secretRef := range pod.Spec.ImagePullSecrets {
		secret, err := kl.kubeClient.Secrets(pod.Namespace).Get(secretRef.Name)
		if err != nil {
			glog.Warningf("Unable to retrieve pull secret %s/%s for %s/%s due to %v.  The image pull may not succeed.", pod.Namespace, secretRef.Name, pod.Namespace, pod.Name, err)
			continue
		}

		pullSecrets = append(pullSecrets, *secret)
	}

	return pullSecrets, nil
}

// Stores all volumes defined by the set of pods into a map.
// Keys for each entry are in the format (POD_ID)/(VOLUME_NAME)
func getDesiredVolumes(pods []*api.Pod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Spec.Volumes {
			identifier := path.Join(string(pod.UID), volume.Name)
			desiredVolumes[identifier] = volume
		}
	}
	return desiredVolumes
}

// cleanupOrphanedPodDirs removes a pod directory if the pod is not in the
// desired set of pods and there is no running containers in the pod.
func (kl *Kubelet) cleanupOrphanedPodDirs(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	active := sets.NewString()
	for _, pod := range pods {
		active.Insert(string(pod.UID))
	}
	for _, pod := range runningPods {
		active.Insert(string(pod.ID))
	}

	found, err := kl.listPodsFromDisk()
	if err != nil {
		return err
	}
	errlist := []error{}
	for _, uid := range found {
		if active.Has(string(uid)) {
			continue
		}
		if volumes, err := kl.getPodVolumes(uid); err != nil || len(volumes) != 0 {
			glog.V(3).Infof("Orphaned pod %q found, but volumes are not cleaned up; err: %v, volumes: %v ", uid, err, volumes)
			continue
		}

		glog.V(3).Infof("Orphaned pod %q found, removing", uid)
		if err := os.RemoveAll(kl.getPodDir(uid)); err != nil {
			errlist = append(errlist, err)
		}
	}
	return utilErrors.NewAggregate(errlist)
}

func (kl *Kubelet) cleanupBandwidthLimits(allPods []*api.Pod) error {
	if kl.shaper == nil {
		return nil
	}
	currentCIDRs, err := kl.shaper.GetCIDRs()
	if err != nil {
		return err
	}
	possibleCIDRs := sets.String{}
	for ix := range allPods {
		pod := allPods[ix]
		ingress, egress, err := extractBandwidthResources(pod)
		if err != nil {
			return err
		}
		if ingress == nil && egress == nil {
			glog.V(8).Infof("Not a bandwidth limited container...")
			continue
		}
		status, found := kl.statusManager.GetPodStatus(pod.UID)
		if !found {
			statusPtr, err := kl.containerRuntime.GetPodStatus(pod)
			if err != nil {
				return err
			}
			status = *statusPtr
		}
		if status.Phase == api.PodRunning {
			possibleCIDRs.Insert(fmt.Sprintf("%s/32", status.PodIP))
		}
	}
	for _, cidr := range currentCIDRs {
		if !possibleCIDRs.Has(cidr) {
			glog.V(2).Infof("Removing CIDR: %s (%v)", cidr, possibleCIDRs)
			if err := kl.shaper.Reset(cidr); err != nil {
				return err
			}
		}
	}
	return nil
}

// Compares the map of current volumes to the map of desired volumes.
// If an active volume does not have a respective desired volume, clean it up.
func (kl *Kubelet) cleanupOrphanedVolumes(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	desiredVolumes := getDesiredVolumes(pods)
	currentVolumes := kl.getPodVolumesFromDisk()

	runningSet := sets.String{}
	for _, pod := range runningPods {
		runningSet.Insert(string(pod.ID))
	}

	for name, vol := range currentVolumes {
		if _, ok := desiredVolumes[name]; !ok {
			parts := strings.Split(name, "/")
			if runningSet.Has(parts[0]) {
				glog.Infof("volume %q, still has a container running %q, skipping teardown", name, parts[0])
				continue
			}
			//TODO (jonesdl) We should somehow differentiate between volumes that are supposed
			//to be deleted and volumes that are leftover after a crash.
			glog.Warningf("Orphaned volume %q found, tearing down volume", name)
			// TODO(yifan): Refactor this hacky string manipulation.
			kl.volumeManager.DeleteVolumes(types.UID(parts[0]))
			//TODO (jonesdl) This should not block other kubelet synchronization procedures
			err := vol.TearDown()
			if err != nil {
				glog.Errorf("Could not tear down volume %q: %v", name, err)
			}
		}
	}
	return nil
}

// Delete any pods that are no longer running and are marked for deletion.
func (kl *Kubelet) cleanupTerminatedPods(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	var terminating []*api.Pod
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			found := false
			for _, runningPod := range runningPods {
				if runningPod.ID == pod.UID {
					found = true
					break
				}
			}
			if found {
				podFullName := kubecontainer.GetPodFullName(pod)
				glog.V(5).Infof("Keeping terminated pod %q and uid %q, still running", podFullName, pod.UID)
				continue
			}
			terminating = append(terminating, pod)
		}
	}
	if !kl.statusManager.TerminatePods(terminating) {
		return errors.New("not all pods were successfully terminated")
	}
	return nil
}

// pastActiveDeadline returns true if the pod has been active for more than
// ActiveDeadlineSeconds.
func (kl *Kubelet) pastActiveDeadline(pod *api.Pod) bool {
	now := unversioned.Now()
	if pod.Spec.ActiveDeadlineSeconds != nil {
		podStatus, ok := kl.statusManager.GetPodStatus(pod.UID)
		if !ok {
			podStatus = pod.Status
		}
		if !podStatus.StartTime.IsZero() {
			startTime := podStatus.StartTime.Time
			duration := now.Time.Sub(startTime)
			allowedDuration := time.Duration(*pod.Spec.ActiveDeadlineSeconds) * time.Second
			if duration >= allowedDuration {
				return true
			}
		}
	}
	return false
}

// Returns true if pod is in the terminated state ("Failed" or "Succeeded").
func (kl *Kubelet) podIsTerminated(pod *api.Pod) bool {
	var status api.PodStatus
	// Check the cached pod status which was set after the last sync.
	status, ok := kl.statusManager.GetPodStatus(pod.UID)
	if !ok {
		// If there is no cached status, use the status from the
		// apiserver. This is useful if kubelet has recently been
		// restarted.
		status = pod.Status
	}
	if status.Phase == api.PodFailed || status.Phase == api.PodSucceeded {
		return true
	}

	return false
}

func (kl *Kubelet) filterOutTerminatedPods(pods []*api.Pod) []*api.Pod {
	var filteredPods []*api.Pod
	for _, p := range pods {
		if kl.podIsTerminated(p) {
			continue
		}
		filteredPods = append(filteredPods, p)
	}
	return filteredPods
}

// removeOrphanedPodStatuses removes obsolete entries in podStatus where
// the pod is no longer considered bound to this node.
func (kl *Kubelet) removeOrphanedPodStatuses(pods []*api.Pod, mirrorPods []*api.Pod) {
	podUIDs := make(map[types.UID]bool)
	for _, pod := range pods {
		podUIDs[pod.UID] = true
	}
	for _, pod := range mirrorPods {
		podUIDs[pod.UID] = true
	}
	kl.statusManager.RemoveOrphanedStatuses(podUIDs)
}

func (kl *Kubelet) deletePod(uid types.UID) error {
	if !kl.sourcesReady() {
		// If the sources aren't ready, skip deletion, as we may accidentally delete pods
		// for sources that haven't reported yet.
		return fmt.Errorf("skipping delete because sources aren't ready yet")
	}
	kl.podWorkers.ForgetWorker(uid)

	// Runtime cache may not have been updated to with the pod, but it's okay
	// because the periodic cleanup routine will attempt to delete again later.
	runningPods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return fmt.Errorf("error listing containers: %v", err)
	}
	pod := kubecontainer.Pods(runningPods).FindPod("", uid)
	if pod.IsEmpty() {
		return fmt.Errorf("pod not found")
	}

	kl.podKillingCh <- &pod
	// TODO: delete the mirror pod here?

	// We leave the volume/directory cleanup to the periodic cleanup routine.
	return nil
}

// HandlePodCleanups performs a series of cleanup work, including terminating
// pod workers, killing unwanted pods, and removing orphaned volumes/pod
// directories.
// TODO(yujuhong): This function is executed by the main sync loop, so it
// should not contain any blocking calls. Re-examine the function and decide
// whether or not we should move it into a separte goroutine.
func (kl *Kubelet) HandlePodCleanups() error {
	allPods, mirrorPods := kl.podManager.GetPodsAndMirrorPods()
	// Pod phase progresses monotonically. Once a pod has reached a final state,
	// it should never leave regardless of the restart policy. The statuses
	// of such pods should not be changed, and there is no need to sync them.
	// TODO: the logic here does not handle two cases:
	//   1. If the containers were removed immediately after they died, kubelet
	//      may fail to generate correct statuses, let alone filtering correctly.
	//   2. If kubelet restarted before writing the terminated status for a pod
	//      to the apiserver, it could still restart the terminated pod (even
	//      though the pod was not considered terminated by the apiserver).
	// These two conditions could be alleviated by checkpointing kubelet.
	activePods := kl.filterOutTerminatedPods(allPods)

	desiredPods := make(map[types.UID]empty)
	for _, pod := range activePods {
		desiredPods[pod.UID] = empty{}
	}
	// Stop the workers for no-longer existing pods.
	// TODO: is here the best place to forget pod workers?
	kl.podWorkers.ForgetNonExistingPodWorkers(desiredPods)

	runningPods, err := kl.runtimeCache.GetPods()
	if err != nil {
		glog.Errorf("Error listing containers: %#v", err)
		return err
	}
	for _, pod := range runningPods {
		if _, found := desiredPods[pod.ID]; !found {
			kl.podKillingCh <- pod
		}
	}

	kl.removeOrphanedPodStatuses(allPods, mirrorPods)
	// Note that we just killed the unwanted pods. This may not have reflected
	// in the cache. We need to bypass the cache to get the latest set of
	// running pods to clean up the volumes.
	// TODO: Evaluate the performance impact of bypassing the runtime cache.
	runningPods, err = kl.containerRuntime.GetPods(false)
	if err != nil {
		glog.Errorf("Error listing containers: %#v", err)
		return err
	}

	// Remove any orphaned volumes.
	// Note that we pass all pods (including terminated pods) to the function,
	// so that we don't remove volumes associated with terminated but not yet
	// deleted pods.
	err = kl.cleanupOrphanedVolumes(allPods, runningPods)
	if err != nil {
		glog.Errorf("Failed cleaning up orphaned volumes: %v", err)
		return err
	}

	// Remove any orphaned pod directories.
	// Note that we pass all pods (including terminated pods) to the function,
	// so that we don't remove directories associated with terminated but not yet
	// deleted pods.
	err = kl.cleanupOrphanedPodDirs(allPods, runningPods)
	if err != nil {
		glog.Errorf("Failed cleaning up orphaned pod directories: %v", err)
		return err
	}

	// Remove any orphaned mirror pods.
	kl.podManager.DeleteOrphanedMirrorPods()

	if err := kl.cleanupTerminatedPods(allPods, runningPods); err != nil {
		glog.Errorf("Failed to cleanup terminated pods: %v", err)
	}

	// Clear out any old bandwith rules
	if err = kl.cleanupBandwidthLimits(allPods); err != nil {
		return err
	}

	kl.backOff.GC()
	return err
}

// podKiller launches a goroutine to kill a pod received from the channel if
// another goroutine isn't already in action.
func (kl *Kubelet) podKiller() {
	killing := sets.NewString()
	resultCh := make(chan types.UID)
	defer close(resultCh)
	for {
		select {
		case pod, ok := <-kl.podKillingCh:
			if !ok {
				return
			}
			if killing.Has(string(pod.ID)) {
				// The pod is already being killed.
				break
			}
			killing.Insert(string(pod.ID))
			go func(pod *kubecontainer.Pod, ch chan types.UID) {
				defer func() {
					ch <- pod.ID
				}()
				glog.V(2).Infof("Killing unwanted pod %q", pod.Name)
				err := kl.killPod(nil, *pod)
				if err != nil {
					glog.Errorf("Failed killing the pod %q: %v", pod.Name, err)
				}
			}(pod, resultCh)

		case podID := <-resultCh:
			killing.Delete(string(podID))
		}
	}
}

type podsByCreationTime []*api.Pod

func (s podsByCreationTime) Len() int {
	return len(s)
}

func (s podsByCreationTime) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s podsByCreationTime) Less(i, j int) bool {
	return s[i].CreationTimestamp.Before(s[j].CreationTimestamp)
}

// checkHostPortConflicts detects pods with conflicted host ports.
func hasHostPortConflicts(pods []*api.Pod) bool {
	ports := sets.String{}
	for _, pod := range pods {
		if errs := validation.AccumulateUniqueHostPorts(pod.Spec.Containers, &ports); len(errs) > 0 {
			glog.Errorf("Pod %q: HostPort is already allocated, ignoring: %v", kubecontainer.GetPodFullName(pod), errs)
			return true
		}
	}
	return false
}

// hasInsufficientfFreeResources detects pods that exceeds node's resources.
// TODO: Consider integrate disk space into this function, and returns a
// suitable reason and message per resource type.
func (kl *Kubelet) hasInsufficientfFreeResources(pods []*api.Pod) (bool, bool) {
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		glog.Errorf("error getting machine info: %v", err)
		// TODO: Should we admit the pod when machine info is unavailable?
		return false, false
	}
	capacity := CapacityFromMachineInfo(info)
	_, notFittingCPU, notFittingMemory := predicates.CheckPodsExceedingFreeResources(pods, capacity)
	return len(notFittingCPU) > 0, len(notFittingMemory) > 0
}

// handleOutOfDisk detects if pods can't fit due to lack of disk space.
func (kl *Kubelet) isOutOfDisk() bool {
	outOfDockerDisk := false
	outOfRootDisk := false
	// Check disk space once globally and reject or accept all new pods.
	withinBounds, err := kl.diskSpaceManager.IsDockerDiskSpaceAvailable()
	// Assume enough space in case of errors.
	if err == nil && !withinBounds {
		outOfDockerDisk = true
	}

	withinBounds, err = kl.diskSpaceManager.IsRootDiskSpaceAvailable()
	// Assume enough space in case of errors.
	if err == nil && !withinBounds {
		outOfRootDisk = true
	}
	// Kubelet would indicate all pods as newly created on the first run after restart.
	// We ignore the first disk check to ensure that running pods are not killed.
	// Disk manager will only declare out of disk problems if unfreeze has been called.
	kl.diskSpaceManager.Unfreeze()

	return outOfDockerDisk || outOfRootDisk
}

// matchesNodeSelector returns true if pod matches node's labels.
func (kl *Kubelet) matchesNodeSelector(pod *api.Pod) bool {
	if kl.standaloneMode {
		return true
	}
	node, err := kl.GetNode()
	if err != nil {
		glog.Errorf("error getting node: %v", err)
		return true
	}
	return predicates.PodMatchesNodeLabels(pod, node)
}

func (kl *Kubelet) rejectPod(pod *api.Pod, reason, message string) {
	kl.recorder.Eventf(pod, reason, message)
	kl.statusManager.SetPodStatus(pod, api.PodStatus{
		Phase:   api.PodFailed,
		Reason:  reason,
		Message: "Pod " + message})
}

// canAdmitPod determines if a pod can be admitted, and gives a reason if it
// cannot. "pod" is new pod, while "pods" include all admitted pods plus the
// new pod. The function returns a boolean value indicating whether the pod
// can be admitted, a brief single-word reason and a message explaining why
// the pod cannot be admitted.
func (kl *Kubelet) canAdmitPod(pods []*api.Pod, pod *api.Pod) (bool, string, string) {
	if hasHostPortConflicts(pods) {
		return false, "HostPortConflict", "cannot start the pod due to host port conflict."
	}
	if !kl.matchesNodeSelector(pod) {
		return false, "NodeSelectorMismatching", "cannot be started due to node selector mismatch"
	}
	cpu, memory := kl.hasInsufficientfFreeResources(pods)
	if cpu {
		return false, "InsufficientFreeCPU", "cannot start the pod due to insufficient free CPU."
	} else if memory {
		return false, "InsufficientFreeMemory", "cannot be started due to insufficient free memory"
	}
	if kl.isOutOfDisk() {
		return false, "OutOfDisk", "cannot be started due to lack of disk space."
	}

	return true, "", ""
}

// syncLoop is the main loop for processing changes. It watches for changes from
// three channels (file, apiserver, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync-frequency seconds. Never returns.
func (kl *Kubelet) syncLoop(updates <-chan PodUpdate, handler SyncHandler) {
	glog.Info("Starting kubelet main sync loop.")
	kl.resyncTicker = time.NewTicker(kl.resyncInterval)
	var housekeepingTimestamp time.Time
	for {
		if !kl.containerRuntimeUp() {
			time.Sleep(5 * time.Second)
			glog.Infof("Skipping pod synchronization, container runtime is not up.")
			continue
		}
		if !kl.doneNetworkConfigure() {
			time.Sleep(5 * time.Second)
			glog.Infof("Skipping pod synchronization, network is not configured")
			continue
		}

		// Make sure we sync first to receive the pods from the sources before
		// performing housekeeping.
		if !kl.syncLoopIteration(updates, handler) {
			break
		}
		// We don't want to perform housekeeping too often, so we set a minimum
		// period for it. Housekeeping would be performed at least once every
		// kl.resyncInterval, and *no* more than once every
		// housekeepingMinimumPeriod.
		// TODO (#13418): Investigate whether we can/should spawn a dedicated
		// goroutine for housekeeping
		if !kl.sourcesReady() {
			// If the sources aren't ready, skip housekeeping, as we may
			// accidentally delete pods from unready sources.
			glog.V(4).Infof("Skipping cleanup, sources aren't ready yet.")
		} else if housekeepingTimestamp.IsZero() {
			housekeepingTimestamp = time.Now()
		} else if time.Since(housekeepingTimestamp) > housekeepingMinimumPeriod {
			glog.V(4).Infof("SyncLoop (housekeeping)")
			if err := handler.HandlePodCleanups(); err != nil {
				glog.Errorf("Failed cleaning pods: %v", err)
			}
			housekeepingTimestamp = time.Now()
		}
	}
}

func (kl *Kubelet) syncLoopIteration(updates <-chan PodUpdate, handler SyncHandler) bool {
	kl.syncLoopMonitor.Store(time.Now())
	select {
	case u, open := <-updates:
		if !open {
			glog.Errorf("Update channel is closed. Exiting the sync loop.")
			return false
		}
		switch u.Op {
		case ADD:
			glog.V(2).Infof("SyncLoop (ADD): %q", kubeletUtil.FormatPodNames(u.Pods))
			handler.HandlePodAdditions(u.Pods)
		case UPDATE:
			glog.V(2).Infof("SyncLoop (UPDATE): %q", kubeletUtil.FormatPodNames(u.Pods))
			handler.HandlePodUpdates(u.Pods)
		case REMOVE:
			glog.V(2).Infof("SyncLoop (REMOVE): %q", kubeletUtil.FormatPodNames(u.Pods))
			handler.HandlePodDeletions(u.Pods)
		case SET:
			// TODO: Do we want to support this?
			glog.Errorf("Kubelet does not support snapshot update")
		}
	case <-kl.resyncTicker.C:
		// Periodically syncs all the pods and performs cleanup tasks.
		glog.V(4).Infof("SyncLoop (periodic sync)")
		handler.HandlePodSyncs(kl.podManager.GetPods())
	}
	kl.syncLoopMonitor.Store(time.Now())
	return true
}

func (kl *Kubelet) dispatchWork(pod *api.Pod, syncType SyncPodType, mirrorPod *api.Pod, start time.Time) {
	if kl.podIsTerminated(pod) {
		return
	}
	// Run the sync in an async worker.
	kl.podWorkers.UpdatePod(pod, mirrorPod, func() {
		metrics.PodWorkerLatency.WithLabelValues(syncType.String()).Observe(metrics.SinceInMicroseconds(start))
	})
	// Note the number of containers for new pods.
	if syncType == SyncPodCreate {
		metrics.ContainersPerPodCount.Observe(float64(len(pod.Spec.Containers)))
	}
}

// TODO: Consider handling all mirror pods updates in a separate component.
func (kl *Kubelet) handleMirrorPod(mirrorPod *api.Pod, start time.Time) {
	// Mirror pod ADD/UPDATE/DELETE operations are considered an UPDATE to the
	// corresponding static pod. Send update to the pod worker if the static
	// pod exists.
	if pod, ok := kl.podManager.GetPodByMirrorPod(mirrorPod); ok {
		kl.dispatchWork(pod, SyncPodUpdate, mirrorPod, start)
	}
}

func (kl *Kubelet) HandlePodAdditions(pods []*api.Pod) {
	start := time.Now()
	sort.Sort(podsByCreationTime(pods))
	for _, pod := range pods {
		kl.podManager.AddPod(pod)
		if isMirrorPod(pod) {
			kl.handleMirrorPod(pod, start)
			continue
		}
		// Note that allPods includes the new pod since we added at the
		// beginning of the loop.
		allPods := kl.podManager.GetPods()
		// We failed pods that we rejected, so activePods include all admitted
		// pods that are alive and the new pod.
		activePods := kl.filterOutTerminatedPods(allPods)
		// Check if we can admit the pod; if not, reject it.
		if ok, reason, message := kl.canAdmitPod(activePods, pod); !ok {
			kl.rejectPod(pod, reason, message)
			continue
		}
		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		kl.dispatchWork(pod, SyncPodCreate, mirrorPod, start)
	}
}

func (kl *Kubelet) HandlePodUpdates(pods []*api.Pod) {
	start := time.Now()
	for _, pod := range pods {
		kl.podManager.UpdatePod(pod)
		if isMirrorPod(pod) {
			kl.handleMirrorPod(pod, start)
			continue
		}
		// TODO: Evaluate if we need to validate and reject updates.

		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		kl.dispatchWork(pod, SyncPodUpdate, mirrorPod, start)
	}
}

func (kl *Kubelet) HandlePodDeletions(pods []*api.Pod) {
	start := time.Now()
	for _, pod := range pods {
		kl.podManager.DeletePod(pod)
		if isMirrorPod(pod) {
			kl.handleMirrorPod(pod, start)
			continue
		}
		// Deletion is allowed to fail because the periodic cleanup routine
		// will trigger deletion again.
		if err := kl.deletePod(pod.UID); err != nil {
			glog.V(2).Infof("Failed to delete pod %q, err: %v", kubeletUtil.FormatPodName(pod), err)
		}
	}
}

func (kl *Kubelet) HandlePodSyncs(pods []*api.Pod) {
	start := time.Now()
	for _, pod := range pods {
		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		kl.dispatchWork(pod, SyncPodSync, mirrorPod, start)
	}
}

func (kl *Kubelet) LatestLoopEntryTime() time.Time {
	val := kl.syncLoopMonitor.Load()
	if val == nil {
		return time.Time{}
	}
	return val.(time.Time)
}

// Returns the container runtime version for this Kubelet.
func (kl *Kubelet) GetContainerRuntimeVersion() (kubecontainer.Version, error) {
	if kl.containerRuntime == nil {
		return nil, fmt.Errorf("no container runtime")
	}
	return kl.containerRuntime.Version()
}

func (kl *Kubelet) validatePodPhase(podStatus *api.PodStatus) error {
	switch podStatus.Phase {
	case api.PodRunning, api.PodSucceeded, api.PodFailed:
		return nil
	}
	return fmt.Errorf("pod is not in 'Running', 'Succeeded' or 'Failed' state - State: %q", podStatus.Phase)
}

func (kl *Kubelet) validateContainerStatus(podStatus *api.PodStatus, containerName string, previous bool) (containerID string, err error) {
	var cID string

	cStatus, found := api.GetContainerStatus(podStatus.ContainerStatuses, containerName)
	if !found {
		return "", fmt.Errorf("container %q not found", containerName)
	}
	if previous {
		if cStatus.LastTerminationState.Terminated == nil {
			return "", fmt.Errorf("previous terminated container %q not found", containerName)
		}
		cID = cStatus.LastTerminationState.Terminated.ContainerID
	} else {
		if cStatus.State.Waiting != nil {
			return "", fmt.Errorf("container %q is in waiting state.", containerName)
		}
		cID = cStatus.ContainerID
	}
	return kubecontainer.TrimRuntimePrefix(cID), nil
}

// GetKubeletContainerLogs returns logs from the container
// TODO: this method is returning logs of random container attempts, when it should be returning the most recent attempt
// or all of them.
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName string, logOptions *api.PodLogOptions, stdout, stderr io.Writer) error {
	// TODO(vmarmol): Refactor to not need the pod status and verification.
	// Pod workers periodically write status to statusManager. If status is not
	// cached there, something is wrong (or kubelet just restarted and hasn't
	// caught up yet). Just assume the pod is not ready yet.
	name, namespace, err := kubecontainer.ParsePodFullName(podFullName)
	if err != nil {
		return fmt.Errorf("unable to parse pod full name %q: %v", podFullName, err)
	}

	pod, ok := kl.GetPodByName(namespace, name)
	if !ok {
		return fmt.Errorf("unable to get logs for container %q in pod %q namespace %q: unable to find pod", containerName, name, namespace)
	}

	podUID := pod.UID
	if mirrorPod, ok := kl.podManager.GetMirrorPodByPod(pod); ok {
		podUID = mirrorPod.UID
	}
	podStatus, found := kl.statusManager.GetPodStatus(podUID)
	if !found {
		return fmt.Errorf("failed to get status for pod %q in namespace %q", name, namespace)
	}

	if err := kl.validatePodPhase(&podStatus); err != nil {
		// No log is available if pod is not in a "known" phase (e.g. Unknown).
		return fmt.Errorf("Pod %q in namespace %q : %v", name, namespace, err)
	}
	containerID, err := kl.validateContainerStatus(&podStatus, containerName, logOptions.Previous)
	if err != nil {
		// No log is available if the container status is missing or is in the
		// waiting state.
		return fmt.Errorf("Pod %q in namespace %q: %v", name, namespace, err)
	}
	return kl.containerRuntime.GetContainerLogs(pod, containerID, logOptions, stdout, stderr)
}

// GetHostname Returns the hostname as the kubelet sees it.
func (kl *Kubelet) GetHostname() string {
	return kl.hostname
}

// Returns host IP or nil in case of error.
func (kl *Kubelet) GetHostIP() (net.IP, error) {
	node, err := kl.GetNode()
	if err != nil {
		return nil, fmt.Errorf("cannot get node: %v", err)
	}
	return nodeutil.GetNodeHostIP(node)
}

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pods.
func (kl *Kubelet) GetPods() []*api.Pod {
	return kl.podManager.GetPods()
}

// GetRunningPods returns all pods running on kubelet from looking at the
// container runtime cache. This function converts kubecontainer.Pod to
// api.Pod, so only the fields that exist in both kubecontainer.Pod and
// api.Pod are considered meaningful.
func (kl *Kubelet) GetRunningPods() ([]*api.Pod, error) {
	pods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return nil, err
	}

	apiPods := make([]*api.Pod, 0, len(pods))
	for _, pod := range pods {
		apiPods = append(apiPods, pod.ToAPIPod())
	}
	return apiPods, nil
}

func (kl *Kubelet) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	return kl.podManager.GetPodByFullName(podFullName)
}

// GetPodByName provides the first pod that matches namespace and name, as well
// as whether the pod was found.
func (kl *Kubelet) GetPodByName(namespace, name string) (*api.Pod, bool) {
	return kl.podManager.GetPodByName(namespace, name)
}

func (kl *Kubelet) updateRuntimeUp() {
	start := time.Now()
	err := waitUntilRuntimeIsUp(kl.containerRuntime, 100*time.Millisecond)
	kl.runtimeMutex.Lock()
	defer kl.runtimeMutex.Unlock()
	if err == nil {
		kl.lastTimestampRuntimeUp = time.Now()
	} else {
		glog.Errorf("Container runtime sanity check failed after %v, err: %v", time.Since(start), err)
	}
}

func (kl *Kubelet) reconcileCBR0(podCIDR string) error {
	if podCIDR == "" {
		glog.V(5).Info("PodCIDR not set. Will not configure cbr0.")
		return nil
	}
	glog.V(5).Infof("PodCIDR is set to %q", podCIDR)
	_, cidr, err := net.ParseCIDR(podCIDR)
	if err != nil {
		return err
	}
	// Set cbr0 interface address to first address in IPNet
	cidr.IP.To4()[3] += 1
	if err := ensureCbr0(cidr); err != nil {
		return err
	}
	if kl.shaper == nil {
		glog.V(5).Info("Shaper is nil, creating")
		kl.shaper = bandwidth.NewTCShaper("cbr0")
	}
	return kl.shaper.ReconcileInterface()
}

// updateNodeStatus updates node status to master with retries.
func (kl *Kubelet) updateNodeStatus() error {
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		if err := kl.tryUpdateNodeStatus(); err != nil {
			glog.Errorf("Error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("update node status exceeds retry count")
}

func (kl *Kubelet) recordNodeStatusEvent(event string) {
	glog.V(2).Infof("Recording %s event message for node %s", event, kl.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, event, "Node %s status is now: %s", kl.nodeName, event)
}

// Maintains Node.Spec.Unschedulable value from previous run of tryUpdateNodeStatus()
var oldNodeUnschedulable bool

func (kl *Kubelet) syncNetworkStatus() {
	kl.networkConfigMutex.Lock()
	defer kl.networkConfigMutex.Unlock()

	networkConfigured := true
	if kl.configureCBR0 {
		if err := ensureIPTablesMasqRule(); err != nil {
			networkConfigured = false
			glog.Errorf("Error on adding ip table rules: %v", err)
		}
		if len(kl.podCIDR) == 0 {
			glog.Warningf("ConfigureCBR0 requested, but PodCIDR not set. Will not configure CBR0 right now")
			networkConfigured = false
		} else if err := kl.reconcileCBR0(kl.podCIDR); err != nil {
			networkConfigured = false
			glog.Errorf("Error configuring cbr0: %v", err)
		}
	}
	kl.networkConfigured = networkConfigured
}

// setNodeStatus fills in the Status fields of the given Node, overwriting
// any fields that are currently set.
// TODO(madhusudancs): Simplify the logic for setting node conditions and
// refactor the node status condtion code out to a different file.
func (kl *Kubelet) setNodeStatus(node *api.Node) error {
	// Set addresses for the node.
	if kl.cloud != nil {
		instances, ok := kl.cloud.Instances()
		if !ok {
			return fmt.Errorf("failed to get instances from cloud provider")
		}
		// TODO(roberthbailey): Can we do this without having credentials to talk
		// to the cloud provider?
		// TODO(justinsb): We can if CurrentNodeName() was actually CurrentNode() and returned an interface
		nodeAddresses, err := instances.NodeAddresses(kl.nodeName)
		if err != nil {
			return fmt.Errorf("failed to get node address from cloud provider: %v", err)
		}
		node.Status.Addresses = nodeAddresses
	} else {
		addr := net.ParseIP(kl.hostname)
		if addr != nil {
			node.Status.Addresses = []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: addr.String()},
				{Type: api.NodeInternalIP, Address: addr.String()},
			}
		} else {
			addrs, err := net.LookupIP(node.Name)
			if err != nil {
				return fmt.Errorf("can't get ip address of node %s: %v", node.Name, err)
			} else if len(addrs) == 0 {
				return fmt.Errorf("no ip address for node %v", node.Name)
			} else {
				// check all ip addresses for this node.Name and try to find the first non-loopback IPv4 address.
				// If no match is found, it uses the IP of the interface with gateway on it.
				for _, ip := range addrs {
					if ip.IsLoopback() {
						continue
					}

					if ip.To4() != nil {
						node.Status.Addresses = []api.NodeAddress{
							{Type: api.NodeLegacyHostIP, Address: ip.String()},
							{Type: api.NodeInternalIP, Address: ip.String()},
						}
						break
					}
				}

				if len(node.Status.Addresses) == 0 {
					ip, err := util.ChooseHostInterface()
					if err != nil {
						return err
					}

					node.Status.Addresses = []api.NodeAddress{
						{Type: api.NodeLegacyHostIP, Address: ip.String()},
						{Type: api.NodeInternalIP, Address: ip.String()},
					}
				}
			}
		}
	}

	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		// TODO(roberthbailey): This is required for test-cmd.sh to pass.
		// See if the test should be updated instead.
		node.Status.Capacity = api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(0, resource.DecimalSI),
			api.ResourceMemory: resource.MustParse("0Gi"),
			api.ResourcePods:   *resource.NewQuantity(int64(kl.pods), resource.DecimalSI),
		}
		glog.Errorf("Error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Status.Capacity = CapacityFromMachineInfo(info)
		node.Status.Capacity[api.ResourcePods] = *resource.NewQuantity(
			int64(kl.pods), resource.DecimalSI)
		if node.Status.NodeInfo.BootID != "" &&
			node.Status.NodeInfo.BootID != info.BootID {
			// TODO: This requires a transaction, either both node status is updated
			// and event is recorded or neither should happen, see issue #6055.
			kl.recorder.Eventf(kl.nodeRef, "Rebooted",
				"Node %s has been rebooted, boot id: %s", kl.nodeName, info.BootID)
		}
		node.Status.NodeInfo.BootID = info.BootID
	}

	verinfo, err := kl.cadvisor.VersionInfo()
	if err != nil {
		glog.Errorf("Error getting version info: %v", err)
	} else {
		node.Status.NodeInfo.KernelVersion = verinfo.KernelVersion
		node.Status.NodeInfo.OsImage = verinfo.ContainerOsVersion
		// TODO: Determine the runtime is docker or rocket
		node.Status.NodeInfo.ContainerRuntimeVersion = "docker://" + verinfo.DockerVersion
		node.Status.NodeInfo.KubeletVersion = version.Get().String()
		// TODO: kube-proxy might be different version from kubelet in the future
		node.Status.NodeInfo.KubeProxyVersion = version.Get().String()
	}

	node.Status.DaemonEndpoints = *kl.daemonEndpoints

	// Check whether container runtime can be reported as up.
	containerRuntimeUp := kl.containerRuntimeUp()
	// Check whether network is configured properly
	networkConfigured := kl.doneNetworkConfigure()

	currentTime := unversioned.Now()
	var newNodeReadyCondition api.NodeCondition
	var oldNodeReadyConditionStatus api.ConditionStatus
	if containerRuntimeUp && networkConfigured {
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionTrue,
			Reason:            "KubeletReady",
			Message:           "kubelet is posting ready status",
			LastHeartbeatTime: currentTime,
		}
	} else {
		var reasons []string
		var messages []string
		if !containerRuntimeUp {
			messages = append(messages, "container runtime is down")
		}
		if !networkConfigured {
			messages = append(reasons, "network not configured correctly")
		}
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionFalse,
			Reason:            "KubeletNotReady",
			Message:           strings.Join(messages, ","),
			LastHeartbeatTime: currentTime,
		}
	}

	updated := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			oldNodeReadyConditionStatus = node.Status.Conditions[i].Status
			if oldNodeReadyConditionStatus == newNodeReadyCondition.Status {
				newNodeReadyCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			} else {
				newNodeReadyCondition.LastTransitionTime = currentTime
			}
			node.Status.Conditions[i] = newNodeReadyCondition
			updated = true
		}
	}
	if !updated {
		newNodeReadyCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newNodeReadyCondition)
	}
	if !updated || oldNodeReadyConditionStatus != newNodeReadyCondition.Status {
		if newNodeReadyCondition.Status == api.ConditionTrue {
			kl.recordNodeStatusEvent("NodeReady")
		} else {
			kl.recordNodeStatusEvent("NodeNotReady")
		}
	}

	var nodeOODCondition *api.NodeCondition

	// Check if NodeOutOfDisk condition already exists and if it does, just pick it up for update.
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeOutOfDisk {
			nodeOODCondition = &node.Status.Conditions[i]
		}
	}

	newOODCondition := false
	// If the NodeOutOfDisk condition doesn't exist, create one.
	if nodeOODCondition == nil {
		nodeOODCondition = &api.NodeCondition{
			Type:               api.NodeOutOfDisk,
			Status:             api.ConditionUnknown,
			LastTransitionTime: currentTime,
		}
		// nodeOODCondition cannot be appended to node.Status.Conditions here because it gets
		// copied to the slice. So if we append nodeOODCondition to the slice here none of the
		// updates we make to nodeOODCondition below are reflected in the slice.
		newOODCondition = true
	}

	// Update the heartbeat time irrespective of all the conditions.
	nodeOODCondition.LastHeartbeatTime = currentTime

	// Note: The conditions below take care of the case when a new NodeOutOfDisk condition is
	// created and as well as the case when the condition already exists. When a new condition
	// is created its status is set to api.ConditionUnknown which matches either
	// nodeOODCondition.Status != api.ConditionTrue or
	// nodeOODCondition.Status != api.ConditionFalse in the conditions below depending on whether
	// the kubelet is out of disk or not.
	if kl.isOutOfDisk() {
		if nodeOODCondition.Status != api.ConditionTrue {
			nodeOODCondition.Status = api.ConditionTrue
			nodeOODCondition.Reason = "KubeletOutOfDisk"
			nodeOODCondition.Message = "out of disk space"
			nodeOODCondition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent("NodeOutOfDisk")
		}
	} else {
		if nodeOODCondition.Status != api.ConditionFalse {
			nodeOODCondition.Status = api.ConditionFalse
			nodeOODCondition.Reason = "KubeletHasSufficientDisk"
			nodeOODCondition.Message = "kubelet has sufficient disk space available"
			nodeOODCondition.LastTransitionTime = currentTime
			kl.recordNodeStatusEvent("NodeHasSufficientDisk")
		}
	}

	if newOODCondition {
		node.Status.Conditions = append(node.Status.Conditions, *nodeOODCondition)
	}

	if oldNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			kl.recordNodeStatusEvent("NodeNotSchedulable")
		} else {
			kl.recordNodeStatusEvent("NodeSchedulable")
		}
		oldNodeUnschedulable = node.Spec.Unschedulable
	}
	return nil
}

func (kl *Kubelet) containerRuntimeUp() bool {
	kl.runtimeMutex.Lock()
	defer kl.runtimeMutex.Unlock()
	return kl.lastTimestampRuntimeUp.Add(kl.runtimeUpThreshold).After(time.Now())
}

func (kl *Kubelet) doneNetworkConfigure() bool {
	kl.networkConfigMutex.Lock()
	defer kl.networkConfigMutex.Unlock()
	return kl.networkConfigured
}

// tryUpdateNodeStatus tries to update node status to master. If ReconcileCBR0
// is set, this function will also confirm that cbr0 is configured correctly.
func (kl *Kubelet) tryUpdateNodeStatus() error {
	node, err := kl.kubeClient.Nodes().Get(kl.nodeName)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", kl.nodeName, err)
	}
	if node == nil {
		return fmt.Errorf("no node instance returned for %q", kl.nodeName)
	}
	kl.networkConfigMutex.Lock()
	kl.podCIDR = node.Spec.PodCIDR
	kl.networkConfigMutex.Unlock()

	if err := kl.setNodeStatus(node); err != nil {
		return err
	}
	// Update the current status on the API server
	_, err = kl.kubeClient.Nodes().UpdateStatus(node)
	return err
}

// GetPhase returns the phase of a pod given its container info.
// This func is exported to simplify integration with 3rd party kubelet
// integrations like kubernetes-mesos.
func GetPhase(spec *api.PodSpec, info []api.ContainerStatus) api.PodPhase {
	running := 0
	waiting := 0
	stopped := 0
	failed := 0
	succeeded := 0
	unknown := 0
	for _, container := range spec.Containers {
		if containerStatus, ok := api.GetContainerStatus(info, container.Name); ok {
			if containerStatus.State.Running != nil {
				running++
			} else if containerStatus.State.Terminated != nil {
				stopped++
				if containerStatus.State.Terminated.ExitCode == 0 {
					succeeded++
				} else {
					failed++
				}
			} else if containerStatus.State.Waiting != nil {
				if containerStatus.LastTerminationState.Terminated != nil {
					stopped++
				} else {
					waiting++
				}
			} else {
				unknown++
			}
		} else {
			unknown++
		}
	}
	switch {
	case waiting > 0:
		glog.V(5).Infof("pod waiting > 0, pending")
		// One or more containers has not been started
		return api.PodPending
	case running > 0 && unknown == 0:
		// All containers have been started, and at least
		// one container is running
		return api.PodRunning
	case running == 0 && stopped > 0 && unknown == 0:
		// All containers are terminated
		if spec.RestartPolicy == api.RestartPolicyAlways {
			// All containers are in the process of restarting
			return api.PodRunning
		}
		if stopped == succeeded {
			// RestartPolicy is not Always, and all
			// containers are terminated in success
			return api.PodSucceeded
		}
		if spec.RestartPolicy == api.RestartPolicyNever {
			// RestartPolicy is Never, and all containers are
			// terminated with at least one in failure
			return api.PodFailed
		}
		// RestartPolicy is OnFailure, and at least one in failure
		// and in the process of restarting
		return api.PodRunning
	default:
		glog.V(5).Infof("pod default case, pending")
		return api.PodPending
	}
}

// Disabled LastProbeTime/LastTranitionTime for Pods to avoid constantly sending pod status
// update to the apiserver. See http://issues.k8s.io/14273. Functional revert of a PR #12894

// getPodReadyCondition returns ready condition if all containers in a pod are ready, else it returns an unready condition.
func getPodReadyCondition(spec *api.PodSpec, containerStatuses []api.ContainerStatus, existingStatus *api.PodStatus) []api.PodCondition {
	ready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionTrue,
	}}
	notReady := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionFalse,
	}}
	if containerStatuses == nil {
		return notReady
	}
	for _, container := range spec.Containers {
		if containerStatus, ok := api.GetContainerStatus(containerStatuses, container.Name); ok {
			if !containerStatus.Ready {
				return notReady
			}
		} else {
			return notReady
		}
	}
	return ready
}

// By passing the pod directly, this method avoids pod lookup, which requires
// grabbing a lock.
func (kl *Kubelet) generatePodStatus(pod *api.Pod) (api.PodStatus, error) {

	start := time.Now()
	defer func() {
		metrics.PodStatusLatency.Observe(metrics.SinceInMicroseconds(start))
	}()

	podFullName := kubecontainer.GetPodFullName(pod)
	glog.V(3).Infof("Generating status for %q", podFullName)
	if existingStatus, hasExistingStatus := kl.statusManager.GetPodStatus(pod.UID); hasExistingStatus {
		// This is a hacky fix to ensure container restart counts increment
		// monotonically. Normally, we should not modify given pod. In this
		// case, we check if there are cached status for this pod, and update
		// the pod so that we update restart count appropriately.
		// TODO(yujuhong): We will not need to count dead containers every time
		// once we add the runtime pod cache.
		// Note that kubelet restarts may still cause temporarily setback of
		// restart counts.
		pod.Status = existingStatus
	}

	// TODO: Consider include the container information.
	if kl.pastActiveDeadline(pod) {
		reason := "DeadlineExceeded"
		kl.recorder.Eventf(pod, reason, "Pod was active on the node longer than specified deadline")
		return api.PodStatus{
			Phase:   api.PodFailed,
			Reason:  reason,
			Message: "Pod was active on the node longer than specified deadline"}, nil
	}

	spec := &pod.Spec
	podStatus, err := kl.containerRuntime.GetPodStatus(pod)

	if err != nil {
		// Error handling
		glog.Infof("Query container info for pod %q failed with error (%v)", podFullName, err)
		if strings.Contains(err.Error(), "resource temporarily unavailable") {
			// Leave upstream layer to decide what to do
			return api.PodStatus{}, err
		}

		pendingStatus := api.PodStatus{
			Phase:   api.PodPending,
			Reason:  "GeneralError",
			Message: fmt.Sprintf("Query container info failed with error (%v)", err),
		}
		return pendingStatus, nil
	}

	// Assume info is ready to process
	podStatus.Phase = GetPhase(spec, podStatus.ContainerStatuses)
	for _, c := range spec.Containers {
		for i, st := range podStatus.ContainerStatuses {
			if st.Name == c.Name {
				ready := st.State.Running != nil && kl.readinessManager.GetReadiness(kubecontainer.TrimRuntimePrefix(st.ContainerID))
				podStatus.ContainerStatuses[i].Ready = ready
				break
			}
		}
	}
	podStatus.Conditions = append(podStatus.Conditions, getPodReadyCondition(spec, podStatus.ContainerStatuses, nil /* unused */)...)

	if !kl.standaloneMode {
		hostIP, err := kl.GetHostIP()
		if err != nil {
			glog.V(4).Infof("Cannot get host IP: %v", err)
		} else {
			podStatus.HostIP = hostIP.String()
			if pod.Spec.HostNetwork && podStatus.PodIP == "" {
				podStatus.PodIP = hostIP.String()
			}
		}
	}

	return *podStatus, nil
}

// Returns logs of current machine.
func (kl *Kubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	// TODO: whitelist logs we are willing to serve
	kl.logServer.ServeHTTP(w, req)
}

// findContainer finds and returns the container with the given pod ID, full name, and container name.
// It returns nil if not found.
func (kl *Kubelet) findContainer(podFullName string, podUID types.UID, containerName string) (*kubecontainer.Container, error) {
	pods, err := kl.containerRuntime.GetPods(false)
	if err != nil {
		return nil, err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	return pod.FindContainerByName(containerName), nil
}

// Run a command in a container, returns the combined stdout, stderr as an array of bytes
func (kl *Kubelet) RunInContainer(podFullName string, podUID types.UID, containerName string, cmd []string) ([]byte, error) {
	podUID = kl.podManager.TranslatePodUID(podUID)

	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return nil, err
	}
	if container == nil {
		return nil, fmt.Errorf("container not found (%q)", containerName)
	}
	return kl.runner.RunInContainer(string(container.ID), cmd)
}

// ExecInContainer executes a command in a container, connecting the supplied
// stdin/stdout/stderr to the command's IO streams.
func (kl *Kubelet) ExecInContainer(podFullName string, podUID types.UID, containerName string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	podUID = kl.podManager.TranslatePodUID(podUID)

	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return err
	}
	if container == nil {
		return fmt.Errorf("container not found (%q)", containerName)
	}
	return kl.runner.ExecInContainer(string(container.ID), cmd, stdin, stdout, stderr, tty)
}

func (kl *Kubelet) AttachContainer(podFullName string, podUID types.UID, containerName string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	podUID = kl.podManager.TranslatePodUID(podUID)

	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return err
	}
	if container == nil {
		return fmt.Errorf("container not found (%q)", containerName)
	}
	return kl.containerRuntime.AttachContainer(string(container.ID), stdin, stdout, stderr, tty)
}

// PortForward connects to the pod's port and copies data between the port
// and the stream.
func (kl *Kubelet) PortForward(podFullName string, podUID types.UID, port uint16, stream io.ReadWriteCloser) error {
	podUID = kl.podManager.TranslatePodUID(podUID)

	pods, err := kl.containerRuntime.GetPods(false)
	if err != nil {
		return err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	if pod.IsEmpty() {
		return fmt.Errorf("pod not found (%q)", podFullName)
	}
	return kl.runner.PortForward(&pod, port, stream)
}

// BirthCry sends an event that the kubelet has started up.
func (kl *Kubelet) BirthCry() {
	// Make an event that kubelet restarted.
	kl.recorder.Eventf(kl.nodeRef, "Starting", "Starting kubelet.")
}

func (kl *Kubelet) StreamingConnectionIdleTimeout() time.Duration {
	return kl.streamingConnectionIdleTimeout
}

func (kl *Kubelet) ResyncInterval() time.Duration {
	return kl.resyncInterval
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName string, podUID types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {

	podUID = kl.podManager.TranslatePodUID(podUID)

	pods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return nil, err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	container := pod.FindContainerByName(containerName)
	if container == nil {
		return nil, ErrContainerNotFound
	}

	ci, err := kl.cadvisor.DockerContainer(string(container.ID), req)
	if err != nil {
		return nil, err
	}
	return &ci, nil
}

// Returns stats (from Cadvisor) for a non-Kubernetes container.
func (kl *Kubelet) GetRawContainerInfo(containerName string, req *cadvisorApi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorApi.ContainerInfo, error) {
	if subcontainers {
		return kl.cadvisor.SubcontainerInfo(containerName, req)
	} else {
		containerInfo, err := kl.cadvisor.ContainerInfo(containerName, req)
		if err != nil {
			return nil, err
		}
		return map[string]*cadvisorApi.ContainerInfo{
			containerInfo.Name: containerInfo,
		}, nil
	}
}

// GetCachedMachineInfo assumes that the machine info can't change without a reboot
func (kl *Kubelet) GetCachedMachineInfo() (*cadvisorApi.MachineInfo, error) {
	if kl.machineInfo == nil {
		info, err := kl.cadvisor.MachineInfo()
		if err != nil {
			return nil, err
		}
		kl.machineInfo = info
	}
	return kl.machineInfo, nil
}

func (kl *Kubelet) ListenAndServe(address net.IP, port uint, tlsOptions *TLSOptions, enableDebuggingHandlers bool) {
	ListenAndServeKubeletServer(kl, address, port, tlsOptions, enableDebuggingHandlers)
}

func (kl *Kubelet) ListenAndServeReadOnly(address net.IP, port uint) {
	ListenAndServeKubeletReadOnlyServer(kl, address, port)
}

// GetRuntime returns the current Runtime implementation in use by the kubelet. This func
// is exported to simplify integration with third party kubelet extensions (e.g. kubernetes-mesos).
func (kl *Kubelet) GetRuntime() kubecontainer.Runtime {
	return kl.containerRuntime
}

var minRsrc = resource.MustParse("1k")
var maxRsrc = resource.MustParse("1P")

func validateBandwidthIsReasonable(rsrc *resource.Quantity) error {
	if rsrc.Value() < minRsrc.Value() {
		return fmt.Errorf("resource is unreasonably small (< 1kbit)")
	}
	if rsrc.Value() > maxRsrc.Value() {
		return fmt.Errorf("resoruce is unreasonably large (> 1Pbit)")
	}
	return nil
}

func extractBandwidthResources(pod *api.Pod) (ingress, egress *resource.Quantity, err error) {
	str, found := pod.Annotations["kubernetes.io/ingress-bandwidth"]
	if found {
		if ingress, err = resource.ParseQuantity(str); err != nil {
			return nil, nil, err
		}
		if err := validateBandwidthIsReasonable(ingress); err != nil {
			return nil, nil, err
		}
	}
	str, found = pod.Annotations["kubernetes.io/egress-bandwidth"]
	if found {
		if egress, err = resource.ParseQuantity(str); err != nil {
			return nil, nil, err
		}
		if err := validateBandwidthIsReasonable(egress); err != nil {
			return nil, nil, err
		}
	}
	return ingress, egress, nil
}
