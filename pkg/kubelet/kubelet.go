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
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/envvars"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/pleg"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/rkt"
	"k8s.io/kubernetes/pkg/kubelet/server"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/util/queue"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/atomic"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	kubeio "k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/oom"
	"k8s.io/kubernetes/pkg/util/procfs"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/selinux"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/util/wait"
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

	// max backoff period, exported for the e2e test
	MaxContainerBackOff = 300 * time.Second

	// Capacity of the channel for storing pods to kill. A small number should
	// suffice because a goroutine is dedicated to check the channel and does
	// not block on anything else.
	podKillingChannelCapacity = 50

	// Period for performing global cleanup tasks.
	housekeepingPeriod = time.Second * 2

	etcHostsPath = "/etc/hosts"

	// Capacity of the channel for recieving pod lifecycle events. This number
	// is a bit arbitrary and may be adjusted in the future.
	plegChannelCapacity = 1000

	// Generic PLEG relies on relisting for discovering container events.
	// A longer period means that kubelet will take longer to detect container
	// changes and to update pod status. On the other hand, a shorter period
	// will cause more frequent relisting (e.g., container runtime operations),
	// leading to higher cpu usage.
	// Note that even though we set the period to 1s, the relisting itself can
	// take more than 1s to finish if the container runtime responds slowly
	// and/or when there are many container changes in one cycle.
	plegRelistPeriod = time.Second * 1

	// backOffPeriod is the period to back off when pod syncing resulting in an
	// error. It is also used as the base period for the exponential backoff
	// container restarts and image pulls.
	backOffPeriod = time.Second * 10
)

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {
	HandlePodAdditions(pods []*api.Pod)
	HandlePodUpdates(pods []*api.Pod)
	HandlePodDeletions(pods []*api.Pod)
	HandlePodReconcile(pods []*api.Pod)
	HandlePodSyncs(pods []*api.Pod)
	HandlePodCleanups() error
}

type SourcesReadyFn func(sourcesSeen sets.String) bool

// New instantiates a new Kubelet object along with all the required internal modules.
// No initialization of Kubelet and its modules should happen here.
func NewMainKubelet(
	hostname string,
	nodeName string,
	dockerClient dockertools.DockerInterface,
	kubeClient clientset.Interface,
	rootDirectory string,
	podInfraContainerImage string,
	resyncInterval time.Duration,
	pullQPS float32,
	pullBurst int,
	eventQPS float32,
	eventBurst int,
	containerGCPolicy kubecontainer.ContainerGCPolicy,
	sourcesReady SourcesReadyFn,
	registerNode bool,
	registerSchedulable bool,
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
	nodeLabels map[string]string,
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
	nonMasqueradeCIDR string,
	podCIDR string,
	reconcileCIDR bool,
	maxPods int,
	dockerExecHandler dockertools.ExecHandler,
	resolverConfig string,
	cpuCFSQuota bool,
	daemonEndpoints *api.NodeDaemonEndpoints,
	oomAdjuster *oom.OOMAdjuster,
	serializeImagePulls bool,
	containerManager cm.ContainerManager,
	outOfDiskTransitionFrequency time.Duration,
	flannelExperimentalOverlay bool,
	nodeIP net.IP,
	reservation kubetypes.Reservation,
	enableCustomMetrics bool,
	volumeStatsAggPeriod time.Duration,
	containerRuntimeOptions []kubecontainer.Option,
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
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return kubeClient.Core().Services(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return kubeClient.Core().Services(api.NamespaceAll).Watch(options)
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
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				options.FieldSelector = fieldSelector
				return kubeClient.Core().Nodes().List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				options.FieldSelector = fieldSelector
				return kubeClient.Core().Nodes().Watch(options)
			},
		}
		cache.NewReflector(listWatch, &api.Node{}, nodeStore, 0).Run()
	}
	nodeLister := &cache.StoreToNodeLister{Store: nodeStore}
	nodeInfo := &predicates.CachedNodeInfo{nodeLister}

	// TODO: get the real node object of ourself,
	// and use the real node name and UID.
	// TODO: what is namespace for node?
	nodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      nodeName,
		UID:       types.UID(nodeName),
		Namespace: "",
	}

	diskSpaceManager, err := newDiskSpaceManager(cadvisorInterface, diskSpacePolicy)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize disk manager: %v", err)
	}
	containerRefManager := kubecontainer.NewRefManager()

	volumeManager := newVolumeManager()

	oomWatcher := NewOOMWatcher(cadvisorInterface, recorder)

	// TODO: remove when internal cbr0 implementation gets removed in favor
	// of the kubenet network plugin
	if networkPluginName == "kubenet" {
		configureCBR0 = false
		flannelExperimentalOverlay = false
	}

	klet := &Kubelet{
		hostname:                       hostname,
		nodeName:                       nodeName,
		dockerClient:                   dockerClient,
		kubeClient:                     kubeClient,
		rootDirectory:                  rootDirectory,
		resyncInterval:                 resyncInterval,
		containerRefManager:            containerRefManager,
		httpClient:                     &http.Client{},
		sourcesReady:                   sourcesReady,
		registerNode:                   registerNode,
		registerSchedulable:            registerSchedulable,
		standaloneMode:                 standaloneMode,
		clusterDomain:                  clusterDomain,
		clusterDNS:                     clusterDNS,
		serviceLister:                  serviceLister,
		nodeLister:                     nodeLister,
		nodeInfo:                       nodeInfo,
		masterServiceNamespace:         masterServiceNamespace,
		streamingConnectionIdleTimeout: streamingConnectionIdleTimeout,
		recorder:                       recorder,
		cadvisor:                       cadvisorInterface,
		diskSpaceManager:               diskSpaceManager,
		volumeManager:                  volumeManager,
		cloud:                          cloud,
		nodeRef:                        nodeRef,
		nodeLabels:                     nodeLabels,
		nodeStatusUpdateFrequency:      nodeStatusUpdateFrequency,
		resourceContainer:              resourceContainer,
		os:                             osInterface,
		oomWatcher:                     oomWatcher,
		cgroupRoot:                     cgroupRoot,
		mounter:                        mounter,
		writer:                         writer,
		configureCBR0:                  configureCBR0,
		nonMasqueradeCIDR:              nonMasqueradeCIDR,
		reconcileCIDR:                  reconcileCIDR,
		maxPods:                        maxPods,
		syncLoopMonitor:                atomic.Value{},
		resolverConfig:                 resolverConfig,
		cpuCFSQuota:                    cpuCFSQuota,
		daemonEndpoints:                daemonEndpoints,
		containerManager:               containerManager,
		flannelExperimentalOverlay:     flannelExperimentalOverlay,
		flannelHelper:                  NewFlannelHelper(),
		nodeIP:                         nodeIP,
		clock:                          util.RealClock{},
		outOfDiskTransitionFrequency: outOfDiskTransitionFrequency,
		reservation:                  reservation,
		enableCustomMetrics:          enableCustomMetrics,
	}
	// TODO: Factor out "StatsProvider" from Kubelet so we don't have a cyclic dependency
	klet.resourceAnalyzer = stats.NewResourceAnalyzer(klet, volumeStatsAggPeriod)

	if klet.flannelExperimentalOverlay {
		glog.Infof("Flannel is in charge of podCIDR and overlay networking.")
	}
	if klet.nodeIP != nil {
		if err := klet.validateNodeIP(); err != nil {
			return nil, err
		}
		glog.Infof("Using node IP: %q", klet.nodeIP.String())
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

	procFs := procfs.NewProcFS()
	imageBackOff := util.NewBackOff(backOffPeriod, MaxContainerBackOff)

	klet.livenessManager = proberesults.NewManager()

	klet.podCache = kubecontainer.NewCache()

	// Initialize the runtime.
	switch containerRuntime {
	case "docker":
		// Only supported one for now, continue.
		klet.containerRuntime = dockertools.NewDockerManager(
			dockerClient,
			kubecontainer.FilterEventRecorder(recorder),
			klet.livenessManager,
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
			imageBackOff,
			serializeImagePulls,
			enableCustomMetrics,
			containerRuntimeOptions...,
		)
	case "rkt":
		conf := &rkt.Config{
			Path:            rktPath,
			Stage1Image:     rktStage1Image,
			InsecureOptions: "image,ondisk",
		}
		rktRuntime, err := rkt.New(
			conf,
			klet,
			recorder,
			containerRefManager,
			klet.livenessManager,
			klet.volumeManager,
			imageBackOff,
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

	klet.pleg = pleg.NewGenericPLEG(klet.containerRuntime, plegChannelCapacity, plegRelistPeriod, klet.podCache)
	klet.runtimeState = newRuntimeState(maxWaitForContainerRuntime, configureCBR0, klet.isContainerRuntimeVersionCompatible)
	klet.updatePodCIDR(podCIDR)

	// setup containerGC
	containerGC, err := kubecontainer.NewContainerGC(klet.containerRuntime, containerGCPolicy)
	if err != nil {
		return nil, err
	}
	klet.containerGC = containerGC

	// setup imageManager
	imageManager, err := newImageManager(klet.containerRuntime, cadvisorInterface, recorder, nodeRef, imageGCPolicy)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize image manager: %v", err)
	}
	klet.imageManager = imageManager

	// Setup container manager, can fail if the devices hierarchy is not mounted
	// (it is required by Docker however).
	klet.nodeConfig = cm.NodeConfig{
		DockerDaemonContainerName: dockerDaemonContainer,
		SystemContainerName:       systemContainer,
		KubeletContainerName:      resourceContainer,
	}
	klet.runtimeState.setRuntimeSync(klet.clock.Now())

	klet.runner = klet.containerRuntime
	klet.podManager = kubepod.NewBasicPodManager(kubepod.NewBasicMirrorClient(klet.kubeClient))
	klet.statusManager = status.NewManager(kubeClient, klet.podManager)

	klet.probeManager = prober.NewManager(
		klet.statusManager,
		klet.livenessManager,
		klet.runner,
		containerRefManager,
		recorder)

	if err := klet.volumePluginMgr.InitPlugins(volumePlugins, &volumeHost{klet}); err != nil {
		return nil, err
	}

	runtimeCache, err := kubecontainer.NewRuntimeCache(klet.containerRuntime)
	if err != nil {
		return nil, err
	}
	klet.runtimeCache = runtimeCache
	klet.reasonCache = NewReasonCache()
	klet.workQueue = queue.NewBasicWorkQueue()
	klet.podWorkers = newPodWorkers(klet.syncPod, recorder, klet.workQueue, klet.resyncInterval, backOffPeriod, klet.podCache)

	klet.backOff = util.NewBackOff(backOffPeriod, MaxContainerBackOff)
	klet.podKillingCh = make(chan *kubecontainer.PodPair, podKillingChannelCapacity)
	klet.sourcesSeen = sets.NewString()
	return klet, nil
}

type serviceLister interface {
	List() (api.ServiceList, error)
}

type nodeLister interface {
	List() (machines api.NodeList, err error)
}

// Kubelet is the main kubelet implementation.
type Kubelet struct {
	hostname      string
	nodeName      string
	dockerClient  dockertools.DockerInterface
	runtimeCache  kubecontainer.RuntimeCache
	kubeClient    clientset.Interface
	rootDirectory string
	podWorkers    PodWorkers

	resyncInterval time.Duration
	sourcesReady   SourcesReadyFn
	// sourcesSeen records the sources seen by kubelet. This set is not thread
	// safe and should only be access by the main kubelet syncloop goroutine.
	sourcesSeen sets.String

	podManager kubepod.Manager

	// Needed to report events for containers belonging to deleted/modified pods.
	// Tracks references for reporting events
	containerRefManager *kubecontainer.RefManager

	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
	// Optional, defaults to simple Docker implementation
	runner kubecontainer.ContainerCommandRunner
	// Optional, client for http requests, defaults to empty client
	httpClient kubetypes.HttpGetter

	// cAdvisor used for container information.
	cadvisor cadvisor.Interface

	// Set to true to have the node register itself with the apiserver.
	registerNode bool
	// Set to true to have the node register itself as schedulable.
	registerSchedulable bool
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
	nodeInfo               predicates.NodeInfo

	// a list of node labels to register
	nodeLabels map[string]string

	// Last timestamp when runtime responded on ping.
	// Mutex is used to protect this value.
	runtimeState *runtimeState

	// Volume plugins.
	volumePluginMgr volume.VolumePluginMgr

	// Network plugin.
	networkPlugin network.NetworkPlugin

	// Handles container probing.
	probeManager prober.Manager
	// Manages container health check results.
	livenessManager proberesults.Manager

	// How long to keep idle streaming command execution/port forwarding
	// connections open before terminating them
	streamingConnectionIdleTimeout time.Duration

	// The EventRecorder to use
	recorder record.EventRecorder

	// Policy for handling garbage collection of dead containers.
	containerGC kubecontainer.ContainerGC

	// Manager for images.
	imageManager imageManager

	// Diskspace manager.
	diskSpaceManager diskSpaceManager

	// Cached MachineInfo returned by cadvisor.
	machineInfo *cadvisorapi.MachineInfo

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

	// reasonCache caches the failure reason of the last creation of all containers, which is
	// used for generating ContainerStatus.
	reasonCache *ReasonCache

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

	// Generates pod events.
	pleg pleg.PodLifecycleEventGenerator

	// Store kubecontainer.PodStatus for all pods.
	podCache kubecontainer.Cache

	// The name of the resource-only container to run the Kubelet in (empty for no container).
	// Name must be absolute.
	resourceContainer string

	os kubecontainer.OSInterface

	// Watcher of out of memory events.
	oomWatcher OOMWatcher

	// Monitor resource usage
	resourceAnalyzer stats.ResourceAnalyzer

	// If non-empty, pass this to the container runtime as the root cgroup.
	cgroupRoot string

	// Mounter to use for volumes.
	mounter mount.Interface

	// Writer interface to use for volumes.
	writer kubeio.Writer

	// Manager of non-Runtime containers.
	containerManager cm.ContainerManager
	nodeConfig       cm.NodeConfig

	// Whether or not kubelet should take responsibility for keeping cbr0 in
	// the correct state.
	configureCBR0 bool
	reconcileCIDR bool

	// Traffic to IPs outside this range will use IP masquerade.
	nonMasqueradeCIDR string

	// Maximum Number of Pods which can be run by this Kubelet
	maxPods int

	// Monitor Kubelet's sync loop
	syncLoopMonitor atomic.Value

	// Container restart Backoff
	backOff *util.Backoff

	// Channel for sending pods to kill.
	podKillingCh chan *kubecontainer.PodPair

	// The configuration file used as the base to generate the container's
	// DNS resolver configuration file. This can be used in conjunction with
	// clusterDomain and clusterDNS.
	resolverConfig string

	// Optionally shape the bandwidth of a pod
	// TODO: remove when kubenet plugin is ready
	shaper bandwidth.BandwidthShaper

	// True if container cpu limits should be enforced via cgroup CFS quota
	cpuCFSQuota bool

	// Information about the ports which are opened by daemons on Node running this Kubelet server.
	daemonEndpoints *api.NodeDaemonEndpoints

	// A queue used to trigger pod workers.
	workQueue queue.WorkQueue

	// oneTimeInitializer is used to initialize modules that are dependent on the runtime to be up.
	oneTimeInitializer sync.Once

	flannelExperimentalOverlay bool

	// TODO: Flannelhelper doesn't store any state, we can instantiate it
	// on the fly if we're confident the dbus connetions it opens doesn't
	// put the system under duress.
	flannelHelper *FlannelHelper

	// If non-nil, use this IP address for the node
	nodeIP net.IP

	// clock is an interface that provides time related functionality in a way that makes it
	// easy to test the code.
	clock util.Clock

	// outOfDiskTransitionFrequency specifies the amount of time the kubelet has to be actually
	// not out of disk before it can transition the node condition status from out-of-disk to
	// not-out-of-disk. This prevents a pod that causes out-of-disk condition from repeatedly
	// getting rescheduled onto the node.
	outOfDiskTransitionFrequency time.Duration

	// reservation specifies resources which are reserved for non-pod usage, including kubernetes and
	// non-kubernetes system processes.
	reservation kubetypes.Reservation

	// support gathering custom metrics.
	enableCustomMetrics bool
}

// Validate given node IP belongs to the current host
func (kl *Kubelet) validateNodeIP() error {
	if kl.nodeIP == nil {
		return nil
	}

	// Honor IP limitations set in setNodeStatus()
	if kl.nodeIP.IsLoopback() {
		return fmt.Errorf("nodeIP can't be loopback address")
	}
	if kl.nodeIP.To4() == nil {
		return fmt.Errorf("nodeIP must be IPv4 address")
	}

	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return err
	}
	for _, addr := range addrs {
		var ip net.IP
		switch v := addr.(type) {
		case *net.IPNet:
			ip = v.IP
		case *net.IPAddr:
			ip = v.IP
		}
		if ip != nil && ip.Equal(kl.nodeIP) {
			return nil
		}
	}
	return fmt.Errorf("Node IP: %q not found in the host's network interfaces", kl.nodeIP.String())
}

func (kl *Kubelet) allSourcesReady() bool {
	// Make a copy of the sourcesSeen list because it's not thread-safe.
	return kl.sourcesReady(sets.NewString(kl.sourcesSeen.List()...))
}

func (kl *Kubelet) addSource(source string) {
	kl.sourcesSeen.Insert(source)
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
	return kl.nodeInfo.GetNodeInfo(kl.nodeName)
}

// Starts garbage collection threads.
func (kl *Kubelet) StartGarbageCollection() {
	go wait.Until(func() {
		if err := kl.containerGC.GarbageCollect(); err != nil {
			glog.Errorf("Container garbage collection failed: %v", err)
		}
	}, time.Minute, wait.NeverStop)

	go wait.Until(func() {
		if err := kl.imageManager.GarbageCollect(); err != nil {
			glog.Errorf("Image garbage collection failed: %v", err)
		}
	}, 5*time.Minute, wait.NeverStop)
}

// initializeModules will initialize internal modules that do not require the container runtime to be up.
// Note that the modules here must not depend on modules that are not initialized here.
func (kl *Kubelet) initializeModules() error {
	// Step 1: Move Kubelet to a container, if required.
	if kl.resourceContainer != "" {
		// Fixme: I need to reside inside ContainerManager interface.
		err := util.RunInResourceContainer(kl.resourceContainer)
		if err != nil {
			glog.Warningf("Failed to move Kubelet to container %q: %v", kl.resourceContainer, err)
		}
		glog.Infof("Running in container %q", kl.resourceContainer)
	}

	// Step 2: Promethues metrics.
	metrics.Register(kl.runtimeCache)

	// Step 3: Setup filesystem directories.
	if err := kl.setupDataDirs(); err != nil {
		return err
	}

	// Step 4: If the container logs directory does not exist, create it.
	if _, err := os.Stat(containerLogsDir); err != nil {
		if err := kl.os.Mkdir(containerLogsDir, 0755); err != nil {
			glog.Errorf("Failed to create directory %q: %v", containerLogsDir, err)
		}
	}

	// Step 5: Start the image manager.
	if err := kl.imageManager.Start(); err != nil {
		return fmt.Errorf("Failed to start ImageManager, images may not be garbage collected: %v", err)
	}

	// Step 6: Start container manager.
	if err := kl.containerManager.Start(kl.nodeConfig); err != nil {
		return fmt.Errorf("Failed to start ContainerManager %v", err)
	}

	// Step 7: Start out of memory watcher.
	if err := kl.oomWatcher.Start(kl.nodeRef); err != nil {
		return fmt.Errorf("Failed to start OOM watcher %v", err)
	}

	// Step 7: Start resource analyzer
	kl.resourceAnalyzer.Start()
	return nil
}

// initializeRuntimeDependentModules will initialize internal modules that require the container runtime to be up.
func (kl *Kubelet) initializeRuntimeDependentModules() {
	if err := kl.cadvisor.Start(); err != nil {
		kl.runtimeState.setInternalError(fmt.Errorf("Failed to start cAdvisor %v", err))
	}
}

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan kubetypes.PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.kubeClient == nil {
		glog.Warning("No api server defined - no node status update will be sent.")
	}
	if err := kl.initializeModules(); err != nil {
		kl.recorder.Eventf(kl.nodeRef, api.EventTypeWarning, kubecontainer.KubeletSetupFailed, err.Error())
		glog.Error(err)
		kl.runtimeState.setInitError(err)
	}

	if kl.kubeClient != nil {
		// Start syncing node status immediately, this may set up things the runtime needs to run.
		go wait.Until(kl.syncNodeStatus, kl.nodeStatusUpdateFrequency, wait.NeverStop)
	}
	go wait.Until(kl.syncNetworkStatus, 30*time.Second, wait.NeverStop)
	go wait.Until(kl.updateRuntimeUp, 5*time.Second, wait.NeverStop)

	// Start a goroutine responsible for killing pods (that are not properly
	// handled by pod workers).
	go wait.Until(kl.podKiller, 1*time.Second, wait.NeverStop)

	// Start component sync loops.
	kl.statusManager.Start()
	kl.probeManager.Start()
	// Start the pod lifecycle event generator.
	kl.pleg.Start()
	kl.syncLoop(updates, kl)
}

func (kl *Kubelet) initialNodeStatus() (*api.Node, error) {
	node := &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name:   kl.nodeName,
			Labels: map[string]string{"kubernetes.io/hostname": kl.hostname},
		},
		Spec: api.NodeSpec{
			Unschedulable: !kl.registerSchedulable,
		},
	}

	// @question: should this be place after the call to the cloud provider? which also applies labels
	for k, v := range kl.nodeLabels {
		if cv, found := node.ObjectMeta.Labels[k]; found {
			glog.Warningf("the node label %s=%s will overwrite default setting %s", k, v, cv)
		}
		node.ObjectMeta.Labels[k] = v
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

		// If the cloud has zone information, label the node with the zone information
		zones, ok := kl.cloud.Zones()
		if ok {
			zone, err := zones.GetZone()
			if err != nil {
				return nil, fmt.Errorf("failed to get zone from cloud provider: %v", err)
			}
			if zone.FailureDomain != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", unversioned.LabelZoneFailureDomain, zone.FailureDomain)
				node.ObjectMeta.Labels[unversioned.LabelZoneFailureDomain] = zone.FailureDomain
			}
			if zone.Region != "" {
				glog.Infof("Adding node label from cloud provider: %s=%s", unversioned.LabelZoneRegion, zone.Region)
				node.ObjectMeta.Labels[unversioned.LabelZoneRegion] = zone.Region
			}
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
		if _, err := kl.kubeClient.Core().Nodes().Create(node); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				glog.V(2).Infof("Unable to register %s with the apiserver: %v", node.Name, err)
				continue
			}
			currentNode, err := kl.kubeClient.Core().Nodes().Get(kl.nodeName)
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
			if err := kl.kubeClient.Core().Nodes().Delete(node.Name, nil); err != nil {
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

// relabelVolumes relabels SELinux volumes to match the pod's
// SELinuxOptions specification. This is only needed if the pod uses
// hostPID or hostIPC. Otherwise relabeling is delegated to docker.
func (kl *Kubelet) relabelVolumes(pod *api.Pod, volumes kubecontainer.VolumeMap) error {
	if pod.Spec.SecurityContext.SELinuxOptions == nil {
		return nil
	}

	rootDirContext, err := kl.getRootDirContext()
	if err != nil {
		return err
	}

	chconRunner := selinux.NewChconRunner()
	// Apply the pod's Level to the rootDirContext
	rootDirSELinuxOptions, err := securitycontext.ParseSELinuxOptions(rootDirContext)
	if err != nil {
		return err
	}

	rootDirSELinuxOptions.Level = pod.Spec.SecurityContext.SELinuxOptions.Level
	volumeContext := fmt.Sprintf("%s:%s:%s:%s", rootDirSELinuxOptions.User, rootDirSELinuxOptions.Role, rootDirSELinuxOptions.Type, rootDirSELinuxOptions.Level)

	for _, vol := range volumes {
		if vol.Builder.GetAttributes().Managed && vol.Builder.GetAttributes().SupportsSELinux {
			// Relabel the volume and its content to match the 'Level' of the pod
			err := filepath.Walk(vol.Builder.GetPath(), func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				return chconRunner.SetContext(path, volumeContext)
			})
			if err != nil {
				return err
			}
			vol.SELinuxLabeled = true
		}
	}
	return nil
}

func makeMounts(pod *api.Pod, podDir string, container *api.Container, podVolumes kubecontainer.VolumeMap) ([]kubecontainer.Mount, error) {
	// Kubernetes only mounts on /etc/hosts if :
	// - container does not use hostNetwork and
	// - container is not a infrastructure(pause) container
	// - container is not already mounting on /etc/hosts
	// When the pause container is being created, its IP is still unknown. Hence, PodIP will not have been set.
	mountEtcHostsFile := (pod.Spec.SecurityContext == nil || !pod.Spec.SecurityContext.HostNetwork) && len(pod.Status.PodIP) > 0
	glog.V(3).Infof("container: %v/%v/%v podIP: %q creating hosts mount: %v", pod.Namespace, pod.Name, container.Name, pod.Status.PodIP, mountEtcHostsFile)
	mounts := []kubecontainer.Mount{}
	for _, mount := range container.VolumeMounts {
		mountEtcHostsFile = mountEtcHostsFile && (mount.MountPath != etcHostsPath)
		vol, ok := podVolumes[mount.Name]
		if !ok {
			glog.Warningf("Mount cannot be satisified for container %q, because the volume is missing: %q", container.Name, mount)
			continue
		}

		relabelVolume := false
		// If the volume supports SELinux and it has not been
		// relabeled already and it is not a read-only volume,
		// relabel it and mark it as labeled
		if vol.Builder.GetAttributes().Managed && vol.Builder.GetAttributes().SupportsSELinux && !vol.SELinuxLabeled {
			vol.SELinuxLabeled = true
			relabelVolume = true
		}
		mounts = append(mounts, kubecontainer.Mount{
			Name:           mount.Name,
			ContainerPath:  mount.MountPath,
			HostPath:       vol.Builder.GetPath(),
			ReadOnly:       mount.ReadOnly,
			SELinuxRelabel: relabelVolume,
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
		return nil, fmt.Errorf("impossible: cannot find the mounted volumes for pod %q", format.Pod(pod))
	}

	opts.PortMappings = makePortMappings(container)
	// Docker does not relabel volumes if the container is running
	// in the host pid or ipc namespaces so the kubelet must
	// relabel the volumes
	if pod.Spec.SecurityContext != nil && (pod.Spec.SecurityContext.HostIPC || pod.Spec.SecurityContext.HostPID) {
		err = kl.relabelVolumes(pod, vol)
		if err != nil {
			return nil, err
		}
	}

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

	opts.DNS, opts.DNSSearch, err = kl.GetClusterDNS(pod)
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
	var (
		tmpEnv      = make(map[string]string)
		configMaps  = make(map[string]*api.ConfigMap)
		secrets     = make(map[string]*api.Secret)
		mappingFunc = expansion.MappingFuncFor(tmpEnv, serviceEnv)
	)
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
		} else if envVar.ValueFrom != nil {
			// Step 1b: resolve alternate env var sources
			switch {
			case envVar.ValueFrom.FieldRef != nil:
				runtimeVal, err = kl.podFieldSelectorRuntimeValue(envVar.ValueFrom.FieldRef, pod)
				if err != nil {
					return result, err
				}
			case envVar.ValueFrom.ConfigMapKeyRef != nil:
				name := envVar.ValueFrom.ConfigMapKeyRef.Name
				key := envVar.ValueFrom.ConfigMapKeyRef.Key
				configMap, ok := configMaps[name]
				if !ok {
					configMap, err = kl.kubeClient.Core().ConfigMaps(pod.Namespace).Get(name)
					if err != nil {
						return result, err
					}
				}
				runtimeVal, ok = configMap.Data[key]
				if !ok {
					return result, fmt.Errorf("Couldn't find key %v in ConfigMap %v/%v", key, pod.Namespace, name)
				}
			case envVar.ValueFrom.SecretKeyRef != nil:
				name := envVar.ValueFrom.SecretKeyRef.Name
				key := envVar.ValueFrom.SecretKeyRef.Key
				secret, ok := secrets[name]
				if !ok {
					secret, err = kl.kubeClient.Core().Secrets(pod.Namespace).Get(name)
					if err != nil {
						return result, err
					}
				}
				runtimeValBytes, ok := secret.Data[key]
				if !ok {
					return result, fmt.Errorf("Couldn't find key %v in Secret %v/%v", key, pod.Namespace, name)
				}
				runtimeVal = string(runtimeValBytes)
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

// GetClusterDNS returns a list of the DNS servers and a list of the DNS search
// domains of the cluster.
func (kl *Kubelet) GetClusterDNS(pod *api.Pod) ([]string, []string, error) {
	var hostDNS, hostSearch []string
	// Get host DNS settings
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
	useClusterFirstPolicy := pod.Spec.DNSPolicy == api.DNSClusterFirst
	if useClusterFirstPolicy && kl.clusterDNS == nil {
		// clusterDNS is not known.
		// pod with ClusterDNSFirst Policy cannot be created
		kl.recorder.Eventf(pod, api.EventTypeWarning, "MissingClusterDNS", "kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. Falling back to DNSDefault policy.", pod.Spec.DNSPolicy)
		log := fmt.Sprintf("kubelet does not have ClusterDNS IP configured and cannot create Pod using %q policy. pod: %q. Falling back to DNSDefault policy.", pod.Spec.DNSPolicy, format.Pod(pod))
		kl.recorder.Eventf(kl.nodeRef, api.EventTypeWarning, "MissingClusterDNS", log)

		// fallback to DNSDefault
		useClusterFirstPolicy = false
	}

	if !useClusterFirstPolicy {
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

	// for a pod with DNSClusterFirst policy, the cluster DNS server is the only nameserver configured for
	// the pod. The cluster DNS server itself will forward queries to other nameservers that is configured to use,
	// in case the cluster DNS server cannot resolve the DNS query itself
	dns := []string{kl.clusterDNS.String()}

	var dnsSearch []string
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

// One of the following aruguements must be non-nil: runningPod, status.
// TODO: Modify containerRuntime.KillPod() to accept the right arguements.
func (kl *Kubelet) killPod(pod *api.Pod, runningPod *kubecontainer.Pod, status *kubecontainer.PodStatus) error {
	var p kubecontainer.Pod
	if runningPod != nil {
		p = *runningPod
	} else if status != nil {
		p = kubecontainer.ConvertPodStatusToRunningPod(status)
	}
	return kl.containerRuntime.KillPod(pod, p)
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

func (kl *Kubelet) syncPod(pod *api.Pod, mirrorPod *api.Pod, podStatus *kubecontainer.PodStatus, updateType kubetypes.SyncPodType) error {
	var firstSeenTime time.Time
	if firstSeenTimeStr, ok := pod.Annotations[kubetypes.ConfigFirstSeenAnnotationKey]; ok {
		firstSeenTime = kubetypes.ConvertToTimestamp(firstSeenTimeStr).Get()
	}

	if updateType == kubetypes.SyncPodCreate {
		if !firstSeenTime.IsZero() {
			// This is the first time we are syncing the pod. Record the latency
			// since kubelet first saw the pod if firstSeenTime is set.
			metrics.PodWorkerStartLatency.Observe(metrics.SinceInMicroseconds(firstSeenTime))
		} else {
			glog.V(3).Infof("First seen time not recorded for pod %q", pod.UID)
		}
	}

	apiPodStatus := kl.generatePodStatus(pod, podStatus)
	// Record the time it takes for the pod to become running.
	existingStatus, ok := kl.statusManager.GetPodStatus(pod.UID)
	if !ok || existingStatus.Phase == api.PodPending && apiPodStatus.Phase == api.PodRunning &&
		!firstSeenTime.IsZero() {
		metrics.PodStartLatency.Observe(metrics.SinceInMicroseconds(firstSeenTime))
	}
	kl.statusManager.SetPodStatus(pod, apiPodStatus)

	// Kill pods we can't run.
	if err := canRunPod(pod); err != nil || pod.DeletionTimestamp != nil {
		if err := kl.killPod(pod, nil, podStatus); err != nil {
			utilruntime.HandleError(err)
		}
		return err
	}

	// Create Mirror Pod for Static Pod if it doesn't already exist
	if kubepod.IsStaticPod(pod) {
		podFullName := kubecontainer.GetPodFullName(pod)
		deleted := false
		if mirrorPod != nil && !kl.podManager.IsMirrorPodOf(mirrorPod, pod) {
			// The mirror pod is semantically different from the static pod. Remove
			// it. The mirror pod will get recreated later.
			glog.Errorf("Deleting mirror pod %q because it is outdated", format.Pod(mirrorPod))
			if err := kl.podManager.DeleteMirrorPod(podFullName); err != nil {
				glog.Errorf("Failed deleting mirror pod %q: %v", format.Pod(mirrorPod), err)
			} else {
				deleted = true
			}
		}
		if mirrorPod == nil || deleted {
			glog.V(3).Infof("Creating a mirror pod for static pod %q", format.Pod(pod))
			if err := kl.podManager.CreateMirrorPod(pod); err != nil {
				glog.Errorf("Failed creating a mirror pod for %q: %v", format.Pod(pod), err)
			}
		}
	}

	if err := kl.makePodDataDirs(pod); err != nil {
		glog.Errorf("Unable to make pod data directories for pod %q: %v", format.Pod(pod), err)
		return err
	}

	// Mount volumes.
	podVolumes, err := kl.mountExternalVolumes(pod)
	if err != nil {
		ref, errGetRef := api.GetReference(pod)
		if errGetRef == nil && ref != nil {
			kl.recorder.Eventf(ref, api.EventTypeWarning, kubecontainer.FailedMountVolume, "Unable to mount volumes for pod %q: %v", format.Pod(pod), err)
			glog.Errorf("Unable to mount volumes for pod %q: %v; skipping pod", format.Pod(pod), err)
			return err
		}
	}
	kl.volumeManager.SetVolumes(pod.UID, podVolumes)

	pullSecrets, err := kl.getPullSecretsForPod(pod)
	if err != nil {
		glog.Errorf("Unable to get pull secrets for pod %q: %v", format.Pod(pod), err)
		return err
	}

	result := kl.containerRuntime.SyncPod(pod, apiPodStatus, podStatus, pullSecrets, kl.backOff)
	kl.reasonCache.Update(pod.UID, result)
	if err = result.Error(); err != nil {
		return err
	}

	ingress, egress, err := extractBandwidthResources(pod)
	if err != nil {
		return err
	}
	if egress != nil || ingress != nil {
		if podUsesHostNetwork(pod) {
			kl.recorder.Event(pod, api.EventTypeWarning, kubecontainer.HostNetworkNotSupported, "Bandwidth shaping is not currently supported on the host network")
		} else if kl.shaper != nil {
			if len(apiPodStatus.PodIP) > 0 {
				err = kl.shaper.ReconcileCIDR(fmt.Sprintf("%s/32", apiPodStatus.PodIP), egress, ingress)
			}
		} else {
			kl.recorder.Event(pod, api.EventTypeWarning, kubecontainer.UndefinedShaper, "Pod requests bandwidth shaping, but the shaper is undefined")
		}
	}

	return nil
}

func podUsesHostNetwork(pod *api.Pod) bool {
	return pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.HostNetwork
}

// getPullSecretsForPod inspects the Pod and retrieves the referenced pull secrets
// TODO duplicate secrets are being retrieved multiple times and there is no cache.  Creating and using a secret manager interface will make this easier to address.
func (kl *Kubelet) getPullSecretsForPod(pod *api.Pod) ([]api.Secret, error) {
	pullSecrets := []api.Secret{}

	for _, secretRef := range pod.Spec.ImagePullSecrets {
		secret, err := kl.kubeClient.Core().Secrets(pod.Namespace).Get(secretRef.Name)
		if err != nil {
			glog.Warningf("Unable to retrieve pull secret %s/%s for %s/%s due to %v.  The image pull may not succeed.", pod.Namespace, secretRef.Name, pod.Namespace, pod.Name, err)
			continue
		}

		pullSecrets = append(pullSecrets, *secret)
	}

	return pullSecrets, nil
}

// Return name of a volume. When the volume is a PersistentVolumeClaim,
// it returns name of the real PersistentVolume bound to the claim.
// It returns errror when the clam is not bound yet.
func (kl *Kubelet) resolveVolumeName(pod *api.Pod, volume *api.Volume) (string, error) {
	claimSource := volume.VolumeSource.PersistentVolumeClaim
	if claimSource != nil {
		// resolve real volume behind the claim
		claim, err := kl.kubeClient.Core().PersistentVolumeClaims(pod.Namespace).Get(claimSource.ClaimName)
		if err != nil {
			return "", fmt.Errorf("Cannot find claim %s/%s for volume %s", pod.Namespace, claimSource.ClaimName, volume.Name)
		}
		if claim.Status.Phase != api.ClaimBound {
			return "", fmt.Errorf("Claim for volume %s/%s is not bound yet", pod.Namespace, claimSource.ClaimName)
		}
		// Use the real bound volume instead of PersistentVolume.Name
		return claim.Spec.VolumeName, nil
	}
	return volume.Name, nil
}

// Stores all volumes defined by the set of pods into a map.
// It stores real volumes there, i.e. persistent volume claims are resolved
// to volumes that are bound to them.
// Keys for each entry are in the format (POD_ID)/(VOLUME_NAME)
func (kl *Kubelet) getDesiredVolumes(pods []*api.Pod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Spec.Volumes {
			volumeName, err := kl.resolveVolumeName(pod, &volume)
			if err != nil {
				glog.V(3).Infof("%v", err)
				// Ignore the error and hope it's resolved next time
				continue
			}
			identifier := path.Join(string(pod.UID), volumeName)
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
	return utilerrors.NewAggregate(errlist)
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
			// TODO(random-liu): Cleanup status get functions. (issue #20477)
			s, err := kl.containerRuntime.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
			if err != nil {
				return err
			}
			status = kl.generatePodStatus(pod, s)
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
// This method is blocking:
// 1) it talks to API server to find volumes bound to persistent volume claims
// 2) it talks to cloud to detach volumes
func (kl *Kubelet) cleanupOrphanedVolumes(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	desiredVolumes := kl.getDesiredVolumes(pods)
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
				glog.V(5).Infof("Keeping terminated pod %q, still running", format.Pod(pod))
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
	if pod.Spec.ActiveDeadlineSeconds != nil {
		podStatus, ok := kl.statusManager.GetPodStatus(pod.UID)
		if !ok {
			podStatus = pod.Status
		}
		if !podStatus.StartTime.IsZero() {
			startTime := podStatus.StartTime.Time
			duration := kl.clock.Since(startTime)
			allowedDuration := time.Duration(*pod.Spec.ActiveDeadlineSeconds) * time.Second
			if duration >= allowedDuration {
				return true
			}
		}
	}
	return false
}

// Get pods which should be resynchronized. Currently, the following pod should be resynchronized:
//   * pod whose work is ready.
//   * pod past the active deadline.
func (kl *Kubelet) getPodsToSync() []*api.Pod {
	allPods := kl.podManager.GetPods()
	podUIDs := kl.workQueue.GetWork()
	podUIDSet := sets.NewString()
	for _, podUID := range podUIDs {
		podUIDSet.Insert(string(podUID))
	}
	var podsToSync []*api.Pod
	for _, pod := range allPods {
		if kl.pastActiveDeadline(pod) {
			// The pod has passed the active deadline
			podsToSync = append(podsToSync, pod)
			continue
		}
		if podUIDSet.Has(string(pod.UID)) {
			// The work of the pod is ready
			podsToSync = append(podsToSync, pod)
		}
	}
	return podsToSync
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

func (kl *Kubelet) deletePod(pod *api.Pod) error {
	if pod == nil {
		return fmt.Errorf("deletePod does not allow nil pod")
	}
	if !kl.allSourcesReady() {
		// If the sources aren't ready, skip deletion, as we may accidentally delete pods
		// for sources that haven't reported yet.
		return fmt.Errorf("skipping delete because sources aren't ready yet")
	}
	kl.podWorkers.ForgetWorker(pod.UID)

	// Runtime cache may not have been updated to with the pod, but it's okay
	// because the periodic cleanup routine will attempt to delete again later.
	runningPods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return fmt.Errorf("error listing containers: %v", err)
	}
	runningPod := kubecontainer.Pods(runningPods).FindPod("", pod.UID)
	if runningPod.IsEmpty() {
		return fmt.Errorf("pod not found")
	}
	podPair := kubecontainer.PodPair{pod, &runningPod}

	kl.podKillingCh <- &podPair
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
	kl.probeManager.CleanupPods(activePods)

	runningPods, err := kl.runtimeCache.GetPods()
	if err != nil {
		glog.Errorf("Error listing containers: %#v", err)
		return err
	}
	for _, pod := range runningPods {
		if _, found := desiredPods[pod.ID]; !found {
			kl.podKillingCh <- &kubecontainer.PodPair{nil, pod}
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
		case podPair, ok := <-kl.podKillingCh:
			runningPod := podPair.RunningPod
			apiPod := podPair.APIPod
			if !ok {
				return
			}
			if killing.Has(string(runningPod.ID)) {
				// The pod is already being killed.
				break
			}
			killing.Insert(string(runningPod.ID))
			go func(apiPod *api.Pod, runningPod *kubecontainer.Pod, ch chan types.UID) {
				defer func() {
					ch <- runningPod.ID
				}()
				glog.V(2).Infof("Killing unwanted pod %q", runningPod.Name)
				err := kl.killPod(apiPod, runningPod, nil)
				if err != nil {
					glog.Errorf("Failed killing the pod %q: %v", runningPod.Name, err)
				}
			}(apiPod, runningPod, resultCh)

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
		if errs := validation.AccumulateUniqueHostPorts(pod.Spec.Containers, &ports, field.NewPath("spec", "containers")); len(errs) > 0 {
			glog.Errorf("Pod %q: HostPort is already allocated, ignoring: %v", format.Pod(pod), errs)
			return true
		}
	}
	return false
}

// hasInsufficientfFreeResources detects pods that exceeds node's cpu and memory resource.
func (kl *Kubelet) hasInsufficientfFreeResources(pods []*api.Pod) (bool, bool) {
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		glog.Errorf("error getting machine info: %v", err)
		// TODO: Should we admit the pod when machine info is unavailable?
		return false, false
	}
	capacity := cadvisor.CapacityFromMachineInfo(info)
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
	kl.recorder.Eventf(pod, api.EventTypeWarning, reason, message)
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
//
// This needs to be kept in sync with the scheduler's and daemonset's fit predicates,
// otherwise there will inevitably be pod delete create loops. This will be fixed
// once we can extract these predicates into a common library. (#12744)
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
func (kl *Kubelet) syncLoop(updates <-chan kubetypes.PodUpdate, handler SyncHandler) {
	glog.Info("Starting kubelet main sync loop.")
	// The resyncTicker wakes up kubelet to checks if there are any pod workers
	// that need to be sync'd. A one-second period is sufficient because the
	// sync interval is defaulted to 10s.
	syncTicker := time.NewTicker(time.Second)
	housekeepingTicker := time.NewTicker(housekeepingPeriod)
	plegCh := kl.pleg.Watch()
	for {
		if rs := kl.runtimeState.errors(); len(rs) != 0 {
			glog.Infof("skipping pod synchronization - %v", rs)
			time.Sleep(5 * time.Second)
			continue
		}
		if !kl.syncLoopIteration(updates, handler, syncTicker.C, housekeepingTicker.C, plegCh) {
			break
		}
	}
}

func (kl *Kubelet) syncLoopIteration(updates <-chan kubetypes.PodUpdate, handler SyncHandler,
	syncCh <-chan time.Time, housekeepingCh <-chan time.Time, plegCh <-chan *pleg.PodLifecycleEvent) bool {
	kl.syncLoopMonitor.Store(kl.clock.Now())
	select {
	case u, open := <-updates:
		if !open {
			glog.Errorf("Update channel is closed. Exiting the sync loop.")
			return false
		}
		kl.addSource(u.Source)

		switch u.Op {
		case kubetypes.ADD:
			glog.V(2).Infof("SyncLoop (ADD, %q): %q", u.Source, format.Pods(u.Pods))
			handler.HandlePodAdditions(u.Pods)
		case kubetypes.UPDATE:
			glog.V(2).Infof("SyncLoop (UPDATE, %q): %q", u.Source, format.Pods(u.Pods))
			handler.HandlePodUpdates(u.Pods)
		case kubetypes.REMOVE:
			glog.V(2).Infof("SyncLoop (REMOVE, %q): %q", u.Source, format.Pods(u.Pods))
			handler.HandlePodDeletions(u.Pods)
		case kubetypes.RECONCILE:
			glog.V(4).Infof("SyncLoop (RECONCILE, %q): %q", u.Source, format.Pods(u.Pods))
			handler.HandlePodReconcile(u.Pods)
		case kubetypes.SET:
			// TODO: Do we want to support this?
			glog.Errorf("Kubelet does not support snapshot update")
		}
	case e := <-plegCh:
		pod, ok := kl.podManager.GetPodByUID(e.ID)
		if !ok {
			// If the pod no longer exists, ignore the event.
			glog.V(4).Infof("SyncLoop (PLEG): ignore irrelevant event: %#v", e)
			break
		}
		glog.V(2).Infof("SyncLoop (PLEG): %q, event: %#v", format.Pod(pod), e)
		// Force the container runtime cache to update.
		if err := kl.runtimeCache.ForceUpdateIfOlder(kl.clock.Now()); err != nil {
			glog.Errorf("SyncLoop: unable to update runtime cache")
			// TODO (yujuhong): should we delay the sync until container
			// runtime can be updated?
		}
		handler.HandlePodSyncs([]*api.Pod{pod})
	case <-syncCh:
		podsToSync := kl.getPodsToSync()
		if len(podsToSync) == 0 {
			break
		}
		glog.V(4).Infof("SyncLoop (SYNC): %d pods; %s", len(podsToSync), format.Pods(podsToSync))
		kl.HandlePodSyncs(podsToSync)
	case update := <-kl.livenessManager.Updates():
		// We only care about failures (signalling container death) here.
		if update.Result == proberesults.Failure {
			// We should not use the pod from livenessManager, because it is never updated after
			// initialization.
			// TODO(random-liu): This is just a quick fix. We should:
			//  * Just pass pod UID in probe updates to make this less confusing.
			//  * Maybe probe manager should rely on pod manager, or at least the pod in probe manager
			//  should be updated.
			pod, ok := kl.podManager.GetPodByUID(update.Pod.UID)
			if !ok {
				// If the pod no longer exists, ignore the update.
				glog.V(4).Infof("SyncLoop (container unhealthy): ignore irrelevant update: %#v", update)
				break
			}
			glog.V(1).Infof("SyncLoop (container unhealthy): %q", format.Pod(pod))
			handler.HandlePodSyncs([]*api.Pod{pod})
		}
	case <-housekeepingCh:
		if !kl.allSourcesReady() {
			// If the sources aren't ready, skip housekeeping, as we may
			// accidentally delete pods from unready sources.
			glog.V(4).Infof("SyncLoop (housekeeping, skipped): sources aren't ready yet.")
		} else {
			glog.V(4).Infof("SyncLoop (housekeeping)")
			if err := handler.HandlePodCleanups(); err != nil {
				glog.Errorf("Failed cleaning pods: %v", err)
			}
		}
	}
	kl.syncLoopMonitor.Store(kl.clock.Now())
	return true
}

func (kl *Kubelet) dispatchWork(pod *api.Pod, syncType kubetypes.SyncPodType, mirrorPod *api.Pod, start time.Time) {
	if kl.podIsTerminated(pod) {
		return
	}
	// Run the sync in an async worker.
	kl.podWorkers.UpdatePod(pod, mirrorPod, syncType, func() {
		metrics.PodWorkerLatency.WithLabelValues(syncType.String()).Observe(metrics.SinceInMicroseconds(start))
	})
	// Note the number of containers for new pods.
	if syncType == kubetypes.SyncPodCreate {
		metrics.ContainersPerPodCount.Observe(float64(len(pod.Spec.Containers)))
	}
}

// TODO: Consider handling all mirror pods updates in a separate component.
func (kl *Kubelet) handleMirrorPod(mirrorPod *api.Pod, start time.Time) {
	// Mirror pod ADD/UPDATE/DELETE operations are considered an UPDATE to the
	// corresponding static pod. Send update to the pod worker if the static
	// pod exists.
	if pod, ok := kl.podManager.GetPodByMirrorPod(mirrorPod); ok {
		kl.dispatchWork(pod, kubetypes.SyncPodUpdate, mirrorPod, start)
	}
}

func (kl *Kubelet) HandlePodAdditions(pods []*api.Pod) {
	start := kl.clock.Now()
	sort.Sort(podsByCreationTime(pods))
	for _, pod := range pods {
		kl.podManager.AddPod(pod)
		if kubepod.IsMirrorPod(pod) {
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
		kl.dispatchWork(pod, kubetypes.SyncPodCreate, mirrorPod, start)
		kl.probeManager.AddPod(pod)
	}
}

func (kl *Kubelet) HandlePodUpdates(pods []*api.Pod) {
	start := kl.clock.Now()
	for _, pod := range pods {
		kl.podManager.UpdatePod(pod)
		if kubepod.IsMirrorPod(pod) {
			kl.handleMirrorPod(pod, start)
			continue
		}
		// TODO: Evaluate if we need to validate and reject updates.

		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		kl.dispatchWork(pod, kubetypes.SyncPodUpdate, mirrorPod, start)
	}
}

func (kl *Kubelet) HandlePodDeletions(pods []*api.Pod) {
	start := kl.clock.Now()
	for _, pod := range pods {
		kl.podManager.DeletePod(pod)
		if kubepod.IsMirrorPod(pod) {
			kl.handleMirrorPod(pod, start)
			continue
		}
		// Deletion is allowed to fail because the periodic cleanup routine
		// will trigger deletion again.
		if err := kl.deletePod(pod); err != nil {
			glog.V(2).Infof("Failed to delete pod %q, err: %v", format.Pod(pod), err)
		}
		kl.probeManager.RemovePod(pod)
	}
}

func (kl *Kubelet) HandlePodReconcile(pods []*api.Pod) {
	for _, pod := range pods {
		// Update the pod in pod manager, status manager will do periodically reconcile according
		// to the pod manager.
		kl.podManager.UpdatePod(pod)
	}
}

func (kl *Kubelet) HandlePodSyncs(pods []*api.Pod) {
	start := kl.clock.Now()
	for _, pod := range pods {
		mirrorPod, _ := kl.podManager.GetMirrorPodByPod(pod)
		kl.dispatchWork(pod, kubetypes.SyncPodSync, mirrorPod, start)
	}
}

func (kl *Kubelet) LatestLoopEntryTime() time.Time {
	val := kl.syncLoopMonitor.Load()
	if val == nil {
		return time.Time{}
	}
	return val.(time.Time)
}

// validateContainerLogStatus returns the container ID for the desired container to retrieve logs for, based on the state
// of the container. The previous flag will only return the logs for the the last terminated container, otherwise, the current
// running container is preferred over a previous termination. If info about the container is not available then a specific
// error is returned to the end user.
func (kl *Kubelet) validateContainerLogStatus(podName string, podStatus *api.PodStatus, containerName string, previous bool) (containerID kubecontainer.ContainerID, err error) {
	var cID string

	cStatus, found := api.GetContainerStatus(podStatus.ContainerStatuses, containerName)
	if !found {
		return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is not available", containerName, podName)
	}
	lastState := cStatus.LastTerminationState
	waiting, running, terminated := cStatus.State.Waiting, cStatus.State.Running, cStatus.State.Terminated

	switch {
	case previous:
		if lastState.Terminated == nil {
			return kubecontainer.ContainerID{}, fmt.Errorf("previous terminated container %q in pod %q not found", containerName, podName)
		}
		cID = lastState.Terminated.ContainerID

	case running != nil:
		cID = cStatus.ContainerID

	case terminated != nil:
		cID = terminated.ContainerID

	case lastState.Terminated != nil:
		cID = lastState.Terminated.ContainerID

	case waiting != nil:
		// output some info for the most common pending failures
		switch reason := waiting.Reason; reason {
		case kubecontainer.ErrImagePull.Error():
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start: image can't be pulled", containerName, podName)
		case kubecontainer.ErrImagePullBackOff.Error():
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start: trying and failing to pull image", containerName, podName)
		default:
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start: %v", containerName, podName, reason)
		}
	default:
		// unrecognized state
		return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start - no logs yet", containerName, podName)
	}

	return kubecontainer.ParseContainerID(cID), nil
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
		return fmt.Errorf("pod %q cannot be found - no logs available", name)
	}

	podUID := pod.UID
	if mirrorPod, ok := kl.podManager.GetMirrorPodByPod(pod); ok {
		podUID = mirrorPod.UID
	}
	podStatus, found := kl.statusManager.GetPodStatus(podUID)
	if !found {
		// If there is no cached status, use the status from the
		// apiserver. This is useful if kubelet has recently been
		// restarted.
		podStatus = pod.Status
	}

	containerID, err := kl.validateContainerLogStatus(pod.Name, &podStatus, containerName, logOptions.Previous)
	if err != nil {
		return err
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
	if _, err := kl.containerRuntime.Version(); err != nil {
		glog.Errorf("Container runtime sanity check failed: %v", err)
		return
	}
	kl.oneTimeInitializer.Do(kl.initializeRuntimeDependentModules)
	kl.runtimeState.setRuntimeSync(kl.clock.Now())
}

// TODO: remove when kubenet plugin is ready
// NOTE!!! if you make changes here, also make them to kubenet
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

func (kl *Kubelet) recordNodeStatusEvent(eventtype, event string) {
	glog.V(2).Infof("Recording %s event message for node %s", event, kl.nodeName)
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, eventtype, event, "Node %s status is now: %s", kl.nodeName, event)
}

func (kl *Kubelet) syncNetworkStatus() {
	var err error
	if kl.configureCBR0 {
		if kl.flannelExperimentalOverlay {
			podCIDR, err := kl.flannelHelper.Handshake()
			if err != nil {
				glog.Infof("Flannel server handshake failed %v", err)
				return
			}
			kl.updatePodCIDR(podCIDR)
		}
		if err := ensureIPTablesMasqRule(kl.nonMasqueradeCIDR); err != nil {
			err = fmt.Errorf("Error on adding ip table rules: %v", err)
			glog.Error(err)
			kl.runtimeState.setNetworkState(err)
			return
		}
		podCIDR := kl.runtimeState.podCIDR()
		if len(podCIDR) == 0 {
			err = fmt.Errorf("ConfigureCBR0 requested, but PodCIDR not set. Will not configure CBR0 right now")
			glog.Warning(err)
		} else if err = kl.reconcileCBR0(podCIDR); err != nil {
			err = fmt.Errorf("Error configuring cbr0: %v", err)
			glog.Error(err)
		}
	}
	kl.runtimeState.setNetworkState(err)
}

// Set addresses for the node.
func (kl *Kubelet) setNodeAddress(node *api.Node) error {
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
		if kl.nodeIP != nil {
			node.Status.Addresses = []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: kl.nodeIP.String()},
				{Type: api.NodeInternalIP, Address: kl.nodeIP.String()},
			}
		} else if addr := net.ParseIP(kl.hostname); addr != nil {
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
					ip, err := utilnet.ChooseHostInterface()
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
	return nil
}

func (kl *Kubelet) setNodeStatusMachineInfo(node *api.Node) {
	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		// TODO(roberthbailey): This is required for test-cmd.sh to pass.
		// See if the test should be updated instead.
		node.Status.Capacity = api.ResourceList{
			api.ResourceCPU:    *resource.NewMilliQuantity(0, resource.DecimalSI),
			api.ResourceMemory: resource.MustParse("0Gi"),
			api.ResourcePods:   *resource.NewQuantity(int64(kl.maxPods), resource.DecimalSI),
		}
		glog.Errorf("Error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Status.Capacity = cadvisor.CapacityFromMachineInfo(info)
		node.Status.Capacity[api.ResourcePods] = *resource.NewQuantity(
			int64(kl.maxPods), resource.DecimalSI)
		if node.Status.NodeInfo.BootID != "" &&
			node.Status.NodeInfo.BootID != info.BootID {
			// TODO: This requires a transaction, either both node status is updated
			// and event is recorded or neither should happen, see issue #6055.
			kl.recorder.Eventf(kl.nodeRef, api.EventTypeWarning, kubecontainer.NodeRebooted,
				"Node %s has been rebooted, boot id: %s", kl.nodeName, info.BootID)
		}
		node.Status.NodeInfo.BootID = info.BootID
	}

	// Set Allocatable.
	node.Status.Allocatable = make(api.ResourceList)
	for k, v := range node.Status.Capacity {
		value := *(v.Copy())
		if kl.reservation.System != nil {
			value.Sub(kl.reservation.System[k])
		}
		if kl.reservation.Kubernetes != nil {
			value.Sub(kl.reservation.Kubernetes[k])
		}
		if value.Amount != nil && value.Amount.Sign() < 0 {
			// Negative Allocatable resources don't make sense.
			value.Set(0)
		}
		node.Status.Allocatable[k] = value
	}
}

// Set versioninfo for the node.
func (kl *Kubelet) setNodeStatusVersionInfo(node *api.Node) {
	verinfo, err := kl.cadvisor.VersionInfo()
	if err != nil {
		glog.Errorf("Error getting version info: %v", err)
	} else {
		node.Status.NodeInfo.KernelVersion = verinfo.KernelVersion
		node.Status.NodeInfo.OSImage = verinfo.ContainerOsVersion

		runtimeVersion := "Unknown"
		if runtimeVer, err := kl.containerRuntime.Version(); err == nil {
			runtimeVersion = runtimeVer.String()
		}
		node.Status.NodeInfo.ContainerRuntimeVersion = fmt.Sprintf("%s://%s", kl.containerRuntime.Type(), runtimeVersion)

		node.Status.NodeInfo.KubeletVersion = version.Get().String()
		// TODO: kube-proxy might be different version from kubelet in the future
		node.Status.NodeInfo.KubeProxyVersion = version.Get().String()
	}

}

// Set daemonEndpoints for the node.
func (kl *Kubelet) setNodeStatusDaemonEndpoints(node *api.Node) {
	node.Status.DaemonEndpoints = *kl.daemonEndpoints
}

// Set images list fot this node
func (kl *Kubelet) setNodeStatusImages(node *api.Node) {
	// Update image list of this node
	var imagesOnNode []api.ContainerImage
	containerImages, err := kl.imageManager.GetImageList()
	if err != nil {
		glog.Errorf("Error getting image list: %v", err)
	} else {
		for _, image := range containerImages {
			imagesOnNode = append(imagesOnNode, api.ContainerImage{
				RepoTags: image.RepoTags,
				Size:     image.Size,
			})
		}
	}
	node.Status.Images = imagesOnNode
}

// Set status for the node.
func (kl *Kubelet) setNodeStatusInfo(node *api.Node) {
	kl.setNodeStatusMachineInfo(node)
	kl.setNodeStatusVersionInfo(node)
	kl.setNodeStatusDaemonEndpoints(node)
	kl.setNodeStatusImages(node)
}

// Set Readycondition for the node.
func (kl *Kubelet) setNodeReadyCondition(node *api.Node) {
	// NOTE(aaronlevy): NodeReady condition needs to be the last in the list of node conditions.
	// This is due to an issue with version skewed kubelet and master components.
	// ref: https://github.com/kubernetes/kubernetes/issues/16961
	currentTime := unversioned.NewTime(kl.clock.Now())
	var newNodeReadyCondition api.NodeCondition
	if rs := kl.runtimeState.errors(); len(rs) == 0 {
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionTrue,
			Reason:            "KubeletReady",
			Message:           "kubelet is posting ready status",
			LastHeartbeatTime: currentTime,
		}
	} else {
		newNodeReadyCondition = api.NodeCondition{
			Type:              api.NodeReady,
			Status:            api.ConditionFalse,
			Reason:            "KubeletNotReady",
			Message:           strings.Join(rs, ","),
			LastHeartbeatTime: currentTime,
		}
	}

	readyConditionUpdated := false
	needToRecordEvent := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			if node.Status.Conditions[i].Status == newNodeReadyCondition.Status {
				newNodeReadyCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			} else {
				newNodeReadyCondition.LastTransitionTime = currentTime
				needToRecordEvent = true
			}
			node.Status.Conditions[i] = newNodeReadyCondition
			readyConditionUpdated = true
			break
		}
	}
	if !readyConditionUpdated {
		newNodeReadyCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newNodeReadyCondition)
	}
	if needToRecordEvent {
		if newNodeReadyCondition.Status == api.ConditionTrue {
			kl.recordNodeStatusEvent(api.EventTypeNormal, kubecontainer.NodeReady)
		} else {
			kl.recordNodeStatusEvent(api.EventTypeNormal, kubecontainer.NodeNotReady)
		}
	}
}

// Set OODcondition for the node.
func (kl *Kubelet) setNodeOODCondition(node *api.Node) {
	currentTime := unversioned.NewTime(kl.clock.Now())
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
			Type:   api.NodeOutOfDisk,
			Status: api.ConditionUnknown,
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
			kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeOutOfDisk")
		}
	} else {
		if nodeOODCondition.Status != api.ConditionFalse {
			// Update the out of disk condition when the condition status is unknown even if we
			// are within the outOfDiskTransitionFrequency duration. We do this to set the
			// condition status correctly at kubelet startup.
			if nodeOODCondition.Status == api.ConditionUnknown || kl.clock.Since(nodeOODCondition.LastTransitionTime.Time) >= kl.outOfDiskTransitionFrequency {
				nodeOODCondition.Status = api.ConditionFalse
				nodeOODCondition.Reason = "KubeletHasSufficientDisk"
				nodeOODCondition.Message = "kubelet has sufficient disk space available"
				nodeOODCondition.LastTransitionTime = currentTime
				kl.recordNodeStatusEvent(api.EventTypeNormal, "NodeHasSufficientDisk")
			} else {
				glog.Infof("Node condition status for OutOfDisk is false, but last transition time is less than %s", kl.outOfDiskTransitionFrequency)
			}
		}
	}

	if newOODCondition {
		node.Status.Conditions = append(node.Status.Conditions, *nodeOODCondition)
	}
}

// Maintains Node.Spec.Unschedulable value from previous run of tryUpdateNodeStatus()
var oldNodeUnschedulable bool

// record if node schedulable change.
func (kl *Kubelet) recordNodeSchdulableEvent(node *api.Node) {
	if oldNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			kl.recordNodeStatusEvent(api.EventTypeNormal, kubecontainer.NodeNotSchedulable)
		} else {
			kl.recordNodeStatusEvent(api.EventTypeNormal, kubecontainer.NodeSchedulable)
		}
		oldNodeUnschedulable = node.Spec.Unschedulable
	}
}

// setNodeStatus fills in the Status fields of the given Node, overwriting
// any fields that are currently set.
// TODO(madhusudancs): Simplify the logic for setting node conditions and
// refactor the node status condtion code out to a different file.
func (kl *Kubelet) setNodeStatus(node *api.Node) error {
	if err := kl.setNodeAddress(node); err != nil {
		return err
	}
	kl.setNodeStatusInfo(node)
	kl.setNodeOODCondition(node)
	kl.setNodeReadyCondition(node)
	kl.recordNodeSchdulableEvent(node)
	return nil
}

// FIXME: Why not combine this with container runtime health check?
func (kl *Kubelet) isContainerRuntimeVersionCompatible() error {
	switch kl.GetRuntime().Type() {
	case "docker":
		version, err := kl.GetRuntime().APIVersion()
		if err != nil {
			return nil
		}
		// Verify the docker version.
		result, err := version.Compare(dockertools.MinimumDockerAPIVersion)
		if err != nil {
			return fmt.Errorf("failed to compare current docker version %v with minimum support Docker version %q - %v", version, dockertools.MinimumDockerAPIVersion, err)
		}
		if result < 0 {
			return fmt.Errorf("container runtime version is older than %s", dockertools.MinimumDockerAPIVersion)
		}
	}
	return nil
}

// tryUpdateNodeStatus tries to update node status to master. If ReconcileCBR0
// is set, this function will also confirm that cbr0 is configured correctly.
func (kl *Kubelet) tryUpdateNodeStatus() error {
	node, err := kl.kubeClient.Core().Nodes().Get(kl.nodeName)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", kl.nodeName, err)
	}
	if node == nil {
		return fmt.Errorf("no node instance returned for %q", kl.nodeName)
	}
	// Flannel is the authoritative source of pod CIDR, if it's running.
	// This is a short term compromise till we get flannel working in
	// reservation mode.
	if kl.flannelExperimentalOverlay {
		flannelPodCIDR := kl.runtimeState.podCIDR()
		if node.Spec.PodCIDR != flannelPodCIDR {
			node.Spec.PodCIDR = flannelPodCIDR
			glog.Infof("Updating podcidr to %v", node.Spec.PodCIDR)
			if updatedNode, err := kl.kubeClient.Core().Nodes().Update(node); err != nil {
				glog.Warningf("Failed to update podCIDR: %v", err)
			} else {
				// Update the node resourceVersion so the status update doesn't fail.
				node = updatedNode
			}
		}
	} else if kl.reconcileCIDR {
		kl.updatePodCIDR(node.Spec.PodCIDR)
	}

	if err := kl.setNodeStatus(node); err != nil {
		return err
	}
	// Update the current status on the API server
	_, err = kl.kubeClient.Core().Nodes().UpdateStatus(node)
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
		containerStatus, ok := api.GetContainerStatus(info, container.Name)
		if !ok {
			unknown++
			continue
		}

		switch {
		case containerStatus.State.Running != nil:
			running++
		case containerStatus.State.Terminated != nil:
			stopped++
			if containerStatus.State.Terminated.ExitCode == 0 {
				succeeded++
			} else {
				failed++
			}
		case containerStatus.State.Waiting != nil:
			if containerStatus.LastTerminationState.Terminated != nil {
				stopped++
			} else {
				waiting++
			}
		default:
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

func (kl *Kubelet) generatePodStatus(pod *api.Pod, podStatus *kubecontainer.PodStatus) api.PodStatus {
	glog.V(3).Infof("Generating status for %q", format.Pod(pod))
	// TODO: Consider include the container information.
	if kl.pastActiveDeadline(pod) {
		reason := "DeadlineExceeded"
		kl.recorder.Eventf(pod, api.EventTypeNormal, reason, "Pod was active on the node longer than specified deadline")
		return api.PodStatus{
			Phase:   api.PodFailed,
			Reason:  reason,
			Message: "Pod was active on the node longer than specified deadline"}
	}

	s := kl.convertStatusToAPIStatus(pod, podStatus)

	// Assume info is ready to process
	spec := &pod.Spec
	s.Phase = GetPhase(spec, s.ContainerStatuses)
	kl.probeManager.UpdatePodStatus(pod.UID, s)
	s.Conditions = append(s.Conditions, status.GeneratePodReadyCondition(spec, s.ContainerStatuses, s.Phase))

	if !kl.standaloneMode {
		hostIP, err := kl.GetHostIP()
		if err != nil {
			glog.V(4).Infof("Cannot get host IP: %v", err)
		} else {
			s.HostIP = hostIP.String()
			if podUsesHostNetwork(pod) && s.PodIP == "" {
				s.PodIP = hostIP.String()
			}
		}
	}

	return *s
}

// TODO(random-liu): Move this to some better place.
// TODO(random-liu): Add test for convertStatusToAPIStatus()
func (kl *Kubelet) convertStatusToAPIStatus(pod *api.Pod, podStatus *kubecontainer.PodStatus) *api.PodStatus {
	var apiPodStatus api.PodStatus
	uid := pod.UID

	convertContainerStatus := func(cs *kubecontainer.ContainerStatus) *api.ContainerStatus {
		cid := cs.ID.String()
		status := &api.ContainerStatus{
			Name:         cs.Name,
			RestartCount: cs.RestartCount,
			Image:        cs.Image,
			ImageID:      cs.ImageID,
			ContainerID:  cid,
		}
		switch cs.State {
		case kubecontainer.ContainerStateRunning:
			status.State.Running = &api.ContainerStateRunning{StartedAt: unversioned.NewTime(cs.StartedAt)}
		case kubecontainer.ContainerStateExited:
			status.State.Terminated = &api.ContainerStateTerminated{
				ExitCode:    cs.ExitCode,
				Reason:      cs.Reason,
				Message:     cs.Message,
				StartedAt:   unversioned.NewTime(cs.StartedAt),
				FinishedAt:  unversioned.NewTime(cs.FinishedAt),
				ContainerID: cid,
			}
		default:
			status.State.Waiting = &api.ContainerStateWaiting{}
		}
		return status
	}

	statuses := make(map[string]*api.ContainerStatus, len(pod.Spec.Containers))
	// Create a map of expected containers based on the pod spec.
	expectedContainers := make(map[string]api.Container)
	for _, container := range pod.Spec.Containers {
		expectedContainers[container.Name] = container
	}

	containerDone := sets.NewString()
	apiPodStatus.PodIP = podStatus.IP
	for _, containerStatus := range podStatus.ContainerStatuses {
		cName := containerStatus.Name
		if _, ok := expectedContainers[cName]; !ok {
			// This would also ignore the infra container.
			continue
		}
		if containerDone.Has(cName) {
			continue
		}
		status := convertContainerStatus(containerStatus)
		if existing, found := statuses[cName]; found {
			existing.LastTerminationState = status.State
			containerDone.Insert(cName)
		} else {
			statuses[cName] = status
		}
	}

	// Handle the containers for which we cannot find any associated active or dead containers or are in restart backoff
	// Fetch old containers statuses from old pod status.
	// TODO(random-liu) Maybe it's better to get status from status manager, because it takes the newest status and there is not
	// status in api.Pod of static pod.
	oldStatuses := make(map[string]api.ContainerStatus, len(pod.Spec.Containers))
	for _, status := range pod.Status.ContainerStatuses {
		oldStatuses[status.Name] = status
	}
	for _, container := range pod.Spec.Containers {
		// TODO(random-liu): We should define "Waiting" state better. And cleanup the following code.
		if containerStatus, found := statuses[container.Name]; found {
			reason, message, ok := kl.reasonCache.Get(uid, container.Name)
			if ok && reason == kubecontainer.ErrCrashLoopBackOff {
				containerStatus.LastTerminationState = containerStatus.State
				containerStatus.State = api.ContainerState{
					Waiting: &api.ContainerStateWaiting{
						Reason:  reason.Error(),
						Message: message,
					},
				}
			}
			continue
		}
		var containerStatus api.ContainerStatus
		containerStatus.Name = container.Name
		containerStatus.Image = container.Image
		if oldStatus, found := oldStatuses[container.Name]; found {
			// Some states may be lost due to GC; apply the last observed
			// values if possible.
			containerStatus.RestartCount = oldStatus.RestartCount
			containerStatus.LastTerminationState = oldStatus.LastTerminationState
		}
		reason, _, ok := kl.reasonCache.Get(uid, container.Name)

		if !ok {
			// default position for a container
			// At this point there are no active or dead containers, the reasonCache is empty (no entry or the entry has expired)
			// its reasonable to say the container is being created till a more accurate reason is logged
			containerStatus.State = api.ContainerState{
				Waiting: &api.ContainerStateWaiting{
					Reason:  fmt.Sprintf("ContainerCreating"),
					Message: fmt.Sprintf("Image: %s is ready, container is creating", container.Image),
				},
			}
		} else if reason == kubecontainer.ErrImagePullBackOff ||
			reason == kubecontainer.ErrImageInspect ||
			reason == kubecontainer.ErrImagePull ||
			reason == kubecontainer.ErrImageNeverPull {
			// mark it as waiting, reason will be filled bellow
			containerStatus.State = api.ContainerState{Waiting: &api.ContainerStateWaiting{}}
		} else if reason == kubecontainer.ErrRunContainer {
			// mark it as waiting, reason will be filled bellow
			containerStatus.State = api.ContainerState{Waiting: &api.ContainerStateWaiting{}}
		}
		statuses[container.Name] = &containerStatus
	}

	apiPodStatus.ContainerStatuses = make([]api.ContainerStatus, 0)
	for containerName, status := range statuses {
		if status.State.Waiting != nil {
			status.State.Running = nil
			// For containers in the waiting state, fill in a specific reason if it is recorded.
			if reason, message, ok := kl.reasonCache.Get(uid, containerName); ok {
				status.State.Waiting.Reason = reason.Error()
				status.State.Waiting.Message = message
			}
		}
		apiPodStatus.ContainerStatuses = append(apiPodStatus.ContainerStatuses, *status)
	}

	// Sort the container statuses since clients of this interface expect the list
	// of containers in a pod has a deterministic order.
	sort.Sort(kubetypes.SortedContainerStatuses(apiPodStatus.ContainerStatuses))
	return &apiPodStatus
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
	return kl.runner.RunInContainer(container.ID, cmd)
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
	return kl.runner.ExecInContainer(container.ID, cmd, stdin, stdout, stderr, tty)
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
	return kl.containerRuntime.AttachContainer(container.ID, stdin, stdout, stderr, tty)
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
	kl.recorder.Eventf(kl.nodeRef, api.EventTypeNormal, kubecontainer.StartingKubelet, "Starting kubelet.")
}

func (kl *Kubelet) StreamingConnectionIdleTimeout() time.Duration {
	return kl.streamingConnectionIdleTimeout
}

func (kl *Kubelet) ResyncInterval() time.Duration {
	return kl.resyncInterval
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName string, podUID types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {

	podUID = kl.podManager.TranslatePodUID(podUID)

	pods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return nil, err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	container := pod.FindContainerByName(containerName)
	if container == nil {
		return nil, kubecontainer.ErrContainerNotFound
	}

	ci, err := kl.cadvisor.DockerContainer(container.ID.ID, req)
	if err != nil {
		return nil, err
	}
	return &ci, nil
}

// GetContainerInfoV2 returns stats (from Cadvisor) for containers.
func (kl *Kubelet) GetContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	return kl.cadvisor.ContainerInfoV2(name, options)
}

func (kl *Kubelet) DockerImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	return kl.cadvisor.DockerImagesFsInfo()
}

func (kl *Kubelet) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return kl.cadvisor.RootFsInfo()
}

// Returns stats (from Cadvisor) for a non-Kubernetes container.
func (kl *Kubelet) GetRawContainerInfo(containerName string, req *cadvisorapi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapi.ContainerInfo, error) {
	if subcontainers {
		return kl.cadvisor.SubcontainerInfo(containerName, req)
	} else {
		containerInfo, err := kl.cadvisor.ContainerInfo(containerName, req)
		if err != nil {
			return nil, err
		}
		return map[string]*cadvisorapi.ContainerInfo{
			containerInfo.Name: containerInfo,
		}, nil
	}
}

// GetCachedMachineInfo assumes that the machine info can't change without a reboot
func (kl *Kubelet) GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error) {
	if kl.machineInfo == nil {
		info, err := kl.cadvisor.MachineInfo()
		if err != nil {
			return nil, err
		}
		kl.machineInfo = info
	}
	return kl.machineInfo, nil
}

func (kl *Kubelet) ListenAndServe(address net.IP, port uint, tlsOptions *server.TLSOptions, auth server.AuthInterface, enableDebuggingHandlers bool) {
	server.ListenAndServeKubeletServer(kl, kl.resourceAnalyzer, address, port, tlsOptions, auth, enableDebuggingHandlers)
}

func (kl *Kubelet) ListenAndServeReadOnly(address net.IP, port uint) {
	server.ListenAndServeKubeletReadOnlyServer(kl, kl.resourceAnalyzer, address, port)
}

// GetRuntime returns the current Runtime implementation in use by the kubelet. This func
// is exported to simplify integration with third party kubelet extensions (e.g. kubernetes-mesos).
func (kl *Kubelet) GetRuntime() kubecontainer.Runtime {
	return kl.containerRuntime
}

func (kl *Kubelet) updatePodCIDR(cidr string) {
	if kl.runtimeState.podCIDR() == cidr {
		return
	}

	glog.Infof("Setting Pod CIDR: %v -> %v", kl.runtimeState.podCIDR(), cidr)
	kl.runtimeState.setPodCIDR(cidr)

	if kl.networkPlugin != nil {
		details := make(map[string]interface{})
		details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = cidr
		kl.networkPlugin.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)
	}
}
func (kl *Kubelet) GetNodeConfig() cm.NodeConfig {
	return kl.nodeConfig
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
