/*
Copyright 2015 Google Inc. All rights reserved.

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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fieldpath"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/envvars"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/lifecycle"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/prober"
	kubeletTypes "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilErrors "github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

const (
	// Max amount of time to wait for the Docker daemon to come up.
	maxWaitForDocker = 5 * time.Minute

	// Initial node status update frequency and incremental frequency, for faster cluster startup.
	// The update frequency will be increameted linearly, until it reaches status_update_frequency.
	initialNodeStatusUpdateFrequency = 100 * time.Millisecond
	nodeStatusUpdateFrequencyInc     = 500 * time.Millisecond

	// nodeStatusUpdateRetry specifies how many times kubelet retries when posting node status failed.
	nodeStatusUpdateRetry = 5

	// Location of container logs.
	containerLogsDir = "/var/log/containers"
)

var (
	// ErrNoKubeletContainers returned when there are not containers managed by
	// the kubelet (ie: either no containers on the node, or none that the kubelet cares about).
	ErrNoKubeletContainers = errors.New("no containers managed by kubelet")

	// ErrContainerNotFound returned when a container in the given pod with the
	// given container name was not found, amongst those managed by the kubelet.
	ErrContainerNotFound = errors.New("no matching container")
)

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {

	// Syncs current state to match the specified pods. SyncPodType specified what
	// type of sync is occuring per pod. StartTime specifies the time at which
	// syncing began (for use in monitoring).
	SyncPods(pods []*api.Pod, podSyncTypes map[types.UID]metrics.SyncPodType, mirrorPods map[string]*api.Pod,
		startTime time.Time) error
}

type SourcesReadyFn func() bool

type volumeMap map[string]volume.Volume

// New creates a new Kubelet for use in main
func NewMainKubelet(
	hostname string,
	dockerClient dockertools.DockerInterface,
	kubeClient client.Interface,
	rootDirectory string,
	podInfraContainerImage string,
	resyncInterval time.Duration,
	pullQPS float32,
	pullBurst int,
	containerGCPolicy ContainerGCPolicy,
	sourcesReady SourcesReadyFn,
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
	cloud cloudprovider.Interface,
	nodeStatusUpdateFrequency time.Duration,
	resourceContainer string,
	osInterface kubecontainer.OSInterface) (*Kubelet, error) {
	if rootDirectory == "" {
		return nil, fmt.Errorf("invalid root directory %q", rootDirectory)
	}
	if resyncInterval <= 0 {
		return nil, fmt.Errorf("invalid sync frequency %d", resyncInterval)
	}
	dockerClient = dockertools.NewInstrumentedDockerInterface(dockerClient)

	// Wait for the Docker daemon to be up (with a timeout).
	waitStart := time.Now()
	dockerUp := false
	for time.Since(waitStart) < maxWaitForDocker {
		_, err := dockerClient.Version()
		if err == nil {
			dockerUp = true
			break
		}

		time.Sleep(100 * time.Millisecond)
	}
	if !dockerUp {
		return nil, fmt.Errorf("timed out waiting for Docker to come up")
	}

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
	serviceLister := &cache.StoreToServiceLister{serviceStore}

	nodeStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	if kubeClient != nil {
		// TODO: cache.NewListWatchFromClient is limited as it takes a client implementation rather
		// than an interface. There is no way to construct a list+watcher using resource name.
		fieldSelector := fields.Set{client.ObjectNameField: hostname}.AsSelector()
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
	nodeLister := &cache.StoreToNodeLister{nodeStore}

	// TODO: get the real minion object of ourself,
	// and use the real minion name and UID.
	// TODO: what is namespace for node?
	nodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      hostname,
		UID:       types.UID(hostname),
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
	statusManager := newStatusManager(kubeClient)
	readinessManager := kubecontainer.NewReadinessManager()
	containerRefManager := kubecontainer.NewRefManager()

	volumeManager := newVolumeManager()

	oomWatcher := NewOOMWatcher(cadvisorInterface, recorder)

	klet := &Kubelet{
		hostname:                       hostname,
		dockerClient:                   dockerClient,
		kubeClient:                     kubeClient,
		rootDirectory:                  rootDirectory,
		resyncInterval:                 resyncInterval,
		containerRefManager:            containerRefManager,
		readinessManager:               readinessManager,
		httpClient:                     &http.Client{},
		sourcesReady:                   sourcesReady,
		clusterDomain:                  clusterDomain,
		clusterDNS:                     clusterDNS,
		serviceLister:                  serviceLister,
		nodeLister:                     nodeLister,
		masterServiceNamespace:         masterServiceNamespace,
		streamingConnectionIdleTimeout: streamingConnectionIdleTimeout,
		recorder:                       recorder,
		cadvisor:                       cadvisorInterface,
		containerGC:                    containerGC,
		imageManager:                   imageManager,
		statusManager:                  statusManager,
		volumeManager:                  volumeManager,
		cloud:                          cloud,
		nodeRef:                        nodeRef,
		nodeStatusUpdateFrequency:      nodeStatusUpdateFrequency,
		resourceContainer:              resourceContainer,
		os:                             osInterface,
		oomWatcher:                     oomWatcher,
		runtimeHooks:                   newKubeletRuntimeHooks(recorder),
	}

	if plug, err := network.InitNetworkPlugin(networkPlugins, networkPluginName, &networkHost{klet}); err != nil {
		return nil, err
	} else {
		klet.networkPlugin = plug
	}
	containerManager := dockertools.NewDockerManager(
		dockerClient,
		recorder,
		readinessManager,
		containerRefManager,
		podInfraContainerImage,
		pullQPS,
		pullBurst,
		containerLogsDir,
		osInterface,
		klet.networkPlugin,
		nil)
	klet.runner = containerManager
	klet.containerManager = containerManager

	klet.podManager = newBasicPodManager(klet.kubeClient)
	klet.prober = prober.New(klet.runner, klet.readinessManager, klet.containerRefManager, klet.recorder)
	klet.handlerRunner = lifecycle.NewHandlerRunner(klet.httpClient, klet.runner, klet.containerManager)

	// TODO(vmarmol): Remove when the circular dependency is removed :(
	containerManager.Prober = klet.prober

	runtimeCache, err := kubecontainer.NewRuntimeCache(containerManager)
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
	dockerClient   dockertools.DockerInterface
	runtimeCache   kubecontainer.RuntimeCache
	kubeClient     client.Interface
	rootDirectory  string
	podWorkers     *podWorkers
	resyncInterval time.Duration
	sourcesReady   SourcesReadyFn

	podManager podManager

	// Needed to report events for containers belonging to deleted/modified pods.
	// Tracks references for reporting events
	containerRefManager *kubecontainer.RefManager

	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
	// Optional, defaults to simple Docker implementation
	runner prober.ContainerCommandRunner
	// Optional, client for http requests, defaults to empty client
	httpClient kubeletTypes.HttpGetter

	// cAdvisor used for container information.
	cadvisor cadvisor.Interface

	// If non-empty, use this for container DNS search.
	clusterDomain string

	// If non-nil, use this for container DNS server.
	clusterDNS net.IP

	masterServiceNamespace string
	serviceLister          serviceLister
	nodeLister             nodeLister

	// Volume plugins.
	volumePluginMgr volume.VolumePluginMgr

	// Network plugin.
	networkPlugin network.NetworkPlugin

	// Healthy check prober.
	prober prober.Prober

	// Container lifecycle handler runner.
	handlerRunner kubecontainer.HandlerRunner

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

	// Cached MachineInfo returned by cadvisor.
	machineInfo *cadvisorApi.MachineInfo

	// Syncs pods statuses with apiserver; also used as a cache of statuses.
	statusManager *statusManager

	// Manager for the volume maps for the pods.
	volumeManager *volumeManager

	//Cloud provider interface
	cloud cloudprovider.Interface

	// Reference to this node.
	nodeRef *api.ObjectReference

	// Manage containers.
	containerManager *dockertools.DockerManager

	// nodeStatusUpdateFrequency specifies how often kubelet posts node status to master.
	// Note: be cautious when changing the constant, it must work with nodeMonitorGracePeriod
	// in nodecontroller. There are several constraints:
	// 1. nodeMonitorGracePeriod must be N times more than nodeStatusUpdateFrequency, where
	//    N means number of retries allowed for kubelet to post node status. It is pointless
	//    to make nodeMonitorGracePeriod be less than nodeStatusUpdateFrequency, since there
	//    will only be fresh values from Kubelet at an interval of nodeStatusUpdateFrequency.
	//    The constant must be less than podEvictionTimeout.
	// 2. nodeStatusUpdateFrequency needs to be large enough for kubelet to generate node
	//    status. Kubelet may fail to update node status reliablly if the value is too small,
	//    as it takes time to gather all necessary node information.
	nodeStatusUpdateFrequency time.Duration
	// The name of the resource-only container to run the Kubelet in (empty for no container).
	// Name must be absolute.
	resourceContainer string

	os kubecontainer.OSInterface

	// Watcher of out of memory events.
	oomWatcher OOMWatcher

	// TODO(vmarmol): Remove this when we only have to inject the hooks into the runtimes.
	// Hooks injected into the container runtime.
	runtimeHooks kubecontainer.RuntimeHooks
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
	l, err := kl.nodeLister.List()
	if err != nil {
		return nil, errors.New("cannot list nodes")
	}
	host := kl.GetHostname()
	for _, n := range l.Items {
		if n.Name == host {
			return &n, nil
		}
	}
	return nil, fmt.Errorf("node %v not found", host)
}

// Starts garbage collection theads.
func (kl *Kubelet) StartGarbageCollection() {
	go util.Forever(func() {
		if err := kl.containerGC.GarbageCollect(); err != nil {
			glog.Errorf("Container garbage collection failed: %v", err)
		}
	}, time.Minute)

	go util.Forever(func() {
		if err := kl.imageManager.GarbageCollect(); err != nil {
			glog.Errorf("Image garbage collection failed: %v", err)
		}
	}, 5*time.Minute)
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
		err := util.RunInResourceContainer(kl.resourceContainer)
		if err != nil {
			glog.Warningf("Failed to move Kubelet to container %q: %v", kl.resourceContainer, err)
		}
		glog.Infof("Running in container %q", kl.resourceContainer)
	}

	go kl.syncNodeStatus()
	// Run the system oom watcher forever.
	go util.Until(kl.runOOMWatcher, time.Second, util.NeverStop)
	kl.statusManager.Start()
	kl.syncLoop(updates, kl)
}

// Watches for system OOMs.
func (kl *Kubelet) runOOMWatcher() {
	glog.V(5).Infof("Starting to record system OOMs")
	if err := kl.oomWatcher.RecordSysOOMs(kl.nodeRef); err != nil {
		glog.Errorf("failed to record system OOMs - %v", err)
	}
}

// syncNodeStatus periodically synchronizes node status to master.
func (kl *Kubelet) syncNodeStatus() {
	if kl.kubeClient == nil {
		return
	}
	glog.Infof("Starting node status updates")
	for feq := initialNodeStatusUpdateFrequency; feq < kl.nodeStatusUpdateFrequency; feq += nodeStatusUpdateFrequencyInc {
		select {
		case <-time.After(feq):
			if err := kl.updateNodeStatus(); err != nil {
				glog.Errorf("Unable to update node status: %v", err)
			}
		}
	}
	for {
		select {
		case <-time.After(kl.nodeStatusUpdateFrequency):
			if err := kl.updateNodeStatus(); err != nil {
				glog.Errorf("Unable to update node status: %v", err)
			}
		}
	}
}

func makeBinds(container *api.Container, podVolumes volumeMap) (binds []string) {
	for _, mount := range container.VolumeMounts {
		vol, ok := podVolumes[mount.Name]
		if !ok {
			glog.Warningf("Mount cannot be satisified for container %q, because the volume is missing: %q", container.Name, mount)
			continue
		}
		b := fmt.Sprintf("%s:%s", vol.GetPath(), mount.MountPath)
		if mount.ReadOnly {
			b += ":ro"
		}
		binds = append(binds, b)
	}
	return
}

// GenerateRunContainerOptions generates the RunContainerOptions, which can be used by
// the container runtime to set parameters for launching a container.
func (kl *Kubelet) GenerateRunContainerOptions(pod *api.Pod, container *api.Container, netMode, ipcMode string) (*kubecontainer.RunContainerOptions, error) {
	var err error
	opts := &kubecontainer.RunContainerOptions{
		NetMode: netMode,
		IpcMode: ipcMode,
	}

	vol, ok := kl.volumeManager.GetVolumes(pod.UID)
	if !ok {
		return nil, fmt.Errorf("impossible: cannot find the mounted volumes for pod %q", kubecontainer.GetPodFullName(pod))
	}
	opts.Binds = makeBinds(container, vol)
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
	if pod.Spec.DNSPolicy == api.DNSClusterFirst {
		opts.DNS, opts.DNSSearch, err = kl.getClusterDNS(pod)
		if err != nil {
			return nil, err
		}
	}
	return opts, nil
}

var masterServices = util.NewStringSet("kubernetes", "kubernetes-ro")

// getServiceEnvVarMap makes a map[string]string of env vars for services a pod in namespace ns should see
func (kl *Kubelet) getServiceEnvVarMap(ns string) (map[string]string, error) {
	var (
		serviceMap = make(map[string]api.Service)
		m          = make(map[string]string)
	)

	// Get all service resources from the master (via a cache),
	// and populate them into service enviroment variables.
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
		// ignore services where PortalIP is "None" or empty
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
				_, exists := serviceMap[serviceName]
				if !exists {
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
func (kl *Kubelet) makeEnvironmentVariables(pod *api.Pod, container *api.Container) ([]string, error) {
	var result []string
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

	for _, value := range container.Env {
		// Accesses apiserver+Pods.
		// So, the master may set service env vars, or kubelet may.  In case both are doing
		// it, we delete the key from the kubelet-generated ones so we don't have duplicate
		// env vars.
		// TODO: remove this net line once all platforms use apiserver+Pods.
		delete(serviceEnv, value.Name)

		runtimeValue, err := kl.runtimeEnvVarValue(value, pod)
		if err != nil {
			return result, err
		}

		result = append(result, fmt.Sprintf("%s=%s", value.Name, runtimeValue))
	}

	// Append remaining service env vars.
	for k, v := range serviceEnv {
		result = append(result, fmt.Sprintf("%s=%s", k, v))
	}
	return result, nil
}

// runtimeEnvVarValue determines the value that an env var should take when a container
// is started.  If the value of the env var is the empty string, the source of the env var
// is resolved, if one is specified.
//
// TODO: preliminary factoring; make better
func (kl *Kubelet) runtimeEnvVarValue(envVar api.EnvVar, pod *api.Pod) (string, error) {
	runtimeVal := envVar.Value
	if runtimeVal != "" {
		return runtimeVal, nil
	}

	if envVar.ValueFrom != nil && envVar.ValueFrom.FieldPath != nil {
		return kl.podFieldSelectorRuntimeValue(envVar.ValueFrom.FieldPath, pod)
	}

	return runtimeVal, nil
}

func (kl *Kubelet) podFieldSelectorRuntimeValue(fs *api.ObjectFieldSelector, pod *api.Pod) (string, error) {
	internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(fs.APIVersion, "Pod", fs.FieldPath, "")
	if err != nil {
		return "", err
	}

	return fieldpath.ExtractFieldPathAsString(pod, internalFieldPath)
}

// getClusterDNS returns a list of the DNS servers and a list of the DNS search
// domains of the cluster.
func (kl *Kubelet) getClusterDNS(pod *api.Pod) ([]string, []string, error) {
	// Get host DNS settings and append them to cluster DNS settings.
	f, err := os.Open("/etc/resolv.conf")
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	hostDNS, hostSearch, err := parseResolvConf(f)
	if err != nil {
		return nil, nil, err
	}

	var dns, dnsSearch []string

	if kl.clusterDNS != nil {
		dns = append([]string{kl.clusterDNS.String()}, hostDNS...)
	}
	if kl.clusterDomain != "" {
		nsDomain := fmt.Sprintf("%s.%s", pod.Namespace, kl.clusterDomain)
		dnsSearch = append([]string{nsDomain, kl.clusterDomain}, hostSearch...)
	}
	return dns, dnsSearch, nil
}

// Returns the list of DNS servers and DNS search domains.
func parseResolvConf(reader io.Reader) (nameservers []string, searches []string, err error) {
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
	return nameservers, searches, nil
}

// Pull the image for the specified pod and container.
func (kl *Kubelet) pullImage(pod *api.Pod, container *api.Container) error {
	present, err := kl.containerManager.IsImagePresent(container.Image)
	if err != nil {
		ref, err := kubecontainer.GenerateContainerRef(pod, container)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
		}
		if ref != nil {
			kl.recorder.Eventf(ref, "failed", "Failed to inspect image %q: %v", container.Image, err)
		}
		return fmt.Errorf("failed to inspect image %q: %v", container.Image, err)
	}

	if !kl.runtimeHooks.ShouldPullImage(pod, container, present) {
		return nil
	}

	err = kl.containerManager.Pull(container.Image)
	kl.runtimeHooks.ReportImagePull(pod, container, err)
	return err
}

// Kill all running containers in a pod (includes the pod infra container).
func (kl *Kubelet) killPod(pod kubecontainer.Pod) error {
	return kl.containerManager.KillPod(pod)
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

func (kl *Kubelet) syncPod(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod) error {
	podFullName := kubecontainer.GetPodFullName(pod)
	uid := pod.UID

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
			kl.statusManager.SetPodStatus(podToUpdate, status)
		}
	}()

	// Kill pods we can't run.
	err := canRunPod(pod)
	if err != nil {
		kl.killPod(runningPod)
		return err
	}

	if err := kl.makePodDataDirs(pod); err != nil {
		glog.Errorf("Unable to make pod data directories for pod %q (uid %q): %v", podFullName, uid, err)
		return err
	}

	podStatus, err := kl.generatePodStatus(pod)
	if err != nil {
		glog.Errorf("Unable to get status for pod %q (uid %q): %v", podFullName, uid, err)
		return err
	}

	containerChanges, err := kl.containerManager.ComputePodContainerChanges(pod, runningPod, podStatus)
	glog.V(3).Infof("Got container changes for pod %q: %+v", podFullName, containerChanges)
	if err != nil {
		return err
	}

	if containerChanges.StartInfraContainer || (len(containerChanges.ContainersToKeep) == 0 && len(containerChanges.ContainersToStart) == 0) {
		if len(containerChanges.ContainersToKeep) == 0 && len(containerChanges.ContainersToStart) == 0 {
			glog.V(4).Infof("Killing Infra Container for %q because all other containers are dead.", podFullName)
		} else {
			glog.V(4).Infof("Killing Infra Container for %q, will start new one", podFullName)
		}

		// Killing phase: if we want to start new infra container, or nothing is running kill everything (including infra container)
		err = kl.killPod(runningPod)
		if err != nil {
			return err
		}
	} else {
		// Otherwise kill any containers in this pod which are not specified as ones to keep.
		for _, container := range runningPod.Containers {
			_, keep := containerChanges.ContainersToKeep[kubeletTypes.DockerID(container.ID)]
			if !keep {
				glog.V(3).Infof("Killing unwanted container %+v", container)
				err = kl.containerManager.KillContainer(container.ID)
				if err != nil {
					glog.Errorf("Error killing container: %v", err)
				}
			}
		}
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
			kl.recorder.Eventf(ref, "failedMount", "Unable to mount volumes for pod %q: %v", podFullName, err)
		}
		glog.Errorf("Unable to mount volumes for pod %q: %v; skipping pod", podFullName, err)
		return err
	}
	kl.volumeManager.SetVolumes(pod.UID, podVolumes)

	// If we should create infra container then we do it first.
	podInfraContainerID := containerChanges.InfraContainerId
	if containerChanges.StartInfraContainer && (len(containerChanges.ContainersToStart) > 0) {
		glog.V(4).Infof("Creating pod infra container for %q", podFullName)
		podInfraContainerID, err = kl.containerManager.CreatePodInfraContainer(pod, kl, kl.handlerRunner)

		// Call the networking plugin
		if err == nil {
			err = kl.networkPlugin.SetUpPod(pod.Namespace, pod.Name, podInfraContainerID)
		}
		if err != nil {
			glog.Errorf("Failed to create pod infra container: %v; Skipping pod %q", err, podFullName)
			return err
		}
	}

	// Start everything
	for container := range containerChanges.ContainersToStart {
		glog.V(4).Infof("Creating container %+v", pod.Spec.Containers[container])
		containerSpec := &pod.Spec.Containers[container]
		if err := kl.pullImage(pod, containerSpec); err != nil {
			glog.Warningf("Failed to pull image %q from pod %q and container %q: %v", containerSpec.Image, kubecontainer.GetPodFullName(pod), containerSpec.Name, err)
			continue
		}
		// TODO(dawnchen): Check RestartPolicy.DelaySeconds before restart a container
		namespaceMode := fmt.Sprintf("container:%v", podInfraContainerID)
		_, err := kl.containerManager.RunContainer(pod, containerSpec, kl, kl.handlerRunner, namespaceMode, namespaceMode)
		if err != nil {
			// TODO(bburns) : Perhaps blacklist a container after N failures?
			glog.Errorf("Error running pod %q container %q: %v", kubecontainer.GetPodFullName(pod), containerSpec.Name, err)
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
			// Pod status update is edge-triggered. If there is any update of the
			// mirror pod, we need to delete the existing status associated with
			// the static pod to trigger an update.
			kl.statusManager.DeletePodStatus(podFullName)
		}
	}
	return nil
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

func (kl *Kubelet) cleanupOrphanedPodDirs(pods []*api.Pod) error {
	desired := util.NewStringSet()
	for _, pod := range pods {
		desired.Insert(string(pod.UID))
	}
	found, err := kl.listPodsFromDisk()
	if err != nil {
		return err
	}
	errlist := []error{}
	for i := range found {
		if !desired.Has(string(found[i])) {
			glog.V(3).Infof("Orphaned pod %q found, removing", found[i])
			if err := os.RemoveAll(kl.getPodDir(found[i])); err != nil {
				errlist = append(errlist, err)
			}
		}
	}
	return utilErrors.NewAggregate(errlist)
}

// Compares the map of current volumes to the map of desired volumes.
// If an active volume does not have a respective desired volume, clean it up.
func (kl *Kubelet) cleanupOrphanedVolumes(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	desiredVolumes := getDesiredVolumes(pods)
	currentVolumes := kl.getPodVolumesFromDisk()

	runningSet := util.StringSet{}
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

// Filter out pods in the terminated state ("Failed" or "Succeeded").
func (kl *Kubelet) filterOutTerminatedPods(allPods []*api.Pod) []*api.Pod {
	var pods []*api.Pod
	for _, pod := range allPods {
		var status api.PodStatus
		// Check the cached pod status which was set after the last sync.
		status, ok := kl.statusManager.GetPodStatus(kubecontainer.GetPodFullName(pod))
		if !ok {
			// If there is no cached status, use the status from the
			// apiserver. This is useful if kubelet has recently been
			// restarted.
			status = pod.Status
		}
		if status.Phase == api.PodFailed || status.Phase == api.PodSucceeded {
			// Pod has reached the final state; ignore it.
			continue
		}
		pods = append(pods, pod)
	}
	return pods
}

// SyncPods synchronizes the configured list of pods (desired state) with the host current state.
func (kl *Kubelet) SyncPods(allPods []*api.Pod, podSyncTypes map[types.UID]metrics.SyncPodType,
	mirrorPods map[string]*api.Pod, start time.Time) error {
	defer func() {
		metrics.SyncPodsLatency.Observe(metrics.SinceInMicroseconds(start))
	}()

	// Remove obsolete entries in podStatus where the pod is no longer considered bound to this node.
	podFullNames := make(map[string]bool)
	for _, pod := range allPods {
		podFullNames[kubecontainer.GetPodFullName(pod)] = true
	}
	kl.statusManager.RemoveOrphanedStatuses(podFullNames)

	// Reject pods that we cannot run.
	kl.handleNotFittingPods(allPods)

	// Pod phase progresses monotonically. Once a pod has reached a final state,
	// it should never leave irregardless of the restart policy. The statuses
	// of such pods should not be changed, and there is no need to sync them.
	// TODO: the logic here does not handle two cases:
	//   1. If the containers were removed immediately after they died, kubelet
	//      may fail to generate correct statuses, let alone filtering correctly.
	//   2. If kubelet restarted before writing the terminated status for a pod
	//      to the apiserver, it could still restart the terminated pod (even
	//      though the pod was not considered terminated by the apiserver).
	// These two conditions could be alleviated by checkpointing kubelet.
	pods := kl.filterOutTerminatedPods(allPods)

	glog.V(4).Infof("Desired: %#v", pods)
	var err error
	desiredPods := make(map[types.UID]empty)

	runningPods, err := kl.runtimeCache.GetPods()
	if err != nil {
		glog.Errorf("Error listing containers: %#v", err)
		return err
	}

	// Check for any containers that need starting
	for _, pod := range pods {
		podFullName := kubecontainer.GetPodFullName(pod)
		uid := pod.UID
		desiredPods[uid] = empty{}

		// Run the sync in an async manifest worker.
		kl.podWorkers.UpdatePod(pod, mirrorPods[podFullName], func() {
			metrics.SyncPodLatency.WithLabelValues(podSyncTypes[pod.UID].String()).Observe(metrics.SinceInMicroseconds(start))
		})

		// Note the number of containers for new pods.
		if val, ok := podSyncTypes[pod.UID]; ok && (val == metrics.SyncPodCreate) {
			metrics.ContainersPerPodCount.Observe(float64(len(pod.Spec.Containers)))
		}
	}
	// Stop the workers for no-longer existing pods.
	kl.podWorkers.ForgetNonExistingPodWorkers(desiredPods)

	if !kl.sourcesReady() {
		// If the sources aren't ready, skip deletion, as we may accidentally delete pods
		// for sources that haven't reported yet.
		glog.V(4).Infof("Skipping deletes, sources aren't ready yet.")
		return nil
	}

	// Kill containers associated with unwanted pods.
	err = kl.killUnwantedPods(desiredPods, runningPods)
	if err != nil {
		glog.Errorf("Failed killing unwanted containers: %v", err)
	}

	// Note that we just killed the unwanted pods. This may not have reflected
	// in the cache. We need to bypass the cach to get the latest set of
	// running pods to clean up the volumes.
	// TODO: Evaluate the performance impact of bypassing the runtime cache.
	runningPods, err = kl.containerManager.GetPods(false)
	if err != nil {
		glog.Errorf("Error listing containers: %#v", err)
		return err
	}

	// Remove any orphaned volumes.
	err = kl.cleanupOrphanedVolumes(pods, runningPods)
	if err != nil {
		glog.Errorf("Failed cleaning up orphaned volumes: %v", err)
		return err
	}

	// Remove any orphaned pod directories.
	err = kl.cleanupOrphanedPodDirs(pods)
	if err != nil {
		glog.Errorf("Failed cleaning up orphaned pod directories: %v", err)
		return err
	}

	// Remove any orphaned mirror pods.
	kl.podManager.DeleteOrphanedMirrorPods()

	return err
}

// killUnwantedPods kills the unwanted, running pods in parallel.
func (kl *Kubelet) killUnwantedPods(desiredPods map[types.UID]empty,
	runningPods []*kubecontainer.Pod) error {
	ch := make(chan error, len(runningPods))
	defer close(ch)
	numWorkers := 0
	for _, pod := range runningPods {
		if _, found := desiredPods[pod.ID]; found {
			// Per-pod workers will handle the desired pods.
			continue
		}
		numWorkers++
		go func(pod *kubecontainer.Pod, ch chan error) {
			var err error = nil
			defer func() {
				ch <- err
			}()
			glog.V(1).Infof("Killing unwanted pod %q", pod.Name)
			// Stop the containers.
			err = kl.killPod(*pod)
			if err != nil {
				glog.Errorf("Failed killing the pod %q: %v", pod.Name, err)
				return
			}
		}(pod, ch)
	}

	// Aggregate errors from the pod killing workers.
	var errs []error
	for i := 0; i < numWorkers; i++ {
		err := <-ch
		if err != nil {
			errs = append(errs, err)
		}
	}
	return utilErrors.NewAggregate(errs)
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
func checkHostPortConflicts(pods []*api.Pod) (fitting []*api.Pod, notFitting []*api.Pod) {
	ports := map[int]bool{}
	extract := func(p *api.ContainerPort) int { return p.HostPort }

	// Respect the pod creation order when resolving conflicts.
	sort.Sort(podsByCreationTime(pods))

	for _, pod := range pods {
		if errs := validation.AccumulateUniquePorts(pod.Spec.Containers, ports, extract); len(errs) != 0 {
			glog.Errorf("Pod %q: HostPort is already allocated, ignoring: %v", kubecontainer.GetPodFullName(pod), errs)
			notFitting = append(notFitting, pod)
			continue
		}
		fitting = append(fitting, pod)
	}
	return
}

// checkCapacityExceeded detects pods that exceeds node's resources.
func (kl *Kubelet) checkCapacityExceeded(pods []*api.Pod) (fitting []*api.Pod, notFitting []*api.Pod) {
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		glog.Errorf("error getting machine info: %v", err)
		return pods, nil
	}

	// Respect the pod creation order when resolving conflicts.
	sort.Sort(podsByCreationTime(pods))

	capacity := CapacityFromMachineInfo(info)
	return scheduler.CheckPodsExceedingCapacity(pods, capacity)
}

// checkNodeSelectorMatching detects pods that do not match node's labels.
func (kl *Kubelet) checkNodeSelectorMatching(pods []*api.Pod) (fitting []*api.Pod, notFitting []*api.Pod) {
	node, err := kl.GetNode()
	if err != nil {
		glog.Errorf("error getting node: %v", err)
		return pods, nil
	}
	for _, pod := range pods {
		if !scheduler.PodMatchesNodeLabels(pod, node) {
			notFitting = append(notFitting, pod)
			continue
		}
		fitting = append(fitting, pod)
	}
	return
}

// handleNotfittingPods handles pods that do not fit on the node.
// Currently conflicts on Port.HostPort values, matching node's labels and exceeding node's capacity are handled.
func (kl *Kubelet) handleNotFittingPods(pods []*api.Pod) {
	fitting, notFitting := checkHostPortConflicts(pods)
	for _, pod := range notFitting {
		kl.recorder.Eventf(pod, "hostPortConflict", "Cannot start the pod due to host port conflict.")
		kl.statusManager.SetPodStatus(pod, api.PodStatus{
			Phase:   api.PodFailed,
			Message: "Pod cannot be started due to host port conflict"})
	}
	fitting, notFitting = kl.checkNodeSelectorMatching(fitting)
	for _, pod := range notFitting {
		kl.recorder.Eventf(pod, "nodeSelectorMismatching", "Cannot start the pod due to node selector mismatch.")
		kl.statusManager.SetPodStatus(pod, api.PodStatus{
			Phase:   api.PodFailed,
			Message: "Pod cannot be started due to node selector mismatch"})
	}
	fitting, notFitting = kl.checkCapacityExceeded(fitting)
	for _, pod := range notFitting {
		kl.recorder.Eventf(pod, "capacityExceeded", "Cannot start the pod due to exceeded capacity.")
		kl.statusManager.SetPodStatus(pod, api.PodStatus{
			Phase:   api.PodFailed,
			Message: "Pod cannot be started due to exceeded capacity"})
	}
}

// syncLoop is the main loop for processing changes. It watches for changes from
// three channels (file, apiserver, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync_frequency seconds. Never returns.
func (kl *Kubelet) syncLoop(updates <-chan PodUpdate, handler SyncHandler) {
	glog.Info("Starting kubelet main sync loop.")
	for {
		unsyncedPod := false
		podSyncTypes := make(map[types.UID]metrics.SyncPodType)
		select {
		case u, ok := <-updates:
			if !ok {
				glog.Errorf("Update channel is closed. Exiting the sync loop.")
				return
			}
			kl.podManager.UpdatePods(u, podSyncTypes)
			unsyncedPod = true
		case <-time.After(kl.resyncInterval):
			glog.V(4).Infof("Periodic sync")
		}
		start := time.Now()
		// If we already caught some update, try to wait for some short time
		// to possibly batch it with other incoming updates.
		for unsyncedPod {
			select {
			case u := <-updates:
				kl.podManager.UpdatePods(u, podSyncTypes)
			case <-time.After(5 * time.Millisecond):
				// Break the for loop.
				unsyncedPod = false
			}
		}
		pods, mirrorPods := kl.podManager.GetPodsAndMirrorMap()
		if err := handler.SyncPods(pods, podSyncTypes, mirrorPods, start); err != nil {
			glog.Errorf("Couldn't sync containers: %v", err)
		}
	}
}

// Returns the container runtime version for this Kubelet.
func (kl *Kubelet) GetContainerRuntimeVersion() (kubecontainer.Version, error) {
	if kl.containerManager == nil {
		return nil, fmt.Errorf("no container runtime")
	}
	return kl.containerManager.Version()
}

func (kl *Kubelet) validatePodPhase(podStatus *api.PodStatus) error {
	switch podStatus.Phase {
	case api.PodRunning, api.PodSucceeded, api.PodFailed:
		return nil
	}
	return fmt.Errorf("pod is not in 'Running', 'Succeeded' or 'Failed' state - State: %q", podStatus.Phase)
}

func (kl *Kubelet) validateContainerStatus(podStatus *api.PodStatus, containerName string) (dockerID string, err error) {
	cStatus, found := api.GetContainerStatus(podStatus.ContainerStatuses, containerName)
	if !found {
		return "", fmt.Errorf("container %q not found in pod", containerName)
	}
	if cStatus.State.Waiting != nil {
		return "", fmt.Errorf("container %q is in waiting state.", containerName)
	}
	return kubecontainer.TrimRuntimePrefixFromImage(cStatus.ContainerID), nil
}

// GetKubeletContainerLogs returns logs from the container
// TODO: this method is returning logs of random container attempts, when it should be returning the most recent attempt
// or all of them.
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	podStatus, err := kl.GetPodStatus(podFullName)
	if err != nil {
		return fmt.Errorf("failed to get status for pod %q - %v", podFullName, err)
	}
	if err := kl.validatePodPhase(&podStatus); err != nil {
		// No log is available if pod is not in a "known" phase (e.g. Unknown).
		return err
	}
	dockerContainerID, err := kl.validateContainerStatus(&podStatus, containerName)
	if err != nil {
		// No log is available if the container status is missing or is in the
		// waiting state.
		return err
	}
	return kl.containerManager.GetContainerLogs(dockerContainerID, tail, follow, stdout, stderr)
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
	addresses := node.Status.Addresses
	addressMap := make(map[api.NodeAddressType][]api.NodeAddress)
	for i := range addresses {
		addressMap[addresses[i].Type] = append(addressMap[addresses[i].Type], addresses[i])
	}
	if addresses, ok := addressMap[api.NodeLegacyHostIP]; ok {
		return net.ParseIP(addresses[0].Address), nil
	}
	if addresses, ok := addressMap[api.NodeInternalIP]; ok {
		return net.ParseIP(addresses[0].Address), nil
	}
	if addresses, ok := addressMap[api.NodeExternalIP]; ok {
		return net.ParseIP(addresses[0].Address), nil
	}
	return nil, fmt.Errorf("host IP unknown; known addresses: %v", addresses)
}

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pods.
func (kl *Kubelet) GetPods() []*api.Pod {
	return kl.podManager.GetPods()
}

func (kl *Kubelet) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	return kl.podManager.GetPodByFullName(podFullName)
}

// GetPodByName provides the first pod that matches namespace and name, as well
// as whether the pod was found.
func (kl *Kubelet) GetPodByName(namespace, name string) (*api.Pod, bool) {
	return kl.podManager.GetPodByName(namespace, name)
}

// updateNodeStatus updates node status to master with retries.
func (kl *Kubelet) updateNodeStatus() error {
	for i := 0; i < nodeStatusUpdateRetry; i++ {
		err := kl.tryUpdateNodeStatus()
		if err != nil {
			glog.Errorf("error updating node status, will retry: %v", err)
		} else {
			return nil
		}
	}
	return fmt.Errorf("Update node status exceeds retry count")
}

func (kl *Kubelet) recordNodeOnlineEvent() {
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, "online", "Node %s is now online", kl.hostname)
}

func (kl *Kubelet) recordNodeSchedulableEvent() {
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, "schedulable", "Node %s is now schedulable", kl.hostname)
}

func (kl *Kubelet) recordNodeUnschedulableEvent() {
	// TODO: This requires a transaction, either both node status is updated
	// and event is recorded or neither should happen, see issue #6055.
	kl.recorder.Eventf(kl.nodeRef, "unschedulable", "Node %s is now unschedulable", kl.hostname)
}

// Maintains Node.Spec.Unschedulable value from previous run of tryUpdateNodeStatus()
var oldNodeUnschedulable bool

// tryUpdateNodeStatus tries to update node status to master.
func (kl *Kubelet) tryUpdateNodeStatus() error {
	node, err := kl.kubeClient.Nodes().Get(kl.hostname)
	if err != nil {
		return fmt.Errorf("error getting node %q: %v", kl.hostname, err)
	}
	if node == nil {
		return fmt.Errorf("no node instance returned for %q", kl.hostname)
	}

	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		glog.Errorf("error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Status.Capacity = CapacityFromMachineInfo(info)
		if node.Status.NodeInfo.BootID != "" &&
			node.Status.NodeInfo.BootID != info.BootID {
			// TODO: This requires a transaction, either both node status is updated
			// and event is recorded or neither should happen, see issue #6055.
			kl.recorder.Eventf(kl.nodeRef, "rebooted",
				"Node %s has been rebooted, boot id: %s", kl.hostname, info.BootID)
		}
		node.Status.NodeInfo.BootID = info.BootID
	}

	verinfo, err := kl.cadvisor.VersionInfo()
	if err != nil {
		glog.Errorf("error getting version info: %v", err)
	} else {
		node.Status.NodeInfo.KernelVersion = verinfo.KernelVersion
		node.Status.NodeInfo.OsImage = verinfo.ContainerOsVersion
		// TODO: Determine the runtime is docker or rocket
		node.Status.NodeInfo.ContainerRuntimeVersion = "docker://" + verinfo.DockerVersion
		node.Status.NodeInfo.KubeletVersion = version.Get().String()
		// TODO: kube-proxy might be different version from kubelet in the future
		node.Status.NodeInfo.KubeProxyVersion = version.Get().String()
	}

	currentTime := util.Now()
	newCondition := api.NodeCondition{
		Type:              api.NodeReady,
		Status:            api.ConditionTrue,
		Reason:            fmt.Sprintf("kubelet is posting ready status"),
		LastHeartbeatTime: currentTime,
	}
	updated := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			newCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			if node.Status.Conditions[i].Status != api.ConditionTrue {
				kl.recordNodeOnlineEvent()
			}
			node.Status.Conditions[i] = newCondition
			updated = true
		}
	}
	if !updated {
		newCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newCondition)
		kl.recordNodeOnlineEvent()
	}

	if oldNodeUnschedulable != node.Spec.Unschedulable {
		if node.Spec.Unschedulable {
			kl.recordNodeUnschedulableEvent()
		} else {
			kl.recordNodeSchedulableEvent()
		}
		oldNodeUnschedulable = node.Spec.Unschedulable
	}

	// Update the current status on the API server
	_, err = kl.kubeClient.Nodes().UpdateStatus(node)
	return err
}

// getPhase returns the phase of a pod given its container info.
func getPhase(spec *api.PodSpec, info []api.ContainerStatus) api.PodPhase {
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
			} else if containerStatus.State.Termination != nil {
				stopped++
				if containerStatus.State.Termination.ExitCode == 0 {
					succeeded++
				} else {
					failed++
				}
			} else if containerStatus.State.Waiting != nil {
				waiting++
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

// getPodReadyCondition returns ready condition if all containers in a pod are ready, else it returns an unready condition.
func getPodReadyCondition(spec *api.PodSpec, statuses []api.ContainerStatus) []api.PodCondition {
	ready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionTrue,
	}}
	unready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionFalse,
	}}
	if statuses == nil {
		return unready
	}
	for _, container := range spec.Containers {
		if containerStatus, ok := api.GetContainerStatus(statuses, container.Name); ok {
			if !containerStatus.Ready {
				return unready
			}
		} else {
			return unready
		}
	}
	return ready
}

// GetPodStatus returns information from Docker about the containers in a pod
func (kl *Kubelet) GetPodStatus(podFullName string) (api.PodStatus, error) {
	// Check to see if we have a cached version of the status.
	cachedPodStatus, found := kl.statusManager.GetPodStatus(podFullName)
	if found {
		glog.V(3).Infof("Returning cached status for %q", podFullName)
		return cachedPodStatus, nil
	}
	pod, found := kl.GetPodByFullName(podFullName)
	if !found {
		return api.PodStatus{}, fmt.Errorf("couldn't find pod %q", podFullName)
	}
	return kl.generatePodStatus(pod)
}

// By passing the pod directly, this method avoids pod lookup, which requires
// grabbing a lock.
func (kl *Kubelet) generatePodStatus(pod *api.Pod) (api.PodStatus, error) {
	podFullName := kubecontainer.GetPodFullName(pod)
	glog.V(3).Infof("Generating status for %q", podFullName)

	spec := &pod.Spec
	podStatus, err := kl.containerManager.GetPodStatus(pod)

	if err != nil {
		// Error handling
		glog.Infof("Query docker container info for pod %q failed with error (%v)", podFullName, err)
		if strings.Contains(err.Error(), "resource temporarily unavailable") {
			// Leave upstream layer to decide what to do
			return api.PodStatus{}, err
		} else {
			pendingStatus := api.PodStatus{
				Phase:   api.PodPending,
				Message: fmt.Sprintf("Query docker container info failed with error (%v)", err),
			}
			return pendingStatus, nil
		}
	}

	// Assume info is ready to process
	podStatus.Phase = getPhase(spec, podStatus.ContainerStatuses)
	for _, c := range spec.Containers {
		for i, st := range podStatus.ContainerStatuses {
			if st.Name == c.Name {
				ready := st.State.Running != nil && kl.readinessManager.GetReadiness(strings.TrimPrefix(st.ContainerID, "docker://"))
				podStatus.ContainerStatuses[i].Ready = ready
				break
			}
		}
	}
	podStatus.Conditions = append(podStatus.Conditions, getPodReadyCondition(spec, podStatus.ContainerStatuses)...)

	hostIP, err := kl.GetHostIP()
	if err != nil {
		glog.Errorf("Cannot get host IP: %v", err)
	} else {
		podStatus.HostIP = hostIP.String()
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
// TODO(yifan): Move this to runtime once the runtime interface has been all implemented.
func (kl *Kubelet) findContainer(podFullName string, podUID types.UID, containerName string) (*kubecontainer.Container, error) {
	pods, err := kl.containerManager.GetPods(false)
	if err != nil {
		return nil, err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	return pod.FindContainerByName(containerName), nil
}

// Run a command in a container, returns the combined stdout, stderr as an array of bytes
func (kl *Kubelet) RunInContainer(podFullName string, podUID types.UID, containerName string, cmd []string) ([]byte, error) {
	podUID = kl.podManager.TranslatePodUID(podUID)

	if kl.runner == nil {
		return nil, fmt.Errorf("no runner specified.")
	}
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

	if kl.runner == nil {
		return fmt.Errorf("no runner specified.")
	}
	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return err
	}
	if container == nil {
		return fmt.Errorf("container not found (%q)", containerName)
	}
	return kl.runner.ExecInContainer(string(container.ID), cmd, stdin, stdout, stderr, tty)
}

// PortForward connects to the pod's port and copies data between the port
// and the stream.
func (kl *Kubelet) PortForward(podFullName string, podUID types.UID, port uint16, stream io.ReadWriteCloser) error {
	podUID = kl.podManager.TranslatePodUID(podUID)

	if kl.runner == nil {
		return fmt.Errorf("no runner specified.")
	}

	pods, err := kl.containerManager.GetPods(false)
	if err != nil {
		return err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	return kl.runner.PortForward(&pod, port, stream)
}

// BirthCry sends an event that the kubelet has started up.
func (kl *Kubelet) BirthCry() {
	// Make an event that kubelet restarted.
	kl.recorder.Eventf(kl.nodeRef, "starting", "Starting kubelet.")
}

func (kl *Kubelet) StreamingConnectionIdleTimeout() time.Duration {
	return kl.streamingConnectionIdleTimeout
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {

	uid = kl.podManager.TranslatePodUID(uid)

	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return nil, err
	}
	if len(dockerContainers) == 0 {
		return nil, ErrNoKubeletContainers
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, containerName)
	if !found {
		return nil, ErrContainerNotFound
	}

	ci, err := kl.cadvisor.DockerContainer(dockerContainer.ID, req)
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
