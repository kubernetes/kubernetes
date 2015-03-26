/*
Copyright 2014 Google Inc. All rights reserved.

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
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/envvars"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scheduler"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilErrors "github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

const (
	// The oom_score_adj of the POD infrastructure container. The default is 0, so
	// any value below that makes it *less* likely to get OOM killed.
	podOomScoreAdj = -100

	// Max amount of time to wait for the Docker daemon to come up.
	maxWaitForDocker = 5 * time.Minute

	// Initial node status update frequency and incremental frequency, for faster cluster startup.
	// The update frequency will be increameted linearly, until it reaches status_update_frequency.
	initialNodeStatusUpdateFrequency = 100 * time.Millisecond
	nodeStatusUpdateFrequencyInc     = 500 * time.Millisecond

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
	nodeStatusUpdateFrequency = 2 * time.Second
	// nodeStatusUpdateRetry specifies how many times kubelet retries when posting node status failed.
	nodeStatusUpdateRetry = 5
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
	SyncPods(pods []api.Pod, podSyncTypes map[types.UID]metrics.SyncPodType, mirrorPods map[string]api.Pod,
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
	cloud cloudprovider.Interface) (*Kubelet, error) {
	if rootDirectory == "" {
		return nil, fmt.Errorf("invalid root directory %q", rootDirectory)
	}
	if resyncInterval <= 0 {
		return nil, fmt.Errorf("invalid sync frequency %d", resyncInterval)
	}
	dockerClient = metrics.NewInstrumentedDockerInterface(dockerClient)

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
		listWatch := &cache.ListWatch{
			// TODO: currently, we are watching all nodes. To make it more efficient,
			// we should be watching only a node with Name equal to kubelet's Hostname.
			// To make it possible, we need to add field selector to ListFunc and WatchFunc,
			// and selection by field needs to be implemented in WatchMinions function in pkg/registry/etcd.
			ListFunc: func() (runtime.Object, error) {
				return kubeClient.Nodes().List()
			},
			WatchFunc: func(resourceVersion string) (watch.Interface, error) {
				return kubeClient.Nodes().Watch(
					labels.Everything(), fields.Everything(), resourceVersion)
			},
		}
		cache.NewReflector(listWatch, &api.Node{}, nodeStore, 0).Run()
	}
	nodeLister := &cache.StoreToNodeLister{nodeStore}

	containerGC, err := newContainerGC(dockerClient, containerGCPolicy)
	if err != nil {
		return nil, err
	}
	imageManager, err := newImageManager(dockerClient, cadvisorInterface, imageGCPolicy)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize image manager: %v", err)
	}
	statusManager := newStatusManager(kubeClient)

	klet := &Kubelet{
		hostname:                       hostname,
		dockerClient:                   dockerClient,
		kubeClient:                     kubeClient,
		rootDirectory:                  rootDirectory,
		resyncInterval:                 resyncInterval,
		podInfraContainerImage:         podInfraContainerImage,
		containerRefManager:            kubecontainer.NewRefManager(),
		runner:                         dockertools.NewDockerContainerCommandRunner(dockerClient),
		httpClient:                     &http.Client{},
		pullQPS:                        pullQPS,
		pullBurst:                      pullBurst,
		sourcesReady:                   sourcesReady,
		clusterDomain:                  clusterDomain,
		clusterDNS:                     clusterDNS,
		serviceLister:                  serviceLister,
		nodeLister:                     nodeLister,
		masterServiceNamespace:         masterServiceNamespace,
		prober:                         newProbeHolder(),
		readiness:                      newReadinessStates(),
		streamingConnectionIdleTimeout: streamingConnectionIdleTimeout,
		recorder:                       recorder,
		cadvisor:                       cadvisorInterface,
		containerGC:                    containerGC,
		imageManager:                   imageManager,
		statusManager:                  statusManager,
		cloud:                          cloud,
	}

	klet.podManager = newBasicPodManager(klet.kubeClient)

	dockerCache, err := dockertools.NewDockerCache(dockerClient)
	if err != nil {
		return nil, err
	}
	klet.dockerCache = dockerCache
	klet.podWorkers = newPodWorkers(dockerCache, klet.syncPod, recorder)

	metrics.Register(dockerCache)

	if err = klet.setupDataDirs(); err != nil {
		return nil, err
	}
	if err = klet.volumePluginMgr.InitPlugins(volumePlugins, &volumeHost{klet}); err != nil {
		return nil, err
	}

	if plug, err := network.InitNetworkPlugin(networkPlugins, networkPluginName, &networkHost{klet}); err != nil {
		return nil, err
	} else {
		klet.networkPlugin = plug
	}

	return klet, nil
}

type httpGetter interface {
	Get(url string) (*http.Response, error)
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
	hostname               string
	dockerClient           dockertools.DockerInterface
	dockerCache            dockertools.DockerCache
	kubeClient             client.Interface
	rootDirectory          string
	podInfraContainerImage string
	podWorkers             *podWorkers
	resyncInterval         time.Duration
	sourcesReady           SourcesReadyFn

	podManager podManager

	// Needed to report events for containers belonging to deleted/modified pods.
	// Tracks references for reporting events
	containerRefManager *kubecontainer.RefManager

	// Optional, defaults to simple Docker implementation
	dockerPuller dockertools.DockerPuller
	// Optional, defaults to /logs/ from /var/log
	logServer http.Handler
	// Optional, defaults to simple Docker implementation
	runner dockertools.ContainerCommandRunner
	// Optional, client for http requests, defaults to empty client
	httpClient httpGetter
	// Optional, maximum pull QPS from the docker registry, 0.0 means unlimited.
	pullQPS float32
	// Optional, maximum burst QPS from the docker registry, must be positive if QPS is > 0.0
	pullBurst int

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

	// Network plugin
	networkPlugin network.NetworkPlugin

	// Probe runner holder
	prober probeHolder
	// Container readiness state holder
	readiness *readinessStates

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

	//Cloud provider interface
	cloud cloudprovider.Interface
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
	if kl.dockerPuller == nil {
		kl.dockerPuller = dockertools.NewDockerPuller(kl.dockerClient, kl.pullQPS, kl.pullBurst)
	}
	if kl.kubeClient == nil {
		glog.Warning("No api server defined - no node status update will be sent.")
	}
	go kl.syncNodeStatus()
	kl.statusManager.Start()
	kl.syncLoop(updates, kl)
}

// syncNodeStatus periodically synchronizes node status to master.
func (kl *Kubelet) syncNodeStatus() {
	if kl.kubeClient == nil {
		return
	}

	for feq := initialNodeStatusUpdateFrequency; feq < nodeStatusUpdateFrequency; feq += nodeStatusUpdateFrequencyInc {
		select {
		case <-time.After(feq):
			if err := kl.updateNodeStatus(); err != nil {
				glog.Errorf("Unable to update node status: %v", err)
			}
		}
	}
	for {
		select {
		case <-time.After(nodeStatusUpdateFrequency):
			if err := kl.updateNodeStatus(); err != nil {
				glog.Errorf("Unable to update node status: %v", err)
			}
		}
	}
}

func makeBinds(container *api.Container, podVolumes volumeMap) []string {
	binds := []string{}
	for _, mount := range container.VolumeMounts {
		vol, ok := podVolumes[mount.Name]
		if !ok {
			continue
		}
		b := fmt.Sprintf("%s:%s", vol.GetPath(), mount.MountPath)
		if mount.ReadOnly {
			b += ":ro"
		}
		binds = append(binds, b)
	}
	return binds
}

// A basic interface that knows how to execute handlers
type actionHandler interface {
	Run(podFullName string, uid types.UID, container *api.Container, handler *api.Handler) error
}

func (kl *Kubelet) newActionHandler(handler *api.Handler) actionHandler {
	switch {
	case handler.Exec != nil:
		return &execActionHandler{kubelet: kl}
	case handler.HTTPGet != nil:
		return &httpActionHandler{client: kl.httpClient, kubelet: kl}
	default:
		glog.Errorf("Invalid handler: %v", handler)
		return nil
	}
}

func (kl *Kubelet) runHandler(podFullName string, uid types.UID, container *api.Container, handler *api.Handler) error {
	actionHandler := kl.newActionHandler(handler)
	if actionHandler == nil {
		return fmt.Errorf("invalid handler")
	}
	return actionHandler.Run(podFullName, uid, container, handler)
}

// generateRunContainerOptions generates the RunContainerOptions, which can be used by
// the container runtime to set parameters for launching a container.
func (kl *Kubelet) generateRunContainerOptions(pod *api.Pod, container *api.Container, podVolumes volumeMap, netMode, ipcMode string) (*kubecontainer.RunContainerOptions, error) {
	var err error
	opts := &kubecontainer.RunContainerOptions{
		NetMode: netMode,
		IpcMode: ipcMode,
	}

	opts.Binds = makeBinds(container, podVolumes)
	opts.Envs, err = kl.makeEnvironmentVariables(pod.Namespace, container)
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

// Run a single container from a pod. Returns the docker container ID
func (kl *Kubelet) runContainer(pod *api.Pod, container *api.Container, podVolumes volumeMap, netMode, ipcMode string) (dockertools.DockerID, error) {
	ref, err := kl.containerRefManager.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	opts, err := kl.generateRunContainerOptions(pod, container, podVolumes, netMode, ipcMode)
	if err != nil {
		return "", err
	}

	// TODO(yifan): Replace with RunContainerInPod, so we can eliminate 'netMode', 'ipcMode'
	// by handling the pod infra container in the container runtime's implementation.
	id, err := dockertools.RunContainer(kl.dockerClient, container, pod, opts, kl.containerRefManager, ref, kl.recorder)
	if err != nil {
		return "", err
	}

	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		handlerErr := kl.runHandler(kubecontainer.GetPodFullName(pod), pod.UID, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			kl.killContainerByID(id)
			return dockertools.DockerID(""), fmt.Errorf("failed to call event handler: %v", handlerErr)
		}
	}
	return dockertools.DockerID(id), err
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
func (kl *Kubelet) makeEnvironmentVariables(ns string, container *api.Container) ([]string, error) {
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
	serviceEnv, err := kl.getServiceEnvVarMap(ns)
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
		result = append(result, fmt.Sprintf("%s=%s", value.Name, value.Value))
	}

	// Append remaining service env vars.
	for k, v := range serviceEnv {
		result = append(result, fmt.Sprintf("%s=%s", k, v))
	}
	return result, nil
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

// Kill a docker container
func (kl *Kubelet) killContainer(c *kubecontainer.Container) error {
	return kl.killContainerByID(string(c.ID))
}

func (kl *Kubelet) killContainerByID(ID string) error {
	glog.V(2).Infof("Killing container with id %q", ID)
	kl.readiness.remove(ID)
	err := kl.dockerClient.StopContainer(ID, 10)

	ref, ok := kl.containerRefManager.GetRef(ID)
	if !ok {
		glog.Warningf("No ref for pod '%v'", ID)
	} else {
		// TODO: pass reason down here, and state, or move this call up the stack.
		kl.recorder.Eventf(ref, "killing", "Killing %v", ID)
	}
	return err
}

const (
	PodInfraContainerImage = "kubernetes/pause:latest"
)

// Determined whether the specified pod is allowed to use host networking
func allowHostNetwork(pod *api.Pod) (bool, error) {
	podSource, err := getPodSource(pod)
	if err != nil {
		return false, err
	}
	for _, source := range capabilities.Get().HostNetworkSources {
		if source == podSource {
			return true, nil
		}
	}
	return false, nil
}

// createPodInfraContainer starts the pod infra container for a pod. Returns the docker container ID of the newly created container.
func (kl *Kubelet) createPodInfraContainer(pod *api.Pod) (dockertools.DockerID, error) {

	// Use host networking if specified and allowed.
	netNamespace := ""
	var ports []api.ContainerPort

	if pod.Spec.HostNetwork {
		netNamespace = "host"
	} else {
		// Docker only exports ports from the pod infra container.  Let's
		// collect all of the relevant ports and export them.
		for _, container := range pod.Spec.Containers {
			ports = append(ports, container.Ports...)
		}
	}

	container := &api.Container{
		Name:  dockertools.PodInfraContainerName,
		Image: kl.podInfraContainerImage,
		Ports: ports,
	}
	ref, err := kl.containerRefManager.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}
	// TODO: make this a TTL based pull (if image older than X policy, pull)
	ok, err := kl.dockerPuller.IsImagePresent(container.Image)
	if err != nil {
		if ref != nil {
			kl.recorder.Eventf(ref, "failed", "Failed to inspect image %q: %v", container.Image, err)
		}
		return "", err
	}
	if !ok {
		if err := kl.pullImage(container.Image, ref); err != nil {
			return "", err
		}
	}
	if ref != nil {
		kl.recorder.Eventf(ref, "pulled", "Successfully pulled image %q", container.Image)
	}

	id, err := kl.runContainer(pod, container, nil, netNamespace, "")
	if err != nil {
		return "", err
	}

	// Set OOM score of POD container to lower than those of the other
	// containers in the pod. This ensures that it is killed only as a last
	// resort.
	containerInfo, err := kl.dockerClient.InspectContainer(string(id))
	if err != nil {
		return "", err
	}

	// Ensure the PID actually exists, else we'll move ourselves.
	if containerInfo.State.Pid == 0 {
		return "", fmt.Errorf("failed to get init PID for Docker pod infra container %q", string(id))
	}
	return id, util.ApplyOomScoreAdj(containerInfo.State.Pid, podOomScoreAdj)
}

func (kl *Kubelet) pullImage(img string, ref *api.ObjectReference) error {
	start := time.Now()
	defer func() {
		metrics.ImagePullLatency.Observe(metrics.SinceInMicroseconds(start))
	}()

	if err := kl.dockerPuller.Pull(img); err != nil {
		if ref != nil {
			kl.recorder.Eventf(ref, "failed", "Failed to pull image %q: %v", img, err)
		}
		return err
	}
	if ref != nil {
		kl.recorder.Eventf(ref, "pulled", "Successfully pulled image %q", img)
	}
	return nil
}

// Kill all running containers in a pod (includes the pod infra container).
func (kl *Kubelet) killPod(runningPod kubecontainer.Pod) error {
	// Send the kills in parallel since they may take a long time.
	errs := make(chan error, len(runningPod.Containers))
	wg := sync.WaitGroup{}
	for _, container := range runningPod.Containers {
		wg.Add(1)
		go func(container *kubecontainer.Container) {
			defer util.HandleCrash()
			err := kl.killContainer(container)
			if err != nil {
				glog.Errorf("Failed to delete container: %v; Skipping pod %q", err, runningPod.ID)
				errs <- err
			}
			wg.Done()
		}(container)
	}
	wg.Wait()
	close(errs)
	if len(errs) > 0 {
		errList := []error{}
		for err := range errs {
			errList = append(errList, err)
		}
		return fmt.Errorf("failed to delete containers (%v)", errList)
	}
	return nil
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

func (kl *Kubelet) shouldContainerBeRestarted(container *api.Container, pod *api.Pod) bool {
	podFullName := kubecontainer.GetPodFullName(pod)
	// Check RestartPolicy for dead container
	recentContainers, err := dockertools.GetRecentDockerContainersWithNameAndUUID(kl.dockerClient, podFullName, pod.UID, container.Name)
	if err != nil {
		glog.Errorf("Error listing recent containers for pod %q: %v", podFullName, err)
		// TODO(dawnchen): error handling here?
	}
	// set dead containers to unready state
	for _, c := range recentContainers {
		kl.readiness.remove(c.ID)
	}

	if len(recentContainers) > 0 {
		if pod.Spec.RestartPolicy == api.RestartPolicyNever {
			glog.Infof("Already ran container %q of pod %q, do nothing", container.Name, podFullName)
			return false

		}
		if pod.Spec.RestartPolicy == api.RestartPolicyOnFailure {
			// Check the exit code of last run
			if recentContainers[0].State.ExitCode == 0 {
				glog.Infof("Already successfully ran container %q of pod %q, do nothing", container.Name, podFullName)
				return false
			}
		}
	}
	return true
}

// Finds an infra container for a pod given by podFullName and UID in dockerContainers. If there is an infra container
// return its ID and true, otherwise it returns empty ID and false.
func (kl *Kubelet) getPodInfraContainer(podFullName string, uid types.UID,
	dockerContainers dockertools.DockerContainers) (dockertools.DockerID, bool) {
	if podInfraDockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, dockertools.PodInfraContainerName); found {
		podInfraContainerID := dockertools.DockerID(podInfraDockerContainer.ID)
		return podInfraContainerID, true
	}
	return "", false
}

// Attempts to start a container pulling the image before that if necessary. It returns DockerID of a started container
// if it was successful, and a non-nil error otherwise.
func (kl *Kubelet) pullImageAndRunContainer(pod *api.Pod, container *api.Container, podVolumes *volumeMap,
	podInfraContainerID dockertools.DockerID) (dockertools.DockerID, error) {
	podFullName := kubecontainer.GetPodFullName(pod)
	ref, err := kl.containerRefManager.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}
	if container.ImagePullPolicy != api.PullNever {
		present, err := kl.dockerPuller.IsImagePresent(container.Image)
		if err != nil {
			if ref != nil {
				kl.recorder.Eventf(ref, "failed", "Failed to inspect image %q: %v", container.Image, err)
			}
			glog.Errorf("Failed to inspect image %q: %v; skipping pod %q container %q", container.Image, err, podFullName, container.Name)
			return "", err
		}
		if container.ImagePullPolicy == api.PullAlways ||
			(container.ImagePullPolicy == api.PullIfNotPresent && (!present)) {
			if err := kl.pullImage(container.Image, ref); err != nil {
				return "", err
			}
		}
	}
	// TODO(dawnchen): Check RestartPolicy.DelaySeconds before restart a container
	namespaceMode := fmt.Sprintf("container:%v", podInfraContainerID)
	containerID, err := kl.runContainer(pod, container, *podVolumes, namespaceMode, namespaceMode)
	if err != nil {
		// TODO(bburns) : Perhaps blacklist a container after N failures?
		glog.Errorf("Error running pod %q container %q: %v", podFullName, container.Name, err)
		return "", err
	}
	return containerID, nil
}

// Structure keeping information on changes that need to happen for a pod. The semantics is as follows:
// - startInfraContainer is true if new Infra Containers have to be started and old one (if running) killed.
//   Additionally if it is true then containersToKeep have to be empty
// - infraContainerId have to be set iff startInfraContainer is false. It stores dockerID of running Infra Container
// - containersToStart keeps indices of Specs of containers that have to be started.
// - containersToKeep stores mapping from dockerIDs of running containers to indices of their Specs for containers that
//   should be kept running. If startInfraContainer is false then it contains an entry for infraContainerId (mapped to -1).
//   It shouldn't be the case where containersToStart is empty and containersToKeep contains only infraContainerId. In such case
//   Infra Container should be killed, hence it's removed from this map.
// - all running containers which are NOT contained in containersToKeep should be killed.
type podContainerChangesSpec struct {
	startInfraContainer bool
	infraContainerId    dockertools.DockerID
	containersToStart   map[int]empty
	containersToKeep    map[dockertools.DockerID]int
}

func (kl *Kubelet) computePodContainerChanges(pod *api.Pod, runningPod kubecontainer.Pod) (podContainerChangesSpec, error) {
	podFullName := kubecontainer.GetPodFullName(pod)
	uid := pod.UID
	glog.V(4).Infof("Syncing Pod %+v, podFullName: %q, uid: %q", pod, podFullName, uid)

	err := kl.makePodDataDirs(pod)
	if err != nil {
		return podContainerChangesSpec{}, err
	}

	containersToStart := make(map[int]empty)
	containersToKeep := make(map[dockertools.DockerID]int)
	createPodInfraContainer := false
	var podStatus api.PodStatus

	var podInfraContainerID dockertools.DockerID
	podInfraContainer := runningPod.FindContainerByName(dockertools.PodInfraContainerName)
	if podInfraContainer != nil {
		glog.V(4).Infof("Found infra pod for %q", podFullName)
		podInfraContainerID = dockertools.DockerID(podInfraContainer.ID)
		containersToKeep[podInfraContainerID] = -1
		podStatus, err = kl.GetPodStatus(podFullName)
		if err != nil {
			glog.Errorf("Unable to get pod with name %q and uid %q info with error(%v)", podFullName, uid, err)
		}
	} else {
		glog.V(2).Infof("No Infra Container for %q found. All containers will be restarted.", podFullName)
		createPodInfraContainer = true
	}

	for index, container := range pod.Spec.Containers {
		expectedHash := dockertools.HashContainer(&container)

		c := runningPod.FindContainerByName(container.Name)
		if c != nil {
			containerID := dockertools.DockerID(c.ID)
			hash := c.Hash
			glog.V(3).Infof("pod %q container %q exists as %v", podFullName, container.Name, containerID)

			if !createPodInfraContainer {
				// look for changes in the container.

				containerChanged := hash != 0 && hash != expectedHash
				if !containerChanged {
					result, err := kl.probeContainer(pod, podStatus, container, string(c.ID), c.Created)
					if err != nil {
						// TODO(vmarmol): examine this logic.
						glog.V(2).Infof("probe no-error: %q", container.Name)
						containersToKeep[containerID] = index
						continue
					}
					if result == probe.Success {
						glog.V(4).Infof("probe success: %q", container.Name)
						containersToKeep[containerID] = index
						continue
					}
					glog.Infof("pod %q container %q is unhealthy (probe result: %v). Container will be killed and re-created.", podFullName, container.Name, result)
					containersToStart[index] = empty{}
				} else {
					glog.Infof("pod %q container %q hash changed (%d vs %d). Pod will be killed and re-created.", podFullName, container.Name, hash, expectedHash)
					createPodInfraContainer = true
					delete(containersToKeep, podInfraContainerID)
					// If we are to restart Infra Container then we move containersToKeep into containersToStart
					// if RestartPolicy allows restarting failed containers.
					if pod.Spec.RestartPolicy != api.RestartPolicyNever {
						for _, v := range containersToKeep {
							containersToStart[v] = empty{}
						}
					}
					containersToStart[index] = empty{}
					containersToKeep = make(map[dockertools.DockerID]int)
				}
			} else { // createPodInfraContainer == true and Container exists
				// If we're creating infra containere everything will be killed anyway
				// If RestartPolicy is Always or OnFailure we restart containers that were running before we
				// killed them when restarting Infra Container.
				if pod.Spec.RestartPolicy != api.RestartPolicyNever {
					glog.V(1).Infof("Infra Container is being recreated. %q will be restarted.", container.Name)
					containersToStart[index] = empty{}
				}
				continue
			}
		} else {
			if kl.shouldContainerBeRestarted(&container, pod) {
				// If we are here it means that the container is dead and sould be restarted, or never existed and should
				// be created. We may be inserting this ID again if the container has changed and it has
				// RestartPolicy::Always, but it's not a big deal.
				glog.V(3).Infof("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				containersToStart[index] = empty{}
			}
		}
	}

	// After the loop one of the following should be true:
	// - createPodInfraContainer is true and containersToKeep is empty
	// - createPodInfraContainer is false and containersToKeep contains at least ID of Infra Container

	// If Infra container is the last running one, we don't want to keep it.
	if !createPodInfraContainer && len(containersToStart) == 0 && len(containersToKeep) == 1 {
		containersToKeep = make(map[dockertools.DockerID]int)
	}

	return podContainerChangesSpec{
		startInfraContainer: createPodInfraContainer,
		infraContainerId:    podInfraContainerID,
		containersToStart:   containersToStart,
		containersToKeep:    containersToKeep,
	}, nil
}

// Check whether we can run the specified pod.
func (kl *Kubelet) canRunPod(pod *api.Pod) error {
	if pod.Spec.HostNetwork {
		allowed, err := allowHostNetwork(pod)
		if err != nil {
			return err
		}
		if !allowed {
			return fmt.Errorf("pod with UID %q specified host networking, but is disallowed", pod.UID)
		}
	}
	// TODO(vmarmol): Check Privileged too.
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
		status, err := kl.generatePodStatusByPod(pod)
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
	err := kl.canRunPod(pod)
	if err != nil {
		kl.killPod(runningPod)
		return err
	}

	containerChanges, err := kl.computePodContainerChanges(pod, runningPod)
	glog.V(3).Infof("Got container changes for pod %q: %+v", podFullName, containerChanges)
	if err != nil {
		return err
	}

	if containerChanges.startInfraContainer || (len(containerChanges.containersToKeep) == 0 && len(containerChanges.containersToStart) == 0) {
		if len(containerChanges.containersToKeep) == 0 && len(containerChanges.containersToStart) == 0 {
			glog.V(4).Infof("Killing Infra Container for %q becase all other containers are dead.", podFullName)
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
			_, keep := containerChanges.containersToKeep[dockertools.DockerID(container.ID)]
			if !keep {
				glog.V(3).Infof("Killing unwanted container %+v", container)
				err = kl.killContainer(container)
				if err != nil {
					glog.Errorf("Error killing container: %v", err)
				}
			}
		}
	}

	// Starting phase: if we should create infra container then we do it first
	var ref *api.ObjectReference
	var podVolumes volumeMap
	podInfraContainerID := containerChanges.infraContainerId
	if containerChanges.startInfraContainer && (len(containerChanges.containersToStart) > 0) {
		ref, err = api.GetReference(pod)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %q: '%v'", podFullName, err)
		}
		glog.Infof("Creating pod infra container for %q", podFullName)
		podInfraContainerID, err = kl.createPodInfraContainer(pod)

		// Call the networking plugin
		if err == nil {
			err = kl.networkPlugin.SetUpPod(pod.Namespace, pod.Name, podInfraContainerID)
		}
		if err != nil {
			glog.Errorf("Failed to create pod infra container: %v; Skipping pod %q", err, podFullName)
			return err
		}
	}

	// Mount volumes
	podVolumes, err = kl.mountExternalVolumes(pod)
	if err != nil {
		if ref != nil {
			kl.recorder.Eventf(ref, "failedMount",
				"Unable to mount volumes for pod %q: %v", podFullName, err)
		}
		glog.Errorf("Unable to mount volumes for pod %q: %v; skipping pod", podFullName, err)
		return err
	}

	// Start everything
	for container := range containerChanges.containersToStart {
		glog.V(4).Infof("Creating container %+v", pod.Spec.Containers[container])
		kl.pullImageAndRunContainer(pod, &pod.Spec.Containers[container], &podVolumes, podInfraContainerID)
	}

	if mirrorPod == nil && isStaticPod(pod) {
		glog.V(4).Infof("Creating a mirror pod %q", podFullName)
		// To make sure we will properly update static pod status we need to delete
		// it from status manager. Otherwise it is possible that we will miss manual
		// deletion of mirror pod in apiserver and will never reset its status to
		// Running after recreating it.
		kl.statusManager.DeletePodStatus(podFullName)
		if err := kl.podManager.CreateMirrorPod(*pod, kl.hostname); err != nil {
			glog.Errorf("Failed creating a mirror pod %q: %#v", podFullName, err)
		}
	}

	return nil
}

// Stores all volumes defined by the set of pods into a map.
// Keys for each entry are in the format (POD_ID)/(VOLUME_NAME)
func getDesiredVolumes(pods []api.Pod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Spec.Volumes {
			identifier := path.Join(string(pod.UID), volume.Name)
			desiredVolumes[identifier] = volume
		}
	}
	return desiredVolumes
}

func (kl *Kubelet) cleanupOrphanedPods(pods []api.Pod) error {
	desired := util.NewStringSet()
	for i := range pods {
		desired.Insert(string(pods[i].UID))
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
func (kl *Kubelet) cleanupOrphanedVolumes(pods []api.Pod, running []*docker.Container) error {
	desiredVolumes := getDesiredVolumes(pods)
	currentVolumes := kl.getPodVolumesFromDisk()
	runningSet := util.StringSet{}
	for ix := range running {
		if len(running[ix].Name) == 0 {
			glog.V(2).Infof("Found running container ix=%d with info: %+v", ix, running[ix])
		}
		containerName, _, err := dockertools.ParseDockerName(running[ix].Name)
		if err != nil {
			continue
		}
		runningSet.Insert(string(containerName.PodUID))
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
			//TODO (jonesdl) This should not block other kubelet synchronization procedures
			err := vol.TearDown()
			if err != nil {
				glog.Errorf("Could not tear down volume %q: %v", name, err)
			}
		}
	}
	return nil
}

// SyncPods synchronizes the configured list of pods (desired state) with the host current state.
func (kl *Kubelet) SyncPods(allPods []api.Pod, podSyncTypes map[types.UID]metrics.SyncPodType,
	mirrorPods map[string]api.Pod, start time.Time) error {
	defer func() {
		metrics.SyncPodsLatency.Observe(metrics.SinceInMicroseconds(start))
	}()

	// Remove obsolete entries in podStatus where the pod is no longer considered bound to this node.
	podFullNames := make(map[string]bool)
	for _, pod := range allPods {
		podFullNames[kubecontainer.GetPodFullName(&pod)] = true
	}
	kl.statusManager.RemoveOrphanedStatuses(podFullNames)

	// Filter out the rejected pod. They don't have running containers.
	kl.handleNotFittingPods(allPods)
	var pods []api.Pod
	for _, pod := range allPods {
		status, ok := kl.statusManager.GetPodStatus(kubecontainer.GetPodFullName(&pod))
		if ok && status.Phase == api.PodFailed {
			continue
		}
		pods = append(pods, pod)
	}

	glog.V(4).Infof("Desired: %#v", pods)
	var err error
	desiredPods := make(map[types.UID]empty)

	runningPods, err := kl.dockerCache.GetPods()
	if err != nil {
		glog.Errorf("Error listing containers: %#v", err)
		return err
	}

	// Check for any containers that need starting
	for ix := range pods {
		pod := &pods[ix]
		podFullName := kubecontainer.GetPodFullName(pod)
		uid := pod.UID
		desiredPods[uid] = empty{}

		// Run the sync in an async manifest worker.
		var mirrorPod *api.Pod = nil
		if m, ok := mirrorPods[podFullName]; ok {
			mirrorPod = &m
		}
		kl.podWorkers.UpdatePod(pod, mirrorPod, func() {
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

	// Kill any containers we don't need.
	killed := []string{}
	for _, pod := range runningPods {
		if _, found := desiredPods[pod.ID]; found {
			// syncPod() will handle this one.
			continue
		}

		// Kill all the containers in the unidentified pod.
		for _, c := range pod.Containers {
			// call the networking plugin for teardown
			if c.Name == dockertools.PodInfraContainerName {
				err := kl.networkPlugin.TearDownPod(pod.Namespace, pod.Name, dockertools.DockerID(c.ID))
				if err != nil {
					glog.Errorf("Network plugin pre-delete method returned an error: %v", err)
				}
			}
			glog.V(1).Infof("Killing unwanted container %+v", c)
			err = kl.killContainer(c)
			if err != nil {
				glog.Errorf("Error killing container %+v: %v", c, err)
			} else {
				killed = append(killed, string(c.ID))
			}
		}
	}

	running, err := dockertools.GetRunningContainers(kl.dockerClient, killed)
	if err != nil {
		glog.Errorf("Failed to poll container state: %v", err)
		return err
	}

	// Remove any orphaned volumes.
	err = kl.cleanupOrphanedVolumes(pods, running)
	if err != nil {
		return err
	}

	// Remove any orphaned pods.
	err = kl.cleanupOrphanedPods(pods)
	if err != nil {
		return err
	}

	// Remove any orphaned mirror pods.
	kl.podManager.DeleteOrphanedMirrorPods()

	return err
}

type podsByCreationTime []api.Pod

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
func checkHostPortConflicts(pods []api.Pod) (fitting []api.Pod, notFitting []api.Pod) {
	ports := map[int]bool{}
	extract := func(p *api.ContainerPort) int { return p.HostPort }

	// Respect the pod creation order when resolving conflicts.
	sort.Sort(podsByCreationTime(pods))

	for i := range pods {
		pod := &pods[i]
		if errs := validation.AccumulateUniquePorts(pod.Spec.Containers, ports, extract); len(errs) != 0 {
			glog.Errorf("Pod %q: HostPort is already allocated, ignoring: %v", kubecontainer.GetPodFullName(pod), errs)
			notFitting = append(notFitting, *pod)
			continue
		}
		fitting = append(fitting, *pod)
	}
	return
}

// checkCapacityExceeded detects pods that exceeds node's resources.
func (kl *Kubelet) checkCapacityExceeded(pods []api.Pod) (fitting []api.Pod, notFitting []api.Pod) {
	info, err := kl.GetCachedMachineInfo()
	if err != nil {
		glog.Error("error getting machine info: %v", err)
		return pods, []api.Pod{}
	}

	// Respect the pod creation order when resolving conflicts.
	sort.Sort(podsByCreationTime(pods))

	capacity := CapacityFromMachineInfo(info)
	return scheduler.CheckPodsExceedingCapacity(pods, capacity)
}

// checkNodeSelectorMatching detects pods that do not match node's labels.
func (kl *Kubelet) checkNodeSelectorMatching(pods []api.Pod) (fitting []api.Pod, notFitting []api.Pod) {
	node, err := kl.GetNode()
	if err != nil {
		glog.Errorf("error getting node: %v", err)
		return pods, []api.Pod{}
	}
	for _, pod := range pods {
		if !scheduler.PodMatchesNodeLabels(&pod, node) {
			notFitting = append(notFitting, pod)
			continue
		}
		fitting = append(fitting, pod)
	}
	return
}

// handleNotfittingPods handles pods that do not fit on the node.
// Currently conflicts on Port.HostPort values, matching node's labels and exceeding node's capacity are handled.
func (kl *Kubelet) handleNotFittingPods(pods []api.Pod) {
	fitting, notFitting := checkHostPortConflicts(pods)
	for _, pod := range notFitting {
		kl.recorder.Eventf(&pod, "hostPortConflict", "Cannot start the pod due to host port conflict.")
		kl.statusManager.SetPodStatus(&pod, api.PodStatus{
			Phase:   api.PodFailed,
			Message: "Pod cannot be started due to host port conflict"})
	}
	fitting, notFitting = kl.checkNodeSelectorMatching(fitting)
	for _, pod := range notFitting {
		kl.recorder.Eventf(&pod, "nodeSelectorMismatching", "Cannot start the pod due to node selector mismatch.")
		kl.statusManager.SetPodStatus(&pod, api.PodStatus{
			Phase:   api.PodFailed,
			Message: "Pod cannot be started due to node selector mismatch"})
	}
	fitting, notFitting = kl.checkCapacityExceeded(fitting)
	for _, pod := range notFitting {
		kl.recorder.Eventf(&pod, "capacityExceeded", "Cannot start the pod due to exceeded capacity.")
		kl.statusManager.SetPodStatus(&pod, api.PodStatus{
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
	for {
		unsyncedPod := false
		podSyncTypes := make(map[types.UID]metrics.SyncPodType)
		select {
		case u := <-updates:
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

// Returns Docker version for this Kubelet.
func (kl *Kubelet) GetDockerVersion() ([]uint, error) {
	if kl.dockerClient == nil {
		return nil, fmt.Errorf("no Docker client")
	}
	dockerRunner := dockertools.NewDockerContainerCommandRunner(kl.dockerClient)
	return dockerRunner.GetDockerServerVersion()
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
	return strings.Replace(cStatus.ContainerID, dockertools.DockerPrefix, "", 1), nil
}

// GetKubeletContainerLogs returns logs from the container
// TODO: this method is returning logs of random container attempts, when it should be returning the most recent attempt
// or all of them.
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	podStatus, err := kl.GetPodStatus(podFullName)
	if err != nil {
		if err == dockertools.ErrNoContainersInPod {
			return fmt.Errorf("pod %q not found\n", podFullName)
		} else {
			return fmt.Errorf("failed to get status for pod %q - %v", podFullName, err)
		}
	}

	if err := kl.validatePodPhase(&podStatus); err != nil {
		return err
	}
	dockerContainerID, err := kl.validateContainerStatus(&podStatus, containerName)
	if err != nil {
		return err
	}
	return dockertools.GetKubeletDockerContainerLogs(kl.dockerClient, dockerContainerID, tail, follow, stdout, stderr)
}

// GetHostname Returns the hostname as the kubelet sees it.
func (kl *Kubelet) GetHostname() string {
	return kl.hostname
}

// Returns host IP or nil in case of error.
func (kl *Kubelet) GetHostIP() (net.IP, error) {
	node, err := kl.GetNode()
	if err != nil {
		return nil, fmt.Errorf("Cannot get node: %v", err)
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
	return nil, fmt.Errorf("Host IP unknown; known addresses: %v", addresses)
}

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pods.
func (kl *Kubelet) GetPods() []api.Pod {
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
		glog.Error("error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Status.NodeInfo.BootID = info.BootID
		node.Status.Capacity = CapacityFromMachineInfo(info)
	}

	currentTime := util.Now()
	newCondition := api.NodeCondition{
		Type:          api.NodeReady,
		Status:        api.ConditionTrue,
		Reason:        fmt.Sprintf("kubelet is posting ready status"),
		LastProbeTime: currentTime,
	}
	updated := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			newCondition.LastTransitionTime = node.Status.Conditions[i].LastTransitionTime
			node.Status.Conditions[i] = newCondition
			updated = true
		}
	}
	if !updated {
		newCondition.LastTransitionTime = currentTime
		node.Status.Conditions = append(node.Status.Conditions, newCondition)
	}

	_, err = kl.kubeClient.Nodes().Update(node)
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
	return kl.generatePodStatus(podFullName)
}

func (kl *Kubelet) generatePodStatus(podFullName string) (api.PodStatus, error) {
	pod, found := kl.GetPodByFullName(podFullName)
	if !found {
		return api.PodStatus{}, fmt.Errorf("couldn't find pod %q", podFullName)
	}
	return kl.generatePodStatusByPod(pod)
}

// By passing the pod directly, this method avoids pod lookup, which requires
// grabbing a lock.
func (kl *Kubelet) generatePodStatusByPod(pod *api.Pod) (api.PodStatus, error) {
	podFullName := kubecontainer.GetPodFullName(pod)
	glog.V(3).Infof("Generating status for %q", podFullName)

	spec := &pod.Spec
	podStatus, err := dockertools.GetDockerPodStatus(kl.dockerClient, *spec, podFullName, pod.UID)

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
				podStatus.ContainerStatuses[i].Ready = kl.readiness.IsReady(st)
				break
			}
		}
	}
	podStatus.Conditions = append(podStatus.Conditions, getPodReadyCondition(spec, podStatus.ContainerStatuses)...)
	podStatus.Host = kl.GetHostname()
	hostIP, err := kl.GetHostIP()
	if err != nil {
		return api.PodStatus{}, fmt.Errorf("Cannot get host IP: %v", err)
	}
	podStatus.HostIP = hostIP.String()

	return *podStatus, nil
}

// Returns logs of current machine.
func (kl *Kubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	// TODO: whitelist logs we are willing to serve
	kl.logServer.ServeHTTP(w, req)
}

// Run a command in a container, returns the combined stdout, stderr as an array of bytes
func (kl *Kubelet) RunInContainer(podFullName string, uid types.UID, container string, cmd []string) ([]byte, error) {
	uid = kl.podManager.TranslatePodUID(uid)

	if kl.runner == nil {
		return nil, fmt.Errorf("no runner specified.")
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return nil, err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, container)
	if !found {
		return nil, fmt.Errorf("container not found (%q)", container)
	}
	return kl.runner.RunInContainer(dockerContainer.ID, cmd)
}

// ExecInContainer executes a command in a container, connecting the supplied
// stdin/stdout/stderr to the command's IO streams.
func (kl *Kubelet) ExecInContainer(podFullName string, uid types.UID, container string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	uid = kl.podManager.TranslatePodUID(uid)

	if kl.runner == nil {
		return fmt.Errorf("no runner specified.")
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, container)
	if !found {
		return fmt.Errorf("container not found (%q)", container)
	}
	return kl.runner.ExecInContainer(dockerContainer.ID, cmd, stdin, stdout, stderr, tty)
}

// PortForward connects to the pod's port and copies data between the port
// and the stream.
func (kl *Kubelet) PortForward(podFullName string, uid types.UID, port uint16, stream io.ReadWriteCloser) error {
	uid = kl.podManager.TranslatePodUID(uid)

	if kl.runner == nil {
		return fmt.Errorf("no runner specified.")
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return err
	}
	podInfraContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, dockertools.PodInfraContainerName)
	if !found {
		return fmt.Errorf("Unable to find pod infra container for pod %q, uid %v", podFullName, uid)
	}
	return kl.runner.PortForward(podInfraContainer.ID, port, stream)
}

// BirthCry sends an event that the kubelet has started up.
func (kl *Kubelet) BirthCry() {
	// Make an event that kubelet restarted.
	// TODO: get the real minion object of ourself,
	// and use the real minion name and UID.
	ref := &api.ObjectReference{
		Kind:      "Minion",
		Name:      kl.hostname,
		UID:       types.UID(kl.hostname),
		Namespace: api.NamespaceDefault,
	}
	kl.recorder.Eventf(ref, "starting", "Starting kubelet.")
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

// GetRootInfo returns stats (from Cadvisor) of current machine (root container).
func (kl *Kubelet) GetRootInfo(req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	return kl.cadvisor.ContainerInfo("/", req)
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
