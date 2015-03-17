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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/envvars"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilErrors "github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000

	// The oom_score_adj of the POD infrastructure container. The default is 0, so
	// any value below that makes it *less* likely to get OOM killed.
	podOomScoreAdj = -100

	// Max amount of time to wait for the Docker daemon to come up.
	maxWaitForDocker = 5 * time.Minute

	// Initial node status update frequency and incremental frequency, for faster cluster startup.
	// The update frequency will be increameted linearly, until it reaches status_update_frequency.
	initialNodeStatusUpdateFrequency = 100 * time.Millisecond
	nodeStatusUpdateFrequencyInc     = 500 * time.Millisecond

	// The retry count for updating node status at each sync period.
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
	SyncPods(pods []api.Pod, podSyncTypes map[types.UID]metrics.SyncPodType, mirrorPods util.StringSet,
		startTime time.Time) error
}

type SourcesReadyFn func() bool

type volumeMap map[string]volume.Interface

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
	volumePlugins []volume.Plugin,
	streamingConnectionIdleTimeout time.Duration,
	recorder record.EventRecorder,
	cadvisorInterface cadvisor.Interface,
	statusUpdateFrequency time.Duration) (*Kubelet, error) {
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

	containerGC, err := newContainerGC(dockerClient, containerGCPolicy)
	if err != nil {
		return nil, err
	}

	klet := &Kubelet{
		hostname:                       hostname,
		dockerClient:                   dockerClient,
		kubeClient:                     kubeClient,
		rootDirectory:                  rootDirectory,
		statusUpdateFrequency:          statusUpdateFrequency,
		resyncInterval:                 resyncInterval,
		podInfraContainerImage:         podInfraContainerImage,
		dockerIDToRef:                  map[dockertools.DockerID]*api.ObjectReference{},
		runner:                         dockertools.NewDockerContainerCommandRunner(dockerClient),
		httpClient:                     &http.Client{},
		pullQPS:                        pullQPS,
		pullBurst:                      pullBurst,
		sourcesReady:                   sourcesReady,
		clusterDomain:                  clusterDomain,
		clusterDNS:                     clusterDNS,
		serviceLister:                  serviceLister,
		masterServiceNamespace:         masterServiceNamespace,
		prober:                         newProbeHolder(),
		readiness:                      newReadinessStates(),
		streamingConnectionIdleTimeout: streamingConnectionIdleTimeout,
		recorder:                       recorder,
		cadvisor:                       cadvisorInterface,
		containerGC:                    containerGC,
	}

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

	klet.podStatuses = make(map[string]api.PodStatus)

	klet.mirrorManager = newBasicMirrorManager(klet.kubeClient)

	return klet, nil
}

type httpGetter interface {
	Get(url string) (*http.Response, error)
}

type serviceLister interface {
	List() (api.ServiceList, error)
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
	statusUpdateFrequency  time.Duration
	resyncInterval         time.Duration
	sourcesReady           SourcesReadyFn

	// Protects the pods array
	// We make complete array copies out of this while locked, which is OK because once added to this array,
	// pods are immutable
	podLock sync.RWMutex
	pods    []api.Pod
	// Record the set of mirror pods (see mirror_manager.go for more details);
	// similar to pods, this is not immutable and is protected by the same podLock.
	// Note that Kubelet.pods do not contain mirror pods as they are filtered
	// out beforehand.
	mirrorPods util.StringSet

	// Needed to report events for containers belonging to deleted/modified pods.
	// Tracks references for reporting events
	dockerIDToRef map[dockertools.DockerID]*api.ObjectReference
	refLock       sync.RWMutex

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

	// Volume plugins.
	volumePluginMgr volume.PluginMgr

	// probe runner holder
	prober probeHolder
	// container readiness state holder
	readiness *readinessStates

	// how long to keep idle streaming command execution/port forwarding
	// connections open before terminating them
	streamingConnectionIdleTimeout time.Duration

	// the EventRecorder to use
	recorder record.EventRecorder

	// A pod status cache currently used to store rejected pods and their statuses.
	podStatusesLock sync.RWMutex
	podStatuses     map[string]api.PodStatus

	// A mirror pod manager which provides helper functions.
	mirrorManager mirrorManager

	// Policy for handling garbage collection of dead containers.
	containerGC containerGC
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

func (kl *Kubelet) GarbageCollectLoop() {
	util.Forever(func() {
		if err := kl.containerGC.GarbageCollect(); err != nil {
			glog.Errorf("Container garbage collect failed: %v", err)
		}
	}, time.Minute*1)
}

func (kl *Kubelet) getPodStatusFromCache(podFullName string) (api.PodStatus, bool) {
	kl.podStatusesLock.RLock()
	defer kl.podStatusesLock.RUnlock()
	status, ok := kl.podStatuses[podFullName]
	return status, ok
}

func (kl *Kubelet) setPodStatusInCache(podFullName string, status api.PodStatus) {
	kl.podStatusesLock.Lock()
	defer kl.podStatusesLock.Unlock()
	kl.podStatuses[podFullName] = status
}

func (kl *Kubelet) removeOrphanedStatuses(podFullNames map[string]bool) {
	kl.podStatusesLock.Lock()
	defer kl.podStatusesLock.Unlock()
	for key := range kl.podStatuses {
		if _, ok := podFullNames[key]; !ok {
			glog.V(5).Infof("Removing %q from status map.", key)
			delete(kl.podStatuses, key)
		}
	}
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
	kl.syncLoop(updates, kl)
}

// syncNodeStatus periodically synchronizes node status to master.
func (kl *Kubelet) syncNodeStatus() {
	if kl.kubeClient == nil {
		return
	}
	for feq := initialNodeStatusUpdateFrequency; feq < kl.statusUpdateFrequency; feq += nodeStatusUpdateFrequencyInc {
		select {
		case <-time.After(feq):
			if err := kl.updateNodeStatus(); err != nil {
				glog.Errorf("Unable to update node status: %v", err)
			}
		}
	}
	for {
		select {
		case <-time.After(kl.statusUpdateFrequency):
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

func makePortsAndBindings(container *api.Container) (map[docker.Port]struct{}, map[docker.Port][]docker.PortBinding) {
	exposedPorts := map[docker.Port]struct{}{}
	portBindings := map[docker.Port][]docker.PortBinding{}
	for _, port := range container.Ports {
		exteriorPort := port.HostPort
		if exteriorPort == 0 {
			// No need to do port binding when HostPort is not specified
			continue
		}
		interiorPort := port.ContainerPort
		// Some of this port stuff is under-documented voodoo.
		// See http://stackoverflow.com/questions/20428302/binding-a-port-to-a-host-interface-using-the-rest-api
		var protocol string
		switch strings.ToUpper(string(port.Protocol)) {
		case "UDP":
			protocol = "/udp"
		case "TCP":
			protocol = "/tcp"
		default:
			glog.Warningf("Unknown protocol %q: defaulting to TCP", port.Protocol)
			protocol = "/tcp"
		}
		dockerPort := docker.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}
		portBindings[dockerPort] = []docker.PortBinding{
			{
				HostPort: strconv.Itoa(exteriorPort),
				HostIP:   port.HostIP,
			},
		}
	}
	return exposedPorts, portBindings
}

func milliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		// zero milliCPU means unset. Use kernel default.
		return 0
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

func makeCapabilites(capAdd []api.CapabilityType, capDrop []api.CapabilityType) ([]string, []string) {
	var (
		addCaps  []string
		dropCaps []string
	)
	for _, cap := range capAdd {
		addCaps = append(addCaps, string(cap))
	}
	for _, cap := range capDrop {
		dropCaps = append(dropCaps, string(cap))
	}
	return addCaps, dropCaps
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

// fieldPath returns a fieldPath locating container within pod.
// Returns an error if the container isn't part of the pod.
func fieldPath(pod *api.Pod, container *api.Container) (string, error) {
	for i := range pod.Spec.Containers {
		here := &pod.Spec.Containers[i]
		if here.Name == container.Name {
			if here.Name == "" {
				return fmt.Sprintf("spec.containers[%d]", i), nil
			} else {
				return fmt.Sprintf("spec.containers{%s}", here.Name), nil
			}
		}
	}
	return "", fmt.Errorf("container %#v not found in pod %#v", container, pod)
}

// containerRef returns an *api.ObjectReference which references the given container within the
// given pod. Returns an error if the reference can't be constructed or the container doesn't
// actually belong to the pod.
// TODO: Pods that came to us by static config or over HTTP have no selfLink set, which makes
// this fail and log an error. Figure out how we want to identify these pods to the rest of the
// system.
func containerRef(pod *api.Pod, container *api.Container) (*api.ObjectReference, error) {
	fieldPath, err := fieldPath(pod, container)
	if err != nil {
		// TODO: figure out intelligent way to refer to containers that we implicitly
		// start (like the pod infra container). This is not a good way, ugh.
		fieldPath = "implicitly required container " + container.Name
	}
	ref, err := api.GetPartialReference(pod, fieldPath)
	if err != nil {
		return nil, err
	}
	return ref, nil
}

// setRef stores a reference to a pod's container, associating it with the given docker id.
func (kl *Kubelet) setRef(id dockertools.DockerID, ref *api.ObjectReference) {
	kl.refLock.Lock()
	defer kl.refLock.Unlock()
	if kl.dockerIDToRef == nil {
		kl.dockerIDToRef = map[dockertools.DockerID]*api.ObjectReference{}
	}
	kl.dockerIDToRef[id] = ref
}

// clearRef forgets the given docker id and its associated container reference.
func (kl *Kubelet) clearRef(id dockertools.DockerID) {
	kl.refLock.Lock()
	defer kl.refLock.Unlock()
	delete(kl.dockerIDToRef, id)
}

// getRef returns the container reference of the given id, or (nil, false) if none is stored.
func (kl *Kubelet) getRef(id dockertools.DockerID) (ref *api.ObjectReference, ok bool) {
	kl.refLock.RLock()
	defer kl.refLock.RUnlock()
	ref, ok = kl.dockerIDToRef[id]
	return ref, ok
}

// Run a single container from a pod. Returns the docker container ID
func (kl *Kubelet) runContainer(pod *api.Pod, container *api.Container, podVolumes volumeMap, netMode, ipcMode string) (id dockertools.DockerID, err error) {
	ref, err := containerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	envVariables, err := kl.makeEnvironmentVariables(pod.Namespace, container)
	if err != nil {
		return "", err
	}
	binds := makeBinds(container, podVolumes)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: dockertools.BuildDockerName(pod.UID, GetPodFullName(pod), container),
		Config: &docker.Config{
			Cmd:          container.Command,
			Env:          envVariables,
			ExposedPorts: exposedPorts,
			Hostname:     pod.Name,
			Image:        container.Image,
			Memory:       container.Resources.Limits.Memory().Value(),
			CPUShares:    milliCPUToShares(container.Resources.Limits.Cpu().MilliValue()),
			WorkingDir:   container.WorkingDir,
		},
	}
	dockerContainer, err := kl.dockerClient.CreateContainer(opts)
	if err != nil {
		if ref != nil {
			kl.recorder.Eventf(ref, "failed", "Failed to create docker container with error: %v", err)
		}
		return "", err
	}
	// Remember this reference so we can report events about this container
	if ref != nil {
		kl.setRef(dockertools.DockerID(dockerContainer.ID), ref)
		kl.recorder.Eventf(ref, "created", "Created with docker id %v", dockerContainer.ID)
	}

	if len(container.TerminationMessagePath) != 0 {
		p := kl.getPodContainerDir(pod.UID, container.Name)
		if err := os.MkdirAll(p, 0750); err != nil {
			glog.Errorf("Error on creating %q: %v", p, err)
		} else {
			containerLogPath := path.Join(p, dockerContainer.ID)
			fs, err := os.Create(containerLogPath)
			if err != nil {
				// TODO: Clean up the previouly created dir? return the error?
				glog.Errorf("Error on creating termination-log file %q: %v", containerLogPath, err)
			} else {
				fs.Close() // Close immediately; we're just doing a `touch` here
				b := fmt.Sprintf("%s:%s", containerLogPath, container.TerminationMessagePath)
				binds = append(binds, b)
			}
		}
	}
	privileged := false
	if capabilities.Get().AllowPrivileged {
		privileged = container.Privileged
	} else if container.Privileged {
		return "", fmt.Errorf("container requested privileged mode, but it is disallowed globally.")
	}

	capAdd, capDrop := makeCapabilites(container.Capabilities.Add, container.Capabilities.Drop)
	hc := &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
		IpcMode:      ipcMode,
		Privileged:   privileged,
		CapAdd:       capAdd,
		CapDrop:      capDrop,
	}
	if pod.Spec.DNSPolicy == api.DNSClusterFirst {
		if err := kl.applyClusterDNS(hc, pod); err != nil {
			return "", err
		}
	}
	err = kl.dockerClient.StartContainer(dockerContainer.ID, hc)
	if err != nil {
		if ref != nil {
			kl.recorder.Eventf(ref, "failed",
				"Failed to start with docker id %v with error: %v", dockerContainer.ID, err)
		}
		return "", err
	}
	if ref != nil {
		kl.recorder.Eventf(ref, "started", "Started with docker id %v", dockerContainer.ID)
	}

	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		handlerErr := kl.runHandler(GetPodFullName(pod), pod.UID, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			kl.killContainerByID(dockerContainer.ID)
			return dockertools.DockerID(""), fmt.Errorf("failed to call event handler: %v", handlerErr)
		}
	}
	return dockertools.DockerID(dockerContainer.ID), err
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

func (kl *Kubelet) applyClusterDNS(hc *docker.HostConfig, pod *api.Pod) error {
	// Get host DNS settings and append them to cluster DNS settings.
	f, err := os.Open("/etc/resolv.conf")
	if err != nil {
		return err
	}
	defer f.Close()

	hostDNS, hostSearch, err := parseResolvConf(f)
	if err != nil {
		return err
	}

	if kl.clusterDNS != nil {
		hc.DNS = append([]string{kl.clusterDNS.String()}, hostDNS...)
	}
	if kl.clusterDomain != "" {
		nsDomain := fmt.Sprintf("%s.%s", pod.Namespace, kl.clusterDomain)
		hc.DNSSearch = append([]string{nsDomain, kl.clusterDomain}, hostSearch...)
	}
	return nil
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
func (kl *Kubelet) killContainer(dockerContainer *docker.APIContainers) error {
	return kl.killContainerByID(dockerContainer.ID)
}

func (kl *Kubelet) killContainerByID(ID string) error {
	glog.V(2).Infof("Killing container with id %q", ID)
	kl.readiness.remove(ID)
	err := kl.dockerClient.StopContainer(ID, 10)

	ref, ok := kl.getRef(dockertools.DockerID(ID))
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

// createPodInfraContainer starts the pod infra container for a pod. Returns the docker container ID of the newly created container.
func (kl *Kubelet) createPodInfraContainer(pod *api.Pod) (dockertools.DockerID, error) {
	var ports []api.ContainerPort
	// Docker only exports ports from the pod infra container.  Let's
	// collect all of the relevant ports and export them.
	for _, container := range pod.Spec.Containers {
		ports = append(ports, container.Ports...)
	}
	container := &api.Container{
		Name:  dockertools.PodInfraContainerName,
		Image: kl.podInfraContainerImage,
		Ports: ports,
	}
	ref, err := containerRef(pod, container)
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
	id, err := kl.runContainer(pod, container, nil, "", "")
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

// Kill all containers in a pod.  Returns the number of containers deleted and an error if one occurs.
func (kl *Kubelet) killContainersInPod(pod *api.Pod, dockerContainers dockertools.DockerContainers) (int, error) {
	podFullName := GetPodFullName(pod)

	count := 0
	errs := make(chan error, len(pod.Spec.Containers))
	wg := sync.WaitGroup{}
	for _, container := range pod.Spec.Containers {
		// TODO: Consider being more aggressive: kill all containers with this pod UID, period.
		if dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, pod.UID, container.Name); found {
			count++
			wg.Add(1)
			go func() {
				defer util.HandleCrash()
				err := kl.killContainer(dockerContainer)
				if err != nil {
					glog.Errorf("Failed to delete container: %v; Skipping pod %q", err, podFullName)
					errs <- err
				}
				wg.Done()
			}()
		}
	}
	wg.Wait()
	close(errs)
	if len(errs) > 0 {
		errList := []error{}
		for err := range errs {
			errList = append(errList, err)
		}
		return -1, fmt.Errorf("failed to delete containers (%v)", errList)
	}
	return count, nil
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
	podFullName := GetPodFullName(pod)
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
	podFullName := GetPodFullName(pod)
	ref, err := containerRef(pod, container)
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

func (kl *Kubelet) computePodContainerChanges(pod *api.Pod, hasMirrorPod bool, containersInPod dockertools.DockerContainers) (podContainerChangesSpec, error) {
	podFullName := GetPodFullName(pod)
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
	podInfraContainerID, found := kl.getPodInfraContainer(podFullName, uid, containersInPod)
	if found {
		glog.V(4).Infof("Found infra pod for %q", podFullName)
		containersToKeep[podInfraContainerID] = -1
		podStatus, err = kl.GetPodStatus(podFullName, uid)
		if err != nil {
			glog.Errorf("Unable to get pod with name %q and uid %q info with error(%v)", podFullName, uid, err)
		}
	} else {
		glog.V(2).Infof("No Infra Container for %q found. All containers will be restarted.", podFullName)
		createPodInfraContainer = true
	}

	for index, container := range pod.Spec.Containers {
		expectedHash := dockertools.HashContainer(&container)
		if dockerContainer, found, hash := containersInPod.FindPodContainer(podFullName, uid, container.Name); found {
			containerID := dockertools.DockerID(dockerContainer.ID)
			glog.V(3).Infof("pod %q container %q exists as %v", podFullName, container.Name, containerID)

			if !createPodInfraContainer {
				// look for changes in the container.
				containerChanged := hash != 0 && hash != expectedHash
				if !containerChanged {
					result, err := kl.probeContainer(pod, podStatus, container, dockerContainer)
					if err != nil {
						// TODO(vmarmol): examine this logic.
						glog.Infof("probe no-error: %s", container.Name)
						containersToKeep[containerID] = index
						continue
					}
					if result == probe.Success {
						glog.Infof("probe success: %s", container.Name)
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

func (kl *Kubelet) syncPod(pod *api.Pod, hasMirrorPod bool, containersInPod dockertools.DockerContainers) error {
	podFullName := GetPodFullName(pod)
	uid := pod.UID
	containerChanges, err := kl.computePodContainerChanges(pod, hasMirrorPod, containersInPod)
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
		if podInfraContainer, found, _ := containersInPod.FindPodContainer(podFullName, uid, dockertools.PodInfraContainerName); found {
			if err := kl.killContainer(podInfraContainer); err != nil {
				glog.Warningf("Failed to kill pod infra container %q: %v", podInfraContainer.ID, err)
			}
		}
		_, err = kl.killContainersInPod(pod, containersInPod)
		if err != nil {
			return err
		}
	} else {
		// Otherwise kill any containers in this pod which are not specified as ones to keep.
		for id, container := range containersInPod {
			_, keep := containerChanges.containersToKeep[id]
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

	if !hasMirrorPod && isStaticPod(pod) {
		glog.V(4).Infof("Creating a mirror pod %q", podFullName)
		if err := kl.mirrorManager.CreateMirrorPod(*pod, kl.hostname); err != nil {
			glog.Errorf("Failed creating a mirror pod %q: %#v", podFullName, err)
		}
	}

	return nil
}

type podContainer struct {
	podFullName   string
	uid           types.UID
	containerName string
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
		_, uid, _, _, err := dockertools.ParseDockerName(running[ix].Name)
		if err != nil {
			continue
		}
		runningSet.Insert(string(uid))
	}
	for name, vol := range currentVolumes {
		if _, ok := desiredVolumes[name]; !ok {
			parts := strings.Split(name, "/")
			if runningSet.Has(parts[0]) {
				glog.Infof("volume %s, still has a container running %s, skipping teardown", name, parts[0])
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
func (kl *Kubelet) SyncPods(allPods []api.Pod, podSyncTypes map[types.UID]metrics.SyncPodType, mirrorPods util.StringSet, start time.Time) error {
	defer func() {
		metrics.SyncPodsLatency.Observe(metrics.SinceInMicroseconds(start))
	}()

	// Remove obsolete entries in podStatus where the pod is no longer considered bound to this node.
	podFullNames := make(map[string]bool)
	for _, pod := range allPods {
		podFullNames[GetPodFullName(&pod)] = true
	}
	kl.removeOrphanedStatuses(podFullNames)

	// Filtered out the rejected pod. They don't have running containers.
	var pods []api.Pod
	for _, pod := range allPods {
		status, ok := kl.getPodStatusFromCache(GetPodFullName(&pod))
		if ok && status.Phase == api.PodFailed {
			continue
		}
		pods = append(pods, pod)
	}

	glog.V(4).Infof("Desired: %#v", pods)
	var err error
	desiredContainers := make(map[podContainer]empty)
	desiredPods := make(map[types.UID]empty)

	dockerContainers, err := kl.dockerCache.RunningContainers()
	if err != nil {
		glog.Errorf("Error listing containers: %#v", dockerContainers)
		return err
	}

	// Check for any containers that need starting
	for ix := range pods {
		pod := &pods[ix]
		podFullName := GetPodFullName(pod)
		uid := pod.UID
		desiredPods[uid] = empty{}

		// Add all containers (including net) to the map.
		desiredContainers[podContainer{podFullName, uid, dockertools.PodInfraContainerName}] = empty{}
		for _, cont := range pod.Spec.Containers {
			desiredContainers[podContainer{podFullName, uid, cont.Name}] = empty{}
		}

		// Run the sync in an async manifest worker.
		kl.podWorkers.UpdatePod(pod, kl.mirrorPods.Has(podFullName), func() {
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
	for ix := range dockerContainers {
		// Don't kill containers that are in the desired pods.
		podFullName, uid, containerName, _, err := dockertools.ParseDockerName(dockerContainers[ix].Names[0])
		_, found := desiredPods[uid]
		if err == nil && found {
			// syncPod() will handle this one.
			continue
		}

		pc := podContainer{podFullName, uid, containerName}
		_, ok := desiredContainers[pc]
		if err != nil || !ok {
			glog.V(1).Infof("Killing unwanted container %+v", pc)
			err = kl.killContainer(dockerContainers[ix])
			if err != nil {
				glog.Errorf("Error killing container %+v: %v", pc, err)
			} else {
				killed = append(killed, dockerContainers[ix].ID)
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
	deleteOrphanedMirrorPods(pods, mirrorPods, kl.mirrorManager)

	return err
}

func updatePods(changed []api.Pod, current []api.Pod) []api.Pod {
	updated := []api.Pod{}
	m := map[types.UID]*api.Pod{}
	for i := range changed {
		pod := &changed[i]
		m[pod.UID] = pod
	}

	for i := range current {
		pod := &current[i]
		if m[pod.UID] != nil {
			updated = append(updated, *m[pod.UID])
			glog.V(4).Infof("pod with UID: %q has a new spec %+v", pod.UID, *m[pod.UID])
		} else {
			updated = append(updated, *pod)
			glog.V(4).Infof("pod with UID: %q stay with the same spec %+v", pod.UID, *pod)
		}
	}

	return updated
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

// getHostPortConflicts detects pods with conflicted host ports and return them.
func getHostPortConflicts(pods []api.Pod) []api.Pod {
	conflicts := []api.Pod{}
	ports := map[int]bool{}
	extract := func(p *api.ContainerPort) int { return p.HostPort }

	// Respect the pod creation order when resolving conflicts.
	sort.Sort(podsByCreationTime(pods))

	for i := range pods {
		pod := &pods[i]
		if errs := validation.AccumulateUniquePorts(pod.Spec.Containers, ports, extract); len(errs) != 0 {
			glog.Errorf("Pod %q: HostPort is already allocated, ignoring: %v", GetPodFullName(pod), errs)
			conflicts = append(conflicts, *pod)
			continue
		}
	}

	return conflicts
}

// handleHostPortConflicts handles pods that conflict on Port.HostPort values.
func (kl *Kubelet) handleHostPortConflicts(pods []api.Pod) {
	conflicts := getHostPortConflicts(pods)
	for _, pod := range conflicts {
		kl.recorder.Eventf(&pod, "hostPortConflict", "Cannot start the pod due to host port conflict.")
		kl.setPodStatusInCache(GetPodFullName(&pod), api.PodStatus{
			Phase:   api.PodFailed,
			Message: "Pod cannot be started due to host port conflict"})
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
			kl.updatePods(u, podSyncTypes)
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
				kl.updatePods(u, podSyncTypes)
			case <-time.After(5 * time.Millisecond):
				// Break the for loop.
				unsyncedPod = false
			}
		}

		pods, mirrorPods, err := kl.GetPods()
		if err != nil {
			glog.Errorf("Failed to get bound pods.")
			return
		}
		if err := handler.SyncPods(pods, podSyncTypes, mirrorPods, start); err != nil {
			glog.Errorf("Couldn't sync containers: %v", err)
		}
	}
}

// Update the Kubelet's internal pods with those provided by the update.
// Records new and updated pods in newPods and updatedPods.
func (kl *Kubelet) updatePods(u PodUpdate, podSyncTypes map[types.UID]metrics.SyncPodType) {
	kl.podLock.Lock()
	defer kl.podLock.Unlock()
	switch u.Op {
	case SET:
		glog.V(3).Infof("SET: Containers changed")
		newPods, newMirrorPods := filterAndCategorizePods(u.Pods)

		// Store the new pods. Don't worry about filtering host ports since those
		// pods will never be looked up.
		existingPods := make(map[types.UID]struct{})
		for i := range kl.pods {
			existingPods[kl.pods[i].UID] = struct{}{}
		}
		for _, pod := range newPods {
			if _, ok := existingPods[pod.UID]; !ok {
				podSyncTypes[pod.UID] = metrics.SyncPodCreate
			}
		}
		// Actually update the pods.
		kl.pods = newPods
		kl.mirrorPods = newMirrorPods

		kl.handleHostPortConflicts(kl.pods)
	case UPDATE:
		glog.V(3).Infof("Update: Containers changed")

		// Store the updated pods. Don't worry about filtering host ports since those
		// pods will never be looked up.
		for i := range u.Pods {
			podSyncTypes[u.Pods[i].UID] = metrics.SyncPodUpdate
		}
		allPods := updatePods(u.Pods, kl.pods)
		kl.pods, kl.mirrorPods = filterAndCategorizePods(allPods)
		kl.handleHostPortConflicts(kl.pods)
	default:
		panic("syncLoop does not support incremental changes")
	}

	// Mark all remaining pods as sync.
	for i := range kl.pods {
		if _, ok := podSyncTypes[kl.pods[i].UID]; !ok {
			podSyncTypes[u.Pods[i].UID] = metrics.SyncPodSync
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
	for cName, cStatus := range podStatus.Info {
		if containerName == cName {
			if cStatus.State.Waiting != nil {
				return "", fmt.Errorf("container %q is in waiting state.", containerName)
			}
			return strings.Replace(podStatus.Info[containerName].ContainerID, dockertools.DockerPrefix, "", 1), nil
		}
	}
	return "", fmt.Errorf("container %q not found in pod", containerName)
}

// GetKubeletContainerLogs returns logs from the container
// The second parameter of GetPodStatus and FindPodContainer methods represents pod UUID, which is allowed to be blank
// TODO: this method is returning logs of random container attempts, when it should be returning the most recent attempt
// or all of them.
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	podStatus, err := kl.GetPodStatus(podFullName, "")
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

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pod map.
func (kl *Kubelet) GetPods() ([]api.Pod, util.StringSet, error) {
	kl.podLock.RLock()
	defer kl.podLock.RUnlock()
	return append([]api.Pod{}, kl.pods...), kl.mirrorPods, nil
}

// GetPodByName provides the first pod that matches namespace and name, as well as whether the node was found.
func (kl *Kubelet) GetPodByName(namespace, name string) (*api.Pod, bool) {
	kl.podLock.RLock()
	defer kl.podLock.RUnlock()
	for i := range kl.pods {
		pod := kl.pods[i]
		if pod.Namespace == namespace && pod.Name == name {
			return &pod, true
		}
	}
	return nil, false
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
		return fmt.Errorf("error getting node %s: %v", kl.hostname, err)
	}
	if node == nil {
		return fmt.Errorf("no node instance returned for %v", kl.hostname)
	}

	// TODO: Post NotReady if we cannot get MachineInfo from cAdvisor. This needs to start
	// cAdvisor locally, e.g. for test-cmd.sh, and in integration test.
	info, err := kl.GetMachineInfo()
	if err != nil {
		glog.Error("error getting machine info: %v", err)
	} else {
		node.Status.NodeInfo.MachineID = info.MachineID
		node.Status.NodeInfo.SystemUUID = info.SystemUUID
		node.Spec.Capacity = api.ResourceList{
			api.ResourceCPU: *resource.NewMilliQuantity(
				int64(info.NumCores*1000),
				resource.DecimalSI),
			api.ResourceMemory: *resource.NewQuantity(
				info.MemoryCapacity,
				resource.BinarySI),
		}
	}

	newCondition := api.NodeCondition{
		Type:          api.NodeReady,
		Status:        api.ConditionFull,
		Reason:        fmt.Sprintf("kubelet is posting ready status"),
		LastProbeTime: util.Now(),
	}
	updated := false
	for i := range node.Status.Conditions {
		if node.Status.Conditions[i].Type == api.NodeReady {
			node.Status.Conditions[i] = newCondition
			updated = true
		}
	}
	if !updated {
		node.Status.Conditions = append(node.Status.Conditions, newCondition)
	}

	_, err = kl.kubeClient.Nodes().Update(node)
	return err
}

// getPhase returns the phase of a pod given its container info.
func getPhase(spec *api.PodSpec, info api.PodInfo) api.PodPhase {
	running := 0
	waiting := 0
	stopped := 0
	failed := 0
	succeeded := 0
	unknown := 0
	for _, container := range spec.Containers {
		if containerStatus, ok := info[container.Name]; ok {
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
func getPodReadyCondition(spec *api.PodSpec, info api.PodInfo) []api.PodCondition {
	ready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionFull,
	}}
	unready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionNone,
	}}
	if info == nil {
		return unready
	}
	for _, container := range spec.Containers {
		if containerStatus, ok := info[container.Name]; ok {
			if !containerStatus.Ready {
				return unready
			}
		} else {
			return unready
		}
	}
	return ready
}

func (kl *Kubelet) GetPodByFullName(podFullName string) (*api.PodSpec, bool) {
	kl.podLock.RLock()
	defer kl.podLock.RUnlock()
	for _, pod := range kl.pods {
		if GetPodFullName(&pod) == podFullName {
			return &pod.Spec, true
		}
	}
	return nil, false
}

// GetPodStatus returns information from Docker about the containers in a pod
func (kl *Kubelet) GetPodStatus(podFullName string, uid types.UID) (api.PodStatus, error) {
	var podStatus api.PodStatus
	spec, found := kl.GetPodByFullName(podFullName)

	if !found {
		return podStatus, fmt.Errorf("Couldn't find spec for pod %s", podFullName)
	}

	// Check to see if the pod has been rejected.
	mappedPodStatus, ok := kl.getPodStatusFromCache(podFullName)
	if ok {
		return mappedPodStatus, nil
	}

	info, err := dockertools.GetDockerPodInfo(kl.dockerClient, *spec, podFullName, uid)

	if err != nil {
		// Error handling
		glog.Infof("Query docker container info for pod %s failed with error (%v)", podFullName, err)
		if strings.Contains(err.Error(), "resource temporarily unavailable") {
			// Leave upstream layer to decide what to do
			return podStatus, err
		} else {
			podStatus.Phase = api.PodPending
			podStatus.Message = fmt.Sprintf("Query docker container info failed with error (%v)", err)
			return podStatus, nil
		}
	}

	// Assume info is ready to process
	podStatus.Phase = getPhase(spec, info)
	podStatus.Info = api.PodInfo{}
	for _, c := range spec.Containers {
		containerStatus := info[c.Name]
		containerStatus.Ready = kl.readiness.IsReady(containerStatus)
		podStatus.Info[c.Name] = containerStatus
	}
	podStatus.Conditions = append(podStatus.Conditions, getPodReadyCondition(spec, podStatus.Info)...)

	netContainerInfo, found := info[dockertools.PodInfraContainerName]
	if found {
		podStatus.PodIP = netContainerInfo.PodIP
	}

	return podStatus, nil
}

// Returns logs of current machine.
func (kl *Kubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	// TODO: whitelist logs we are willing to serve
	kl.logServer.ServeHTTP(w, req)
}

// Run a command in a container, returns the combined stdout, stderr as an array of bytes
func (kl *Kubelet) RunInContainer(podFullName string, uid types.UID, container string, cmd []string) ([]byte, error) {
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
	if kl.runner == nil {
		return fmt.Errorf("no runner specified.")
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return err
	}
	podInfraContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, dockertools.PodInfraContainerName)
	if !found {
		return fmt.Errorf("Unable to find pod infra container for pod %s, uid %v", podFullName, uid)
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

func (kl *Kubelet) GetMachineInfo() (*cadvisorApi.MachineInfo, error) {
	return kl.cadvisor.MachineInfo()
}
