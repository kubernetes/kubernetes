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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/envvars"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const defaultChanSize = 1024

// taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
const minShares = 2
const sharesPerCPU = 1024
const milliCPUToCPU = 1000

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {
	SyncPods([]api.BoundPod) error
}

type SourceReadyFn func(source string) bool

type volumeMap map[string]volume.Interface

// New creates a new Kubelet for use in main
func NewMainKubelet(
	hostname string,
	dockerClient dockertools.DockerInterface,
	etcdClient tools.EtcdClient,
	kubeClient *client.Client,
	rootDirectory string,
	networkContainerImage string,
	resyncInterval time.Duration,
	pullQPS float32,
	pullBurst int,
	minimumGCAge time.Duration,
	maxContainerCount int,
	sourceReady SourceReadyFn,
	clusterDomain string,
	clusterDNS net.IP,
	masterServiceNamespace string) (*Kubelet, error) {
	if rootDirectory == "" {
		return nil, fmt.Errorf("invalid root directory %q", rootDirectory)
	}
	if resyncInterval <= 0 {
		return nil, fmt.Errorf("invalid sync frequency %d", resyncInterval)
	}
	if minimumGCAge <= 0 {
		return nil, fmt.Errorf("invalid minimum GC age %d", minimumGCAge)
	}

	serviceStore := cache.NewStore()
	cache.NewReflector(&cache.ListWatch{kubeClient, labels.Everything(), "services", api.NamespaceAll}, &api.Service{}, serviceStore).Run()
	serviceLister := &cache.StoreToServiceLister{serviceStore}

	klet := &Kubelet{
		hostname:               hostname,
		dockerClient:           dockerClient,
		etcdClient:             etcdClient,
		kubeClient:             kubeClient,
		rootDirectory:          rootDirectory,
		resyncInterval:         resyncInterval,
		networkContainerImage:  networkContainerImage,
		podWorkers:             newPodWorkers(),
		dockerIDToRef:          map[dockertools.DockerID]*api.ObjectReference{},
		runner:                 dockertools.NewDockerContainerCommandRunner(dockerClient),
		httpClient:             &http.Client{},
		pullQPS:                pullQPS,
		pullBurst:              pullBurst,
		minimumGCAge:           minimumGCAge,
		maxContainerCount:      maxContainerCount,
		sourceReady:            sourceReady,
		clusterDomain:          clusterDomain,
		clusterDNS:             clusterDNS,
		serviceLister:          serviceLister,
		masterServiceNamespace: masterServiceNamespace,
	}

	if err := klet.setupDataDirs(); err != nil {
		return nil, err
	}

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
	hostname              string
	dockerClient          dockertools.DockerInterface
	kubeClient            *client.Client
	rootDirectory         string
	networkContainerImage string
	podWorkers            *podWorkers
	resyncInterval        time.Duration
	pods                  []api.BoundPod
	sourceReady           SourceReadyFn

	// Needed to report events for containers belonging to deleted/modified pods.
	// Tracks references for reporting events
	dockerIDToRef map[dockertools.DockerID]*api.ObjectReference
	refLock       sync.RWMutex

	// Tracks active pulls.  Needed to protect image garbage collection
	// See: https://github.com/docker/docker/issues/8926 for details
	// TODO: Remove this when (if?) that issue is fixed.
	pullLock sync.RWMutex

	// Optional, no events will be sent without it
	etcdClient tools.EtcdClient
	// Optional, defaults to simple implementaiton
	healthChecker health.HealthChecker
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

	// Optional, no statistics will be available if omitted
	cadvisorClient cadvisorInterface
	cadvisorLock   sync.RWMutex

	// Optional, minimum age required for garbage collection.  If zero, no limit.
	minimumGCAge      time.Duration
	maxContainerCount int

	// If non-empty, use this for container DNS search.
	clusterDomain string

	// If non-nil, use this for container DNS server.
	clusterDNS net.IP

	masterServiceNamespace string
	serviceLister          serviceLister
}

// GetRootDir returns the full path to the directory under which kubelet can
// store data.  These functions are useful to pass interfaces to other modules
// that may need to know where to write data without getting a whole kubelet
// instance.
func (kl *Kubelet) GetRootDir() string {
	return kl.rootDirectory
}

// GetPodsDir returns the full path to the directory under which pod
// directories are created.
func (kl *Kubelet) GetPodsDir() string {
	return path.Join(kl.GetRootDir(), "pods")
}

// GetPodDir returns the full path to the per-pod data directory for the
// specified pod.  This directory may not exist if the pod does not exist.
func (kl *Kubelet) GetPodDir(podUID types.UID) string {
	// Backwards compat.  The "old" stuff should be removed before 1.0
	// release.  The thinking here is this:
	//     !old && !new = use new
	//     !old && new  = use new
	//     old && !new  = use old
	//     old && new   = use new (but warn)
	oldPath := path.Join(kl.GetRootDir(), string(podUID))
	oldExists := dirExists(oldPath)
	newPath := path.Join(kl.GetPodsDir(), string(podUID))
	newExists := dirExists(newPath)
	if oldExists && !newExists {
		return oldPath
	}
	if oldExists {
		glog.Warningf("Data dir for pod %q exists in both old and new form, using new", podUID)
	}
	return newPath
}

// GetPodVolumesDir returns the full path to the per-pod data directory under
// which volumes are created for the specified pod.  This directory may not
// exist if the pod does not exist.
func (kl *Kubelet) GetPodVolumesDir(podUID types.UID) string {
	return path.Join(kl.GetPodDir(podUID), "volumes")
}

// GetPodContainerDir returns the full path to the per-pod data directory under
// which container data is held for the specified pod.  This directory may not
// exist if the pod or container does not exist.
func (kl *Kubelet) GetPodContainerDir(podUID types.UID, ctrName string) string {
	// Backwards compat.  The "old" stuff should be removed before 1.0
	// release.  The thinking here is this:
	//     !old && !new = use new
	//     !old && new  = use new
	//     old && !new  = use old
	//     old && new   = use new (but warn)
	oldPath := path.Join(kl.GetPodDir(podUID), ctrName)
	oldExists := dirExists(oldPath)
	newPath := path.Join(kl.GetPodDir(podUID), "containers", ctrName)
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
	if err := os.MkdirAll(kl.GetRootDir(), 0750); err != nil {
		return fmt.Errorf("error creating root directory: %v", err)
	}
	if err := os.MkdirAll(kl.GetPodsDir(), 0750); err != nil {
		return fmt.Errorf("error creating pods directory: %v", err)
	}
	return nil
}

// Get a list of pods that have data directories.
func (kl *Kubelet) listPodsFromDisk() ([]types.UID, error) {
	podInfos, err := ioutil.ReadDir(kl.GetPodsDir())
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

type ByCreated []*docker.Container

func (a ByCreated) Len() int           { return len(a) }
func (a ByCreated) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByCreated) Less(i, j int) bool { return a[i].Created.After(a[j].Created) }

// TODO: these removals are racy, we should make dockerclient threadsafe across List/Inspect transactions.
func (kl *Kubelet) purgeOldest(ids []string) error {
	dockerData := []*docker.Container{}
	for _, id := range ids {
		data, err := kl.dockerClient.InspectContainer(id)
		if err != nil {
			return err
		}
		if !data.State.Running && (time.Now().Sub(data.State.FinishedAt) > kl.minimumGCAge) {
			dockerData = append(dockerData, data)
		}
	}
	sort.Sort(ByCreated(dockerData))
	if len(dockerData) <= kl.maxContainerCount {
		return nil
	}
	dockerData = dockerData[kl.maxContainerCount:]
	for _, data := range dockerData {
		if err := kl.dockerClient.RemoveContainer(docker.RemoveContainerOptions{ID: data.ID}); err != nil {
			return err
		}
	}

	return nil
}

func (kl *Kubelet) GarbageCollectLoop() {
	util.Forever(func() {
		if err := kl.GarbageCollectContainers(); err != nil {
			glog.Errorf("Garbage collect failed: %v", err)
		}
		if err := kl.GarbageCollectImages(); err != nil {
			glog.Errorf("Garbage collect images failed: %v", err)
		}
	}, time.Minute*1)
}

func (kl *Kubelet) getUnusedImages() ([]string, error) {
	kl.pullLock.Lock()
	defer kl.pullLock.Unlock()
	return dockertools.GetUnusedImages(kl.dockerClient)
}

func (kl *Kubelet) GarbageCollectImages() error {
	images, err := kl.getUnusedImages()
	if err != nil {
		return err
	}
	for ix := range images {
		if err := kl.dockerClient.RemoveImage(images[ix]); err != nil {
			glog.Errorf("Failed to remove image: %q (%v)", images[ix], err)
		}
	}
	return nil
}

// TODO: Also enforce a maximum total number of containers.
func (kl *Kubelet) GarbageCollectContainers() error {
	if kl.maxContainerCount == 0 {
		return nil
	}
	containers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, true)
	if err != nil {
		return err
	}
	uidToIDMap := map[string][]string{}
	for _, container := range containers {
		_, uid, name, _ := dockertools.ParseDockerName(container.ID)
		uidName := string(uid) + "." + name
		uidToIDMap[uidName] = append(uidToIDMap[uidName], container.ID)
	}
	for _, list := range uidToIDMap {
		if len(list) <= kl.maxContainerCount {
			continue
		}
		if err := kl.purgeOldest(list); err != nil {
			return err
		}
	}
	return nil
}

// SetCadvisorClient sets the cadvisor client in a thread-safe way.
func (kl *Kubelet) SetCadvisorClient(c cadvisorInterface) {
	kl.cadvisorLock.Lock()
	defer kl.cadvisorLock.Unlock()
	kl.cadvisorClient = c
}

// GetCadvisorClient gets the cadvisor client.
func (kl *Kubelet) GetCadvisorClient() cadvisorInterface {
	kl.cadvisorLock.RLock()
	defer kl.cadvisorLock.RUnlock()
	return kl.cadvisorClient
}

// Run starts the kubelet reacting to config updates
func (kl *Kubelet) Run(updates <-chan PodUpdate) {
	if kl.logServer == nil {
		kl.logServer = http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/")))
	}
	if kl.dockerPuller == nil {
		kl.dockerPuller = dockertools.NewDockerPuller(kl.dockerClient, kl.pullQPS, kl.pullBurst)
	}
	if kl.healthChecker == nil {
		kl.healthChecker = health.NewHealthChecker()
	}
	kl.syncLoop(updates, kl)
}

// Per-pod workers.
type podWorkers struct {
	lock sync.Mutex

	// Set of pods with existing workers.
	workers util.StringSet
}

func newPodWorkers() *podWorkers {
	return &podWorkers{
		workers: util.NewStringSet(),
	}
}

// Runs a worker for "podFullName" asynchronously with the specified "action".
// If the worker for the "podFullName" is already running, functions as a no-op.
func (self *podWorkers) Run(podFullName string, action func()) {
	self.lock.Lock()
	defer self.lock.Unlock()

	// This worker is already running, let it finish.
	if self.workers.Has(podFullName) {
		return
	}
	self.workers.Insert(podFullName)

	// Run worker async.
	go func() {
		defer util.HandleCrash()
		action()

		self.lock.Lock()
		defer self.lock.Unlock()
		self.workers.Delete(podFullName)
	}()
}

func makeBinds(pod *api.BoundPod, container *api.Container, podVolumes volumeMap) []string {
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

func (kl *Kubelet) mountExternalVolumes(pod *api.BoundPod) (volumeMap, error) {
	podVolumes := make(volumeMap)
	for _, vol := range pod.Spec.Volumes {
		extVolume, err := volume.CreateVolumeBuilder(&vol, pod.Name, kl.rootDirectory)
		if err != nil {
			return nil, err
		}
		// TODO(jonesdl) When the default volume behavior is no longer supported, this case
		// should never occur and an error should be thrown instead.
		if extVolume == nil {
			continue
		}
		podVolumes[vol.Name] = extVolume
		err = extVolume.SetUp()
		if err != nil {
			return nil, err
		}
	}
	return podVolumes, nil
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
func fieldPath(pod *api.BoundPod, container *api.Container) (string, error) {
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
func containerRef(pod *api.BoundPod, container *api.Container) (*api.ObjectReference, error) {
	fieldPath, err := fieldPath(pod, container)
	if err != nil {
		// TODO: figure out intelligent way to refer to containers that we implicitly
		// start (like the network container). This is not a good way, ugh.
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
func (kl *Kubelet) runContainer(pod *api.BoundPod, container *api.Container, podVolumes volumeMap, netMode string) (id dockertools.DockerID, err error) {
	ref, err := containerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
	}

	envVariables, err := kl.makeEnvironmentVariables(pod.Namespace, container)
	if err != nil {
		return "", err
	}
	binds := makeBinds(pod, container, podVolumes)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: dockertools.BuildDockerName(pod.UID, GetPodFullName(pod), container),
		Config: &docker.Config{
			Cmd:          container.Command,
			Env:          envVariables,
			ExposedPorts: exposedPorts,
			Hostname:     pod.Name,
			Image:        container.Image,
			Memory:       container.Memory.Value(),
			CPUShares:    milliCPUToShares(container.CPU.MilliValue()),
			WorkingDir:   container.WorkingDir,
		},
	}
	dockerContainer, err := kl.dockerClient.CreateContainer(opts)
	if err != nil {
		if ref != nil {
			record.Eventf(ref, "failed",
				"Failed to create docker container with error: %v", err)
		}
		return "", err
	}
	// Remember this reference so we can report events about this container
	if ref != nil {
		kl.setRef(dockertools.DockerID(dockerContainer.ID), ref)
		record.Eventf(ref, "created", "Created with docker id %v", dockerContainer.ID)
	}

	if len(container.TerminationMessagePath) != 0 {
		p := kl.GetPodContainerDir(pod.UID, container.Name)
		if err := os.MkdirAll(p, 0750); err != nil {
			glog.Errorf("Error on creating %q: %v", p, err)
		} else {
			containerLogPath := path.Join(p, dockerContainer.ID)
			fs, err := os.Create(containerLogPath)
			if err != nil {
				glog.Errorf("Error on creating termination-log file %q: %v", containerLogPath, err)
			}
			defer fs.Close()
			b := fmt.Sprintf("%s:%s", containerLogPath, container.TerminationMessagePath)
			binds = append(binds, b)
		}
	}
	privileged := false
	if capabilities.Get().AllowPrivileged {
		privileged = container.Privileged
	} else if container.Privileged {
		return "", fmt.Errorf("container requested privileged mode, but it is disallowed globally.")
	}
	hc := &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
		Privileged:   privileged,
	}
	if pod.Spec.DNSPolicy == api.DNSClusterFirst {
		if err := kl.applyClusterDNS(hc, pod); err != nil {
			return "", err
		}
	}
	err = kl.dockerClient.StartContainer(dockerContainer.ID, hc)
	if err != nil {
		if ref != nil {
			record.Eventf(ref, "failed",
				"Failed to start with docker id %v with error: %v", dockerContainer.ID, err)
		}
		return "", err
	}
	if ref != nil {
		record.Eventf(ref, "started", "Started with docker id %v", dockerContainer.ID)
	}

	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		handlerErr := kl.runHandler(GetPodFullName(pod), pod.UID, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			kl.killContainerByID(dockerContainer.ID, "")
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
		return m, fmt.Errorf("Failed to list services when setting up env vars.")
	}

	// project the services in namespace ns onto the master services
	for _, service := range services.Items {
		serviceName := service.Name

		switch service.Namespace {
		// for the case whether the master service namespace is the namespace the pod
		// is in, pod should receive all the pods in the namespace.
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
		// The code is in transition from using etcd+BoundPods to apiserver+Pods.
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

func (kl *Kubelet) applyClusterDNS(hc *docker.HostConfig, pod *api.BoundPod) error {
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
	return kl.killContainerByID(dockerContainer.ID, dockerContainer.Names[0])
}

func (kl *Kubelet) killContainerByID(ID, name string) error {
	glog.V(2).Infof("Killing container with id %q and name %q", ID, name)
	err := kl.dockerClient.StopContainer(ID, 10)
	if len(name) == 0 {
		return err
	}

	ref, ok := kl.getRef(dockertools.DockerID(ID))
	if !ok {
		glog.Warningf("No ref for pod '%v' - '%v'", ID, name)
	} else {
		// TODO: pass reason down here, and state, or move this call up the stack.
		record.Eventf(ref, "killing", "Killing %v - %v", ID, name)
	}

	return err
}

const (
	networkContainerName  = "net"
	NetworkContainerImage = "kubernetes/pause:latest"
)

// createNetworkContainer starts the network container for a pod. Returns the docker container ID of the newly created container.
func (kl *Kubelet) createNetworkContainer(pod *api.BoundPod) (dockertools.DockerID, error) {
	var ports []api.Port
	// Docker only exports ports from the network container.  Let's
	// collect all of the relevant ports and export them.
	for _, container := range pod.Spec.Containers {
		ports = append(ports, container.Ports...)
	}
	container := &api.Container{
		Name:  networkContainerName,
		Image: kl.networkContainerImage,
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
			record.Eventf(ref, "failed", "Failed to inspect image %q", container.Image)
		}
		return "", err
	}
	if !ok {
		if err := kl.pullImage(container.Image, ref); err != nil {
			return "", err
		}
	}
	if ref != nil {
		record.Eventf(ref, "pulled", "Successfully pulled image %q", container.Image)
	}
	return kl.runContainer(pod, container, nil, "")
}

func (kl *Kubelet) pullImage(img string, ref *api.ObjectReference) error {
	kl.pullLock.RLock()
	defer kl.pullLock.RUnlock()
	if err := kl.dockerPuller.Pull(img); err != nil {
		if ref != nil {
			record.Eventf(ref, "failed", "Failed to pull image %q", img)
		}
		return err
	}
	if ref != nil {
		record.Eventf(ref, "pulled", "Successfully pulled image %q", img)
	}
	return nil
}

// Kill all containers in a pod.  Returns the number of containers deleted and an error if one occurs.
func (kl *Kubelet) killContainersInPod(pod *api.BoundPod, dockerContainers dockertools.DockerContainers) (int, error) {
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

func (kl *Kubelet) syncPod(pod *api.BoundPod, dockerContainers dockertools.DockerContainers) error {
	podFullName := GetPodFullName(pod)
	uid := pod.UID
	containersToKeep := make(map[dockertools.DockerID]empty)
	killedContainers := make(map[dockertools.DockerID]empty)
	glog.V(4).Infof("Syncing Pod, podFullName: %q, uid: %q", podFullName, uid)

	// Make data dirs.
	if err := os.Mkdir(kl.GetPodDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	if err := os.Mkdir(kl.GetPodVolumesDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}

	// Make sure we have a network container
	var netID dockertools.DockerID
	if netDockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, networkContainerName); found {
		netID = dockertools.DockerID(netDockerContainer.ID)
	} else {
		glog.V(2).Infof("Network container doesn't exist for pod %q, killing and re-creating the pod", podFullName)
		count, err := kl.killContainersInPod(pod, dockerContainers)
		if err != nil {
			return err
		}
		netID, err = kl.createNetworkContainer(pod)
		if err != nil {
			glog.Errorf("Failed to introspect network container: %v; Skipping pod %q", err, podFullName)
			return err
		}
		if count > 0 {
			// Re-list everything, otherwise we'll think we're ok.
			dockerContainers, err = dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
			if err != nil {
				glog.Errorf("Error listing containers %#v", dockerContainers)
				return err
			}
		}
	}
	containersToKeep[netID] = empty{}

	podVolumes, err := kl.mountExternalVolumes(pod)
	if err != nil {
		glog.Errorf("Unable to mount volumes for pod %q: %v; skipping pod", podFullName, err)
		return err
	}

	podStatus, err := kl.GetPodStatus(podFullName, uid)
	if err != nil {
		glog.Errorf("Unable to get pod with name %q and uid %q info, health checks may be invalid", podFullName, uid)
	}
	netInfo, found := podStatus.Info[networkContainerName]
	if found {
		podStatus.PodIP = netInfo.PodIP
	}

	for _, container := range pod.Spec.Containers {
		expectedHash := dockertools.HashContainer(&container)
		if dockerContainer, found, hash := dockerContainers.FindPodContainer(podFullName, uid, container.Name); found {
			containerID := dockertools.DockerID(dockerContainer.ID)
			glog.V(3).Infof("pod %q container %q exists as %v", podFullName, container.Name, containerID)

			// look for changes in the container.
			if hash == 0 || hash == expectedHash {
				// TODO: This should probably be separated out into a separate goroutine.
				healthy, err := kl.healthy(podFullName, uid, podStatus, container, dockerContainer)
				if err != nil {
					glog.V(1).Infof("health check errored: %v", err)
					containersToKeep[containerID] = empty{}
					continue
				}
				if healthy == health.Healthy {
					containersToKeep[containerID] = empty{}
					continue
				}
				glog.V(1).Infof("pod %q container %q is unhealthy. Container will be killed and re-created.", podFullName, container.Name, healthy)
			} else {
				glog.V(1).Infof("pod %q container %q hash changed (%d vs %d). Container will be killed and re-created.", podFullName, container.Name, hash, expectedHash)
			}
			if err := kl.killContainer(dockerContainer); err != nil {
				glog.V(1).Infof("Failed to kill container %q: %v", dockerContainer.ID, err)
				continue
			}
			killedContainers[containerID] = empty{}

			// Also kill associated network container
			if netContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, networkContainerName); found {
				if err := kl.killContainer(netContainer); err != nil {
					glog.V(1).Infof("Failed to kill network container %q: %v", netContainer.ID, err)
					continue
				}
			}
		}

		// Check RestartPolicy for container
		recentContainers, err := dockertools.GetRecentDockerContainersWithNameAndUUID(kl.dockerClient, podFullName, uid, container.Name)
		if err != nil {
			glog.Errorf("Error listing recent containers with name and uid:%s--%s--%s", podFullName, uid, container.Name)
			// TODO(dawnchen): error handling here?
		}

		if len(recentContainers) > 0 && pod.Spec.RestartPolicy.Always == nil {
			if pod.Spec.RestartPolicy.Never != nil {
				glog.V(3).Infof("Already ran container with name %s--%s--%s, do nothing",
					podFullName, uid, container.Name)
				continue
			}
			if pod.Spec.RestartPolicy.OnFailure != nil {
				// Check the exit code of last run
				if recentContainers[0].State.ExitCode == 0 {
					glog.V(3).Infof("Already successfully ran container with name %s--%s--%s, do nothing",
						podFullName, uid, container.Name)
					continue
				}
			}
		}

		glog.V(3).Infof("Container with name %s--%s--%s doesn't exist, creating %#v", podFullName, uid, container.Name, container)
		ref, err := containerRef(pod, &container)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %v, container %v: '%v'", pod.Name, container.Name, err)
		}
		if !api.IsPullNever(container.ImagePullPolicy) {
			present, err := kl.dockerPuller.IsImagePresent(container.Image)
			latest := dockertools.RequireLatestImage(container.Image)
			if err != nil {
				if ref != nil {
					record.Eventf(ref, "failed", "Failed to inspect image %q", container.Image)
				}
				glog.Errorf("Failed to inspect image %q: %v; skipping pod %q container %q", container.Image, err, podFullName, container.Name)
				continue
			}
			if api.IsPullAlways(container.ImagePullPolicy) ||
				(api.IsPullIfNotPresent(container.ImagePullPolicy) && (!present || latest)) {
				if err := kl.pullImage(container.Image, ref); err != nil {
					continue
				}
			}
		}
		// TODO(dawnchen): Check RestartPolicy.DelaySeconds before restart a container
		containerID, err := kl.runContainer(pod, &container, podVolumes, "container:"+string(netID))
		if err != nil {
			// TODO(bburns) : Perhaps blacklist a container after N failures?
			glog.Errorf("Error running pod %q container %q: %v", podFullName, container.Name, err)
			continue
		}
		containersToKeep[containerID] = empty{}
	}

	// Kill any containers in this pod which were not identified above (guards against duplicates).
	for id, container := range dockerContainers {
		curPodFullName, curUUID, _, _ := dockertools.ParseDockerName(container.Names[0])
		if curPodFullName == podFullName && curUUID == uid {
			// Don't kill containers we want to keep or those we already killed.
			_, keep := containersToKeep[id]
			_, killed := killedContainers[id]
			if !keep && !killed {
				glog.V(1).Infof("Killing unwanted container in pod %q: %+v", curUUID, container)
				err = kl.killContainer(container)
				if err != nil {
					glog.Errorf("Error killing container: %v", err)
				}
			}
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
func getDesiredVolumes(pods []api.BoundPod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Spec.Volumes {
			identifier := path.Join(pod.Name, volume.Name)
			desiredVolumes[identifier] = volume
		}
	}
	return desiredVolumes
}

func (kl *Kubelet) cleanupOrphanedPods(pods []api.BoundPod) error {
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
			if err := os.RemoveAll(kl.GetPodDir(found[i])); err != nil {
				errlist = append(errlist, err)
			}
		}
	}
	return errors.NewAggregate(errlist)
}

// Compares the map of current volumes to the map of desired volumes.
// If an active volume does not have a respective desired volume, clean it up.
func (kl *Kubelet) cleanupOrphanedVolumes(pods []api.BoundPod) error {
	desiredVolumes := getDesiredVolumes(pods)
	currentVolumes := volume.GetCurrentVolumes(kl.rootDirectory)
	for name, vol := range currentVolumes {
		if _, ok := desiredVolumes[name]; !ok {
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
func (kl *Kubelet) SyncPods(pods []api.BoundPod) error {
	glog.V(4).Infof("Desired: %#v", pods)
	var err error
	desiredContainers := make(map[podContainer]empty)
	desiredPods := make(map[types.UID]empty)

	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
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
		desiredContainers[podContainer{podFullName, uid, networkContainerName}] = empty{}
		for _, cont := range pod.Spec.Containers {
			desiredContainers[podContainer{podFullName, uid, cont.Name}] = empty{}
		}

		// Run the sync in an async manifest worker.
		kl.podWorkers.Run(podFullName, func() {
			if err := kl.syncPod(pod, dockerContainers); err != nil {
				glog.Errorf("Error syncing pod, skipping: %v", err)
				record.Eventf(pod, "failedSync", "Error syncing pod, skipping: %v", err)
			}
		})
	}
	// Kill any containers we don't need.
	for _, container := range dockerContainers {
		// Don't kill containers that are in the desired pods.
		podFullName, uid, containerName, _ := dockertools.ParseDockerName(container.Names[0])
		if _, found := desiredPods[uid]; found {
			// syncPod() will handle this one.
			continue
		}
		_, _, podAnnotations := ParsePodFullName(podFullName)
		if source := podAnnotations[ConfigSourceAnnotationKey]; !kl.sourceReady(source) {
			// If the source for this container is not ready, skip deletion, so that we don't accidentally
			// delete containers for sources that haven't reported yet.
			glog.V(4).Infof("Skipping delete of container (%q), source (%s) aren't ready yet.", podFullName, source)
			continue
		}
		pc := podContainer{podFullName, uid, containerName}
		if _, ok := desiredContainers[pc]; !ok {
			glog.V(1).Infof("Killing unwanted container %+v", pc)
			err = kl.killContainer(container)
			if err != nil {
				glog.Errorf("Error killing container %+v: %v", pc, err)
			}
		}
	}

	// Remove any orphaned pods.
	err = kl.cleanupOrphanedPods(pods)
	if err != nil {
		return err
	}

	// Remove any orphaned volumes.
	err = kl.cleanupOrphanedVolumes(pods)
	if err != nil {
		return err
	}

	return err
}

func updateBoundPods(changed []api.BoundPod, current []api.BoundPod) []api.BoundPod {
	updated := []api.BoundPod{}
	m := map[types.UID]*api.BoundPod{}
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

// filterHostPortConflicts removes pods that conflict on Port.HostPort values
func filterHostPortConflicts(pods []api.BoundPod) []api.BoundPod {
	filtered := []api.BoundPod{}
	ports := map[int]bool{}
	extract := func(p *api.Port) int { return p.HostPort }
	for i := range pods {
		pod := &pods[i]
		if errs := validation.AccumulateUniquePorts(pod.Spec.Containers, ports, extract); len(errs) != 0 {
			glog.Warningf("Pod %q: HostPort is already allocated, ignoring: %v", GetPodFullName(pod), errs)
			continue
		}
		filtered = append(filtered, *pod)
	}

	return filtered
}

// syncLoop is the main loop for processing changes. It watches for changes from
// four channels (file, etcd, server, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync_frequency seconds. Never returns.
func (kl *Kubelet) syncLoop(updates <-chan PodUpdate, handler SyncHandler) {
	for {
		select {
		case u := <-updates:
			switch u.Op {
			case SET:
				glog.V(3).Infof("SET: Containers changed")
				kl.pods = u.Pods
				kl.pods = filterHostPortConflicts(kl.pods)
			case UPDATE:
				glog.V(3).Infof("Update: Containers changed")
				kl.pods = updateBoundPods(u.Pods, kl.pods)
				kl.pods = filterHostPortConflicts(kl.pods)

			default:
				panic("syncLoop does not support incremental changes")
			}
		case <-time.After(kl.resyncInterval):
			glog.V(4).Infof("Periodic sync")
		}

		err := handler.SyncPods(kl.pods)
		if err != nil {
			glog.Errorf("Couldn't sync containers: %v", err)
		}
	}
}

// GetKubeletContainerLogs returns logs from the container
// The second parameter of GetPodStatus and FindPodContainer methods represents pod UUID, which is allowed to be blank
// TODO: this method is returning logs of random container attempts, when it should be returning the most recent attempt
// or all of them.
func (kl *Kubelet) GetKubeletContainerLogs(podFullName, containerName, tail string, follow bool, stdout, stderr io.Writer) error {
	_, err := kl.GetPodStatus(podFullName, "")
	if err == dockertools.ErrNoContainersInPod {
		return fmt.Errorf("pod not found (%q)\n", podFullName)
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, true)
	if err != nil {
		return err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, "", containerName)
	if !found {
		return fmt.Errorf("container not found (%q)\n", containerName)
	}
	return dockertools.GetKubeletDockerContainerLogs(kl.dockerClient, dockerContainer.ID, tail, follow, stdout, stderr)
}

// GetBoundPods returns all pods bound to the kubelet and their spec
func (kl *Kubelet) GetBoundPods() ([]api.BoundPod, error) {
	return kl.pods, nil
}

// GetPodFullName provides the first pod that matches namespace and name, or false
// if no such pod can be found.
func (kl *Kubelet) GetPodByName(namespace, name string) (*api.BoundPod, bool) {
	for i := range kl.pods {
		pod := &kl.pods[i]
		if pod.Namespace == namespace && pod.Name == name {
			return pod, true
		}
	}
	return nil, false
}

// GetPodStatus returns information from Docker about the containers in a pod
func (kl *Kubelet) GetPodStatus(podFullName string, uid types.UID) (api.PodStatus, error) {
	var manifest api.PodSpec
	for _, pod := range kl.pods {
		if GetPodFullName(&pod) == podFullName {
			manifest = pod.Spec
			break
		}
	}

	info, err := dockertools.GetDockerPodInfo(kl.dockerClient, manifest, podFullName, uid)

	// TODO(dchen1107): Determine PodPhase here
	var podStatus api.PodStatus
	podStatus.Info = info

	return podStatus, err
}

func (kl *Kubelet) healthy(podFullName string, podUID types.UID, status api.PodStatus, container api.Container, dockerContainer *docker.APIContainers) (health.Status, error) {
	// Give the container 60 seconds to start up.
	if container.LivenessProbe == nil {
		return health.Healthy, nil
	}
	if time.Now().Unix()-dockerContainer.Created < container.LivenessProbe.InitialDelaySeconds {
		return health.Healthy, nil
	}
	if kl.healthChecker == nil {
		return health.Healthy, nil
	}
	return kl.healthChecker.HealthCheck(podFullName, podUID, status, container)
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
	record.Eventf(ref, "starting", "Starting kubelet.")
}
