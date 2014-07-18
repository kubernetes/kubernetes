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
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"github.com/google/cadvisor/client"
	"github.com/google/cadvisor/info"
	"gopkg.in/v1/yaml"
)

const defaultChanSize = 1024

// taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
const minShares = 2
const sharesPerCPU = 1024
const milliCPUToCPU = 1000

// CadvisorInterface is an abstract interface for testability.  It abstracts the interface of "github.com/google/cadvisor/client".Client.
type CadvisorInterface interface {
	ContainerInfo(name string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error)
	MachineInfo() (*info.MachineInfo, error)
}

// New creates a new Kubelet.
func New() *Kubelet {
	return &Kubelet{}
}

// Kubelet is the main kubelet implementation.
type Kubelet struct {
	Hostname           string
	EtcdClient         tools.EtcdClient
	DockerClient       DockerInterface
	DockerPuller       DockerPuller
	CadvisorClient     CadvisorInterface
	FileCheckFrequency time.Duration
	SyncFrequency      time.Duration
	HTTPCheckFrequency time.Duration
	pullLock           sync.Mutex
	HealthChecker      health.HealthChecker
}

type manifestUpdate struct {
	source    string
	manifests []api.ContainerManifest
}

const (
	fileSource       = "file"
	etcdSource       = "etcd"
	httpClientSource = "http_client"
	httpServerSource = "http_server"
)

// RunKubelet starts background goroutines. If config_path, manifest_url, or address are empty,
// they are not watched. Never returns.
func (kl *Kubelet) RunKubelet(dockerEndpoint, configPath, manifestURL, etcdServers, address string, port uint) {
	if kl.CadvisorClient == nil {
		var err error
		kl.CadvisorClient, err = cadvisor.NewClient("http://127.0.0.1:5000")
		if err != nil {
			glog.Errorf("Error on creating cadvisor client: %v", err)
		}
	}
	if kl.DockerPuller == nil {
		kl.DockerPuller = NewDockerPuller(kl.DockerClient)
	}
	updateChannel := make(chan manifestUpdate)
	if configPath != "" {
		glog.Infof("Watching for file configs at %s", configPath)
		go util.Forever(func() {
			kl.WatchFiles(configPath, updateChannel)
		}, kl.FileCheckFrequency)
	}
	if manifestURL != "" {
		glog.Infof("Watching for HTTP configs at %s", manifestURL)
		go util.Forever(func() {
			if err := kl.extractFromHTTP(manifestURL, updateChannel); err != nil {
				glog.Errorf("Error syncing http: %v", err)
			}
		}, kl.HTTPCheckFrequency)
	}
	if etcdServers != "" {
		servers := []string{etcdServers}
		glog.Infof("Watching for etcd configs at %v", servers)
		kl.EtcdClient = etcd.NewClient(servers)
		go util.Forever(func() { kl.SyncAndSetupEtcdWatch(updateChannel) }, 20*time.Second)
	}
	if address != "" {
		glog.Infof("Starting to listen on %s:%d", address, port)
		handler := Server{
			Kubelet:         kl,
			UpdateChannel:   updateChannel,
			DelegateHandler: http.DefaultServeMux,
		}
		s := &http.Server{
			Addr:           net.JoinHostPort(address, strconv.FormatUint(uint64(port), 10)),
			Handler:        &handler,
			ReadTimeout:    10 * time.Second,
			WriteTimeout:   10 * time.Second,
			MaxHeaderBytes: 1 << 20,
		}
		go util.Forever(func() { s.ListenAndServe() }, 0)
	}
	kl.HealthChecker = health.NewHealthChecker()
	kl.syncLoop(updateChannel, kl)
}

// SyncHandler is an interface implemented by Kubelet, for testability
type SyncHandler interface {
	SyncManifests([]api.ContainerManifest) error
}

// LogEvent logs an event to the etcd backend.
func (kl *Kubelet) LogEvent(event *api.Event) error {
	if kl.EtcdClient == nil {
		return fmt.Errorf("no etcd client connection")
	}
	event.Timestamp = time.Now().Unix()
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	var response *etcd.Response
	response, err = kl.EtcdClient.AddChild(fmt.Sprintf("/events/%s", event.Container.Name), string(data), 60*60*48 /* 2 days */)
	// TODO(bburns) : examine response here.
	if err != nil {
		glog.Errorf("Error writing event: %s\n", err)
		if response != nil {
			glog.Infof("Response was: %v\n", *response)
		}
	}
	return err
}

func makeEnvironmentVariables(container *api.Container) []string {
	var result []string
	for _, value := range container.Env {
		result = append(result, fmt.Sprintf("%s=%s", value.Name, value.Value))
	}
	return result
}

func makeVolumesAndBinds(manifestID string, container *api.Container) (map[string]struct{}, []string) {
	volumes := map[string]struct{}{}
	binds := []string{}
	for _, volume := range container.VolumeMounts {
		var basePath string
		if volume.MountType == "HOST" {
			// Host volumes are not Docker volumes and are directly mounted from the host.
			basePath = fmt.Sprintf("%s:%s", volume.MountPath, volume.MountPath)
		} else {
			volumes[volume.MountPath] = struct{}{}
			basePath = fmt.Sprintf("/exports/%s/%s:%s", manifestID, volume.Name, volume.MountPath)
		}
		if volume.ReadOnly {
			basePath += ":ro"
		}
		binds = append(binds, basePath)
	}
	return volumes, binds
}

func makePortsAndBindings(container *api.Container) (map[docker.Port]struct{}, map[docker.Port][]docker.PortBinding) {
	exposedPorts := map[docker.Port]struct{}{}
	portBindings := map[docker.Port][]docker.PortBinding{}
	for _, port := range container.Ports {
		interiorPort := port.ContainerPort
		exteriorPort := port.HostPort
		// Some of this port stuff is under-documented voodoo.
		// See http://stackoverflow.com/questions/20428302/binding-a-port-to-a-host-interface-using-the-rest-api
		var protocol string
		switch strings.ToUpper(port.Protocol) {
		case "UDP":
			protocol = "/udp"
		case "TCP":
			protocol = "/tcp"
		default:
			glog.Infof("Unknown protocol '%s': defaulting to TCP", port.Protocol)
			protocol = "/tcp"
		}
		dockerPort := docker.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}
		portBindings[dockerPort] = []docker.PortBinding{
			{
				HostPort: strconv.Itoa(exteriorPort),
				HostIp:   port.HostIP,
			},
		}
	}
	return exposedPorts, portBindings
}

func milliCPUToShares(milliCPU int) int {
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

// Run a single container from a manifest. Returns the docker container ID
func (kl *Kubelet) runContainer(manifest *api.ContainerManifest, container *api.Container, netMode string) (id DockerID, err error) {
	envVariables := makeEnvironmentVariables(container)
	volumes, binds := makeVolumesAndBinds(manifest.ID, container)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: buildDockerName(manifest, container),
		Config: &docker.Config{
			Cmd:          container.Command,
			Env:          envVariables,
			ExposedPorts: exposedPorts,
			Hostname:     container.Name,
			Image:        container.Image,
			Memory:       int64(container.Memory),
			CpuShares:    int64(milliCPUToShares(container.CPU)),
			Volumes:      volumes,
			WorkingDir:   container.WorkingDir,
		},
	}
	dockerContainer, err := kl.DockerClient.CreateContainer(opts)
	if err != nil {
		return "", err
	}
	err = kl.DockerClient.StartContainer(dockerContainer.ID, &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
	})
	return DockerID(dockerContainer.ID), err
}

// Kill a docker container
func (kl *Kubelet) killContainer(container docker.APIContainers) error {
	err := kl.DockerClient.StopContainer(container.ID, 10)
	manifestID, containerName := parseDockerName(container.Names[0])
	kl.LogEvent(&api.Event{
		Event: "STOP",
		Manifest: &api.ContainerManifest{
			ID: manifestID,
		},
		Container: &api.Container{
			Name: containerName,
		},
	})

	return err
}

func (kl *Kubelet) extractFromFile(name string) (api.ContainerManifest, error) {
	var file *os.File
	var err error
	var manifest api.ContainerManifest

	if file, err = os.Open(name); err != nil {
		return manifest, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		glog.Errorf("Couldn't read from file: %v", err)
		return manifest, err
	}
	if err = kl.ExtractYAMLData(data, &manifest); err != nil {
		return manifest, err
	}
	return manifest, nil
}

func (kl *Kubelet) extractFromDir(name string) ([]api.ContainerManifest, error) {
	var manifests []api.ContainerManifest

	files, err := filepath.Glob(filepath.Join(name, "[^.]*"))
	if err != nil {
		return manifests, err
	}

	sort.Strings(files)

	for _, file := range files {
		manifest, err := kl.extractFromFile(file)
		if err != nil {
			glog.Errorf("Couldn't read from file %s: %v", file, err)
			return manifests, err
		}
		manifests = append(manifests, manifest)
	}
	return manifests, nil
}

// WatchFiles watches a file or direcory of files for changes to the set of pods that
// should run on this Kubelet.
func (kl *Kubelet) WatchFiles(configPath string, updateChannel chan<- manifestUpdate) {
	var err error

	statInfo, err := os.Stat(configPath)
	if err != nil {
		if !os.IsNotExist(err) {
			glog.Errorf("Error accessing path: %v", err)
		}
		return
	}
	if statInfo.Mode().IsDir() {
		manifests, err := kl.extractFromDir(configPath)
		if err != nil {
			glog.Errorf("Error polling dir: %v", err)
			return
		}
		updateChannel <- manifestUpdate{fileSource, manifests}
	} else if statInfo.Mode().IsRegular() {
		manifest, err := kl.extractFromFile(configPath)
		if err != nil {
			glog.Errorf("Error polling file: %v", err)
			return
		}
		updateChannel <- manifestUpdate{fileSource, []api.ContainerManifest{manifest}}
	} else {
		glog.Errorf("Error accessing config - not a directory or file")
		return
	}
}

func (kl *Kubelet) extractFromHTTP(url string, updateChannel chan<- manifestUpdate) error {
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	data, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return fmt.Errorf("zero-length data received from %v", url)
	}

	// First try as if it's a single manifest
	var manifest api.ContainerManifest
	singleErr := yaml.Unmarshal(data, &manifest)
	if singleErr == nil && manifest.Version == "" {
		// If data is a []ContainerManifest, trying to put it into a ContainerManifest
		// will not give an error but also won't set any of the fields.
		// Our docs say that the version field is mandatory, so using that to judge wether
		// this was actually successful.
		singleErr = fmt.Errorf("got blank version field")
	}
	if singleErr == nil {
		updateChannel <- manifestUpdate{httpClientSource, []api.ContainerManifest{manifest}}
		return nil
	}

	// That didn't work, so try an array of manifests.
	var manifests []api.ContainerManifest
	multiErr := yaml.Unmarshal(data, &manifests)
	// We're not sure if the person reading the logs is going to care about the single or
	// multiple manifest unmarshalling attempt, so we need to put both in the logs, as is
	// done at the end. Hence not returning early here.
	if multiErr == nil && len(manifests) > 0 && manifests[0].Version == "" {
		multiErr = fmt.Errorf("got blank version field")
	}
	if multiErr == nil {
		updateChannel <- manifestUpdate{httpClientSource, manifests}
		return nil
	}
	return fmt.Errorf("%v: received '%v', but couldn't parse as a "+
		"single manifest (%v: %+v) or as multiple manifests (%v: %+v).\n",
		url, string(data), singleErr, manifest, multiErr, manifests)
}

// ResponseToManifests takes an etcd Response object, and turns it into a structured list of containers.
// It returns a list of containers, or an error if one occurs.
func (kl *Kubelet) ResponseToManifests(response *etcd.Response) ([]api.ContainerManifest, error) {
	if response.Node == nil || len(response.Node.Value) == 0 {
		return nil, fmt.Errorf("no nodes field: %v", response)
	}
	var manifests []api.ContainerManifest
	err := kl.ExtractYAMLData([]byte(response.Node.Value), &manifests)
	return manifests, err
}

func (kl *Kubelet) getKubeletStateFromEtcd(key string, updateChannel chan<- manifestUpdate) error {
	response, err := kl.EtcdClient.Get(key, true, false)
	if err != nil {
		if tools.IsEtcdNotFound(err) {
			return nil
		}
		glog.Errorf("Error on etcd get of %s: %v", key, err)
		return err
	}
	manifests, err := kl.ResponseToManifests(response)
	if err != nil {
		glog.Errorf("Error parsing response (%v): %s", response, err)
		return err
	}
	glog.Infof("Got state from etcd: %+v", manifests)
	updateChannel <- manifestUpdate{etcdSource, manifests}
	return nil
}

// SyncAndSetupEtcdWatch synchronizes with etcd, and sets up an etcd watch for new configurations.
// The channel to send new configurations across
// This function loops forever and is intended to be run in a go routine.
func (kl *Kubelet) SyncAndSetupEtcdWatch(updateChannel chan<- manifestUpdate) {
	key := path.Join("registry", "hosts", strings.TrimSpace(kl.Hostname), "kubelet")

	// First fetch the initial configuration (watch only gives changes...)
	for {
		err := kl.getKubeletStateFromEtcd(key, updateChannel)
		if err == nil {
			// We got a successful response, etcd is up, set up the watch.
			break
		}
		time.Sleep(30 * time.Second)
	}

	done := make(chan bool)
	go util.Forever(func() { kl.TimeoutWatch(done) }, 0)
	for {
		// The etcd client will close the watch channel when it exits.  So we need
		// to create and service a new one every time.
		watchChannel := make(chan *etcd.Response)
		// We don't push this through Forever because if it dies, we just do it again in 30 secs.
		// anyway.
		go kl.WatchEtcd(watchChannel, updateChannel)

		kl.getKubeletStateFromEtcd(key, updateChannel)
		glog.V(1).Infof("Setting up a watch for configuration changes in etcd for %s", key)
		kl.EtcdClient.Watch(key, 0, true, watchChannel, done)
	}
}

// TimeoutWatch timeout the watch after 30 seconds.
func (kl *Kubelet) TimeoutWatch(done chan bool) {
	t := time.Tick(30 * time.Second)
	for _ = range t {
		done <- true
	}
}

// ExtractYAMLData extracts data from YAML file into a list of containers.
func (kl *Kubelet) ExtractYAMLData(buf []byte, output interface{}) error {
	err := yaml.Unmarshal(buf, output)
	if err != nil {
		glog.Errorf("Couldn't unmarshal configuration: %v", err)
		return err
	}
	return nil
}

func (kl *Kubelet) extractFromEtcd(response *etcd.Response) ([]api.ContainerManifest, error) {
	var manifests []api.ContainerManifest
	if response.Node == nil || len(response.Node.Value) == 0 {
		return manifests, fmt.Errorf("no nodes field: %v", response)
	}
	err := kl.ExtractYAMLData([]byte(response.Node.Value), &manifests)
	return manifests, err
}

// WatchEtcd watches etcd for changes, receives config objects from the etcd client watch.
// This function loops until the watchChannel is closed, and is intended to be run as a goroutine.
func (kl *Kubelet) WatchEtcd(watchChannel <-chan *etcd.Response, updateChannel chan<- manifestUpdate) {
	defer util.HandleCrash()
	for {
		watchResponse := <-watchChannel
		// This means the channel has been closed.
		if watchResponse == nil {
			return
		}
		glog.Infof("Got etcd change: %v", watchResponse)
		manifests, err := kl.extractFromEtcd(watchResponse)
		if err != nil {
			glog.Errorf("Error handling response from etcd: %v", err)
			continue
		}
		glog.Infof("manifests: %+v", manifests)
		// Ok, we have a valid configuration, send to channel for
		// rejiggering.
		updateChannel <- manifestUpdate{etcdSource, manifests}
	}
}

const networkContainerName = "net"

// Create a network container for a manifest. Returns the docker container ID of the newly created container.
func (kl *Kubelet) createNetworkContainer(manifest *api.ContainerManifest) (DockerID, error) {
	var ports []api.Port
	// Docker only exports ports from the network container.  Let's
	// collect all of the relevant ports and export them.
	for _, container := range manifest.Containers {
		ports = append(ports, container.Ports...)
	}
	container := &api.Container{
		Name:    networkContainerName,
		Image:   "busybox",
		Command: []string{"sh", "-c", "rm -f nap && mkfifo nap && exec cat nap"},
		Ports:   ports,
	}
	kl.DockerPuller.Pull("busybox")
	return kl.runContainer(manifest, container, "")
}

func (kl *Kubelet) syncManifest(manifest *api.ContainerManifest, dockerContainers DockerContainers, keepChannel chan<- DockerID) error {
	// Make sure we have a network container
	var netID DockerID
	if networkDockerContainer, found := dockerContainers.FindPodContainer(manifest.ID, networkContainerName); found {
		netID = DockerID(networkDockerContainer.ID)
	} else {
		dockerNetworkID, err := kl.createNetworkContainer(manifest)
		if err != nil {
			glog.Errorf("Failed to introspect network container. (%v)  Skipping manifest %s", err, manifest.ID)
			return err
		}
		netID = dockerNetworkID
	}
	keepChannel <- netID

	for _, container := range manifest.Containers {
		if dockerContainer, found := dockerContainers.FindPodContainer(manifest.ID, container.Name); found {
			containerID := DockerID(dockerContainer.ID)
			glog.Infof("manifest %s container %s exists as %v", manifest.ID, container.Name, containerID)
			glog.V(1).Infof("manifest %s container %s exists as %v", manifest.ID, container.Name, containerID)

			// TODO: This should probably be separated out into a separate goroutine.
			healthy, err := kl.healthy(container, dockerContainer)
			if err != nil {
				glog.V(1).Infof("health check errored: %v", err)
				continue
			}
			if healthy == health.Healthy {
				keepChannel <- containerID
				continue
			}

			glog.V(1).Infof("manifest %s container %s is unhealthy %d.", manifest.ID, container.Name, healthy)
			if err := kl.killContainer(*dockerContainer); err != nil {
				glog.V(1).Infof("Failed to kill container %s: %v", containerID, err)
				continue
			}
		}

		glog.Infof("%+v doesn't exist, creating", container)
		if err := kl.DockerPuller.Pull(container.Image); err != nil {
			glog.Errorf("Failed to create container: %v skipping manifest %s container %s.", err, manifest.ID, container.Name)
			continue
		}
		containerID, err := kl.runContainer(manifest, &container, "container:"+string(netID))
		if err != nil {
			// TODO(bburns) : Perhaps blacklist a container after N failures?
			glog.Errorf("Error running manifest %s container %s: %v", manifest.ID, container.Name, err)
			continue
		}
		keepChannel <- containerID
	}
	return nil
}

type empty struct{}

// SyncManifests synchronizes the configured list of containers (desired state) with the host current state.
func (kl *Kubelet) SyncManifests(config []api.ContainerManifest) error {
	glog.Infof("Desired: %+v", config)
	dockerIdsToKeep := map[DockerID]empty{}
	keepChannel := make(chan DockerID, defaultChanSize)
	waitGroup := sync.WaitGroup{}

	dockerContainers, err := getKubeletDockerContainers(kl.DockerClient)
	if err != nil {
		glog.Errorf("Error listing containers %#v", dockerContainers)
		return err
	}

	// Check for any containers that need starting
	for ix := range config {
		waitGroup.Add(1)
		go func(index int) {
			defer util.HandleCrash()
			defer waitGroup.Done()
			// necessary to dereference by index here b/c otherwise the shared value
			// in the for each is re-used.
			err := kl.syncManifest(&config[index], dockerContainers, keepChannel)
			if err != nil {
				glog.Errorf("Error syncing manifest: %v skipping.", err)
			}
		}(ix)
	}
	ch := make(chan bool)
	go func() {
		for id := range keepChannel {
			dockerIdsToKeep[id] = empty{}
		}
		ch <- true
	}()
	if len(config) > 0 {
		waitGroup.Wait()
	}
	close(keepChannel)
	<-ch

	// Kill any containers we don't need
	existingContainers, err := getKubeletDockerContainers(kl.DockerClient)
	if err != nil {
		glog.Errorf("Error listing containers: %v", err)
		return err
	}
	for id, container := range existingContainers {
		if _, ok := dockerIdsToKeep[id]; !ok {
			glog.Infof("Killing: %s", id)
			err = kl.killContainer(*container)
			if err != nil {
				glog.Errorf("Error killing container: %v", err)
			}
		}
	}
	return err
}

// Check that all Port.HostPort values are unique across all manifests.
func checkHostPortConflicts(allManifests []api.ContainerManifest, newManifest *api.ContainerManifest) []error {
	allErrs := []error{}

	allPorts := map[int]bool{}
	extract := func(p *api.Port) int { return p.HostPort }
	for i := range allManifests {
		manifest := &allManifests[i]
		errs := api.AccumulateUniquePorts(manifest.Containers, allPorts, extract)
		if len(errs) != 0 {
			allErrs = append(allErrs, errs...)
		}
	}
	if errs := api.AccumulateUniquePorts(newManifest.Containers, allPorts, extract); len(errs) != 0 {
		allErrs = append(allErrs, errs...)
	}
	return allErrs
}

// syncLoop is the main loop for processing changes. It watches for changes from
// four channels (file, etcd, server, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync_frequency seconds.
// Never returns.
func (kl *Kubelet) syncLoop(updateChannel <-chan manifestUpdate, handler SyncHandler) {
	last := make(map[string][]api.ContainerManifest)
	for {
		select {
		case u := <-updateChannel:
			glog.Infof("Got configuration from %s: %+v", u.source, u.manifests)
			last[u.source] = u.manifests
		case <-time.After(kl.SyncFrequency):
		}

		allManifests := []api.ContainerManifest{}
		allIds := util.StringSet{}
		for src, srcManifests := range last {
			for i := range srcManifests {
				allErrs := []error{}

				m := &srcManifests[i]
				if allIds.Has(m.ID) {
					allErrs = append(allErrs, api.ValidationError{api.ErrTypeDuplicate, "ContainerManifest.ID", m.ID})
				} else {
					allIds.Insert(m.ID)
				}
				if errs := api.ValidateManifest(m); len(errs) != 0 {
					allErrs = append(allErrs, errs...)
				}
				// Check for host-wide HostPort conflicts.
				if errs := checkHostPortConflicts(allManifests, m); len(errs) != 0 {
					allErrs = append(allErrs, errs...)
				}
				if len(allErrs) > 0 {
					glog.Warningf("Manifest from %s failed validation, ignoring: %v", src, allErrs)
				}
			}
			// TODO(thockin): There's no reason to collect manifests by value.  Don't pessimize.
			allManifests = append(allManifests, srcManifests...)
		}

		err := handler.SyncManifests(allManifests)
		if err != nil {
			glog.Errorf("Couldn't sync containers : %v", err)
		}
	}
}

func getCadvisorContainerInfoRequest(req *info.ContainerInfoRequest) *info.ContainerInfoRequest {
	ret := &info.ContainerInfoRequest{
		NumStats:               req.NumStats,
		CpuUsagePercentiles:    req.CpuUsagePercentiles,
		MemoryUsagePercentages: req.MemoryUsagePercentages,
	}
	return ret
}

// This method takes a container's absolute path and returns the stats for the
// container.  The container's absolute path refers to its hierarchy in the
// cgroup file system. e.g. The root container, which represents the whole
// machine, has path "/"; all docker containers have path "/docker/<docker id>"
func (kl *Kubelet) statsFromContainerPath(containerPath string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	cinfo, err := kl.CadvisorClient.ContainerInfo(containerPath, getCadvisorContainerInfoRequest(req))
	if err != nil {
		return nil, err
	}
	return cinfo, nil
}

// GetPodInfo returns information from Docker about the containers in a pod
func (kl *Kubelet) GetPodInfo(manifestID string) (api.PodInfo, error) {
	return getDockerPodInfo(kl.DockerClient, manifestID)
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(manifestID, containerName string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	if kl.CadvisorClient == nil {
		return nil, nil
	}
	dockerContainers, err := getKubeletDockerContainers(kl.DockerClient)
	if err != nil {
		return nil, err
	}
	dockerContainer, found := dockerContainers.FindPodContainer(manifestID, containerName)
	if !found {
		return nil, errors.New("couldn't find container")
	}
	return kl.statsFromContainerPath(fmt.Sprintf("/docker/%s", dockerContainer.ID), req)
}

// GetRootInfo returns stats (from Cadvisor) of current machine (root container).
func (kl *Kubelet) GetRootInfo(req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	return kl.statsFromContainerPath("/", req)
}

func (kl *Kubelet) GetMachineInfo() (*info.MachineInfo, error) {
	return kl.CadvisorClient.MachineInfo()
}

func (kl *Kubelet) healthy(container api.Container, dockerContainer *docker.APIContainers) (health.Status, error) {
	// Give the container 60 seconds to start up.
	if container.LivenessProbe == nil {
		return health.Healthy, nil
	}
	if time.Now().Unix()-dockerContainer.Created < container.LivenessProbe.InitialDelaySeconds {
		return health.Healthy, nil
	}
	if kl.HealthChecker == nil {
		return health.Healthy, nil
	}
	return kl.HealthChecker.HealthCheck(container)
}
