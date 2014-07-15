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
	"math/rand"
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
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
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

// DockerContainerData is the structured representation of the JSON object returned by Docker inspect
type DockerContainerData struct {
	state struct {
		Running bool
	}
}

// DockerInterface is an abstract interface for testability.  It abstracts the interface of docker.Client.
type DockerInterface interface {
	ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error)
	InspectContainer(id string) (*docker.Container, error)
	CreateContainer(docker.CreateContainerOptions) (*docker.Container, error)
	StartContainer(id string, hostConfig *docker.HostConfig) error
	StopContainer(id string, timeout uint) error
	PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error
}

// DockerID is an ID of docker container. It is a type to make it clear when we're working with docker container Ids
type DockerID string

// DockerPuller is an abstract interface for testability.  It abstracts image pull operations.
type DockerPuller interface {
	Pull(image string) error
}

// CadvisorInterface is an abstract interface for testability.  It abstracts the interface of "github.com/google/cadvisor/client".Client.
type CadvisorInterface interface {
	ContainerInfo(name string) (*info.ContainerInfo, error)
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
	HealthChecker      HealthChecker
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
		kl.DockerPuller = kl.MakeDockerPuller()
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
		handler := KubeletServer{
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
	kl.HealthChecker = MakeHealthChecker()
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

// Return a map of docker containers that we manage. The map key is the docker container ID
func (kl *Kubelet) getDockerContainers() (map[DockerID]docker.APIContainers, error) {
	result := map[DockerID]docker.APIContainers{}
	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return nil, err
	}
	for _, value := range containerList {
		// Skip containers that we didn't create to allow users to manually
		// spin up their own containers if they want.
		if !strings.HasPrefix(value.Names[0], "/"+containerNamePrefix+"--") {
			continue
		}
		result[DockerID(value.ID)] = value
	}
	return result, nil
}

// Return Docker's container ID for a manifest's container. Returns an empty string if it doesn't exist.
func (kl *Kubelet) getContainerID(manifest *api.ContainerManifest, container *api.Container) (DockerID, error) {
	dockerContainers, err := kl.getDockerContainers()
	if err != nil {
		return "", err
	}
	for id, dockerContainer := range dockerContainers {
		manifestID, containerName := parseDockerName(dockerContainer.Names[0])
		if manifestID == manifest.ID && containerName == container.Name {
			return DockerID(id), nil
		}
	}
	return "", nil
}

func (kl *Kubelet) getContainer(ID DockerID) (*docker.APIContainers, error) {
	dockerContainers, err := kl.getDockerContainers()
	if err != nil {
		return nil, err
	}
	for dockerID, dockerContainer := range dockerContainers {
		if dockerID == ID {
			return &dockerContainer, nil
		}
	}
	return nil, nil
}

// MakeDockerPuller creates a new instance of the default implementation of DockerPuller.
func (kl *Kubelet) MakeDockerPuller() DockerPuller {
	return dockerPuller{
		client: kl.DockerClient,
	}
}

// dockerPuller is the default implementation of DockerPuller.
type dockerPuller struct {
	client DockerInterface
}

func (p dockerPuller) Pull(image string) error {
	image, tag := parseImageName(image)
	opts := docker.PullImageOptions{
		Repository: image,
		Tag:        tag,
	}
	return p.client.PullImage(opts, docker.AuthConfiguration{})
}

// Converts "-" to "_-_" and "_" to "___" so that we can use "--" to meaningfully separate parts of a docker name.
func escapeDash(in string) (out string) {
	out = strings.Replace(in, "_", "___", -1)
	out = strings.Replace(out, "-", "_-_", -1)
	return
}

// Reverses the transformation of escapeDash.
func unescapeDash(in string) (out string) {
	out = strings.Replace(in, "_-_", "-", -1)
	out = strings.Replace(out, "___", "_", -1)
	return
}

const containerNamePrefix = "k8s"

// Creates a name which can be reversed to identify both manifest id and container name.
func buildDockerName(manifest *api.ContainerManifest, container *api.Container) string {
	// Note, manifest.ID could be blank.
	return fmt.Sprintf("%s--%s--%s--%08x", containerNamePrefix, escapeDash(container.Name), escapeDash(manifest.ID), rand.Uint32())
}

// Upacks a container name, returning the manifest id and container name we would have used to
// construct the docker name. If the docker name isn't one we created, we may return empty strings.
func parseDockerName(name string) (manifestID, containerName string) {
	// For some reason docker appears to be appending '/' to names.
	// If its there, strip it.
	if name[0] == '/' {
		name = name[1:]
	}
	parts := strings.Split(name, "--")
	if len(parts) == 0 || parts[0] != containerNamePrefix {
		return
	}
	if len(parts) > 1 {
		containerName = unescapeDash(parts[1])
	}
	if len(parts) > 2 {
		manifestID = unescapeDash(parts[2])
	}
	return
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

// Parses image name including an tag and returns image name and tag
// TODO: Future Docker versions can parse the tag on daemon side, see:
// https://github.com/dotcloud/docker/issues/6876
// So this can be deprecated at some point.
func parseImageName(image string) (string, string) {
	tag := ""
	parts := strings.SplitN(image, "/", 2)
	repo := ""
	if len(parts) == 2 {
		repo = parts[0]
		image = parts[1]
	}
	parts = strings.SplitN(image, ":", 2)
	if len(parts) == 2 {
		image = parts[0]
		tag = parts[1]
	}
	if repo != "" {
		image = fmt.Sprintf("%s/%s", repo, image)
	}
	return image, tag
}

// Run a single container from a manifest. Returns the docker container ID
func (kl *Kubelet) runContainer(manifest *api.ContainerManifest, container *api.Container, netMode string) (id DockerID, err error) {
	envVariables := makeEnvironmentVariables(container)
	volumes, binds := makeVolumesAndBinds(manifest.ID, container)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: buildDockerName(manifest, container),
		Config: &docker.Config{
			Hostname:     container.Name,
			Image:        container.Image,
			ExposedPorts: exposedPorts,
			Env:          envVariables,
			Volumes:      volumes,
			WorkingDir:   container.WorkingDir,
			Cmd:          container.Command,
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

// Return the docker ID for a manifest's network container. Returns an empty string if it doesn't exist.
func (kl *Kubelet) getNetworkContainerID(manifest *api.ContainerManifest) (DockerID, error) {
	return kl.getContainerID(manifest, &api.Container{Name: networkContainerName})
}

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

func (kl *Kubelet) syncManifest(manifest *api.ContainerManifest, keepChannel chan<- DockerID) error {
	// Make sure we have a network container
	netID, err := kl.getNetworkContainerID(manifest)
	if err != nil {
		glog.Errorf("Failed to introspect network container. (%v)  Skipping manifest %s", err, manifest.ID)
		return err
	}
	if netID == "" {
		glog.Infof("Network container doesn't exist, creating")
		netID, err = kl.createNetworkContainer(manifest)
		if err != nil {
			glog.Errorf("Failed to introspect network container. (%v)  Skipping manifest %s", err, manifest.ID)
			return err
		}
	}
	keepChannel <- netID
	for _, container := range manifest.Containers {
		containerID, err := kl.getContainerID(manifest, &container)
		if err != nil {
			glog.Errorf("Error finding container: %v skipping manifest %s container %s.", err, manifest.ID, container.Name)
			continue
		}
		if containerID == "" {
			glog.Infof("%+v doesn't exist, creating", container)
			kl.DockerPuller.Pull(container.Image)
			if err != nil {
				glog.Errorf("Failed to create container: %v skipping manifest %s container %s.", err, manifest.ID, container.Name)
				continue
			}
			containerID, err = kl.runContainer(manifest, &container, "container:"+string(netID))
			if err != nil {
				// TODO(bburns) : Perhaps blacklist a container after N failures?
				glog.Errorf("Error running manifest %s container %s: %v", manifest.ID, container.Name, err)
				continue
			}
		} else {
			glog.Infof("manifest %s container %s exists as %v", manifest.ID, container.Name, containerID)
			glog.V(1).Infof("manifest %s container %s exists as %v", manifest.ID, container.Name, containerID)
			dockerContainer, err := kl.getContainer(containerID)
			// TODO: This should probably be separated out into a separate goroutine.
			healthy, err := kl.healthy(container, dockerContainer)
			if err != nil {
				glog.V(1).Infof("health check errored: %v", err)
				continue
			}
			if healthy != CheckHealthy {
				glog.V(1).Infof("manifest %s container %s is unhealthy.", manifest.ID, container.Name)
				if err != nil {
					glog.V(1).Infof("Failed to get container info %v, for %s", err, containerID)
					continue
				}
				err = kl.killContainer(*dockerContainer)
				if err != nil {
					glog.V(1).Infof("Failed to kill container %s: %v", containerID, err)
					continue
				}
				containerID, err = kl.runContainer(manifest, &container, "container:"+string(netID))
			}
		}
		keepChannel <- containerID
	}
	return nil
}

type empty struct{}

// SyncManifests synchronizes the configured list of containers (desired state) with the host current state.
func (kl *Kubelet) SyncManifests(config []api.ContainerManifest) error {
	glog.Infof("Desired: %+v", config)
	var err error
	dockerIdsToKeep := map[DockerID]empty{}
	keepChannel := make(chan DockerID, defaultChanSize)
	waitGroup := sync.WaitGroup{}

	// Check for any containers that need starting
	for ix := range config {
		waitGroup.Add(1)
		go func(index int) {
			defer util.HandleCrash()
			defer waitGroup.Done()
			// necessary to dereference by index here b/c otherwise the shared value
			// in the for each is re-used.
			err := kl.syncManifest(&config[index], keepChannel)
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
	existingContainers, err := kl.getDockerContainers()
	if err != nil {
		glog.Errorf("Error listing containers: %v", err)
		return err
	}
	for id, container := range existingContainers {
		if _, ok := dockerIdsToKeep[id]; !ok {
			glog.Infof("Killing: %s", id)
			err = kl.killContainer(container)
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

// getContainerIDFromName looks at the list of containers on the machine and returns the ID of the container whose name
// matches 'name'.  It returns the name of the container, or empty string, if the container isn't found.
// it returns true if the container is found, false otherwise, and any error that occurs.
// TODO: This functions exists to support GetContainerInfo and GetContainerStats
//       It should be removed once those two functions start taking proper pod.IDs
func (kl *Kubelet) getContainerIDFromName(name string) (DockerID, bool, error) {
	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return "", false, err
	}
	for _, value := range containerList {
		if strings.Contains(value.Names[0], name) {
			return DockerID(value.ID), true, nil
		}
	}
	return "", false, nil
}

// GetPodInfo returns docker info for all containers in the pod/manifest.
func (kl *Kubelet) GetPodInfo(podID string) (api.PodInfo, error) {
	info := api.PodInfo{}

	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return nil, err
	}

	for _, value := range containerList {
		manifestID, containerName := parseDockerName(value.Names[0])
		if manifestID != podID {
			continue
		}
		inspectResult, err := kl.DockerClient.InspectContainer(value.ID)
		if err != nil {
			return nil, err
		}
		if inspectResult == nil {
			// Why did we not get an error?
			info[containerName] = docker.Container{}
		} else {
			info[containerName] = *inspectResult
		}
	}
	return info, nil
}

// Returns the docker id corresponding to pod-id-container-name pair.
func (kl *Kubelet) getDockerIDFromPodIDAndContainerName(podID, containerName string) (DockerID, error) {
	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return "", err
	}
	for _, value := range containerList {
		manifestID, cName := parseDockerName(value.Names[0])
		if manifestID == podID && cName == containerName {
			return DockerID(value.ID), nil
		}
	}
	return "", errors.New("couldn't find container")
}

// This method takes a container's absolute path and returns the stats for the
// container.  The container's absolute path refers to its hierarchy in the
// cgroup file system. e.g. The root container, which represents the whole
// machine, has path "/"; all docker containers have path "/docker/<docker id>"
func (kl *Kubelet) statsFromContainerPath(containerPath string) (*api.ContainerStats, error) {
	info, err := kl.CadvisorClient.ContainerInfo(containerPath)

	if err != nil {
		return nil, err
	}
	// When the stats data for the container is not available yet.
	if info.StatsPercentiles == nil {
		return nil, nil
	}

	ret := new(api.ContainerStats)
	ret.MaxMemoryUsage = info.StatsPercentiles.MaxMemoryUsage
	if len(info.StatsPercentiles.CpuUsagePercentiles) > 0 {
		percentiles := make([]api.Percentile, len(info.StatsPercentiles.CpuUsagePercentiles))
		for i, p := range info.StatsPercentiles.CpuUsagePercentiles {
			percentiles[i].Percentage = p.Percentage
			percentiles[i].Value = p.Value
		}
		ret.CpuUsagePercentiles = percentiles
	}
	if len(info.StatsPercentiles.MemoryUsagePercentiles) > 0 {
		percentiles := make([]api.Percentile, len(info.StatsPercentiles.MemoryUsagePercentiles))
		for i, p := range info.StatsPercentiles.MemoryUsagePercentiles {
			percentiles[i].Percentage = p.Percentage
			percentiles[i].Value = p.Value
		}
		ret.MemoryUsagePercentiles = percentiles
	}
	return ret, nil
}

// GetContainerStats returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerStats(podID, containerName string) (*api.ContainerStats, error) {
	if kl.CadvisorClient == nil {
		return nil, nil
	}
	dockerID, err := kl.getDockerIDFromPodIDAndContainerName(podID, containerName)
	if err != nil || len(dockerID) == 0 {
		return nil, err
	}
	return kl.statsFromContainerPath(fmt.Sprintf("/docker/%s", string(dockerID)))
}

// GetMachineStats returns stats (from Cadvisor) of current machine.
func (kl *Kubelet) GetMachineStats() (*api.ContainerStats, error) {
	return kl.statsFromContainerPath("/")
}

func (kl *Kubelet) healthy(container api.Container, dockerContainer *docker.APIContainers) (HealthCheckStatus, error) {
	// Give the container 60 seconds to start up.
	if container.LivenessProbe == nil {
		return CheckHealthy, nil
	}
	if time.Now().Unix()-dockerContainer.Created < container.LivenessProbe.InitialDelaySeconds {
		return CheckHealthy, nil
	}
	if kl.HealthChecker == nil {
		return CheckHealthy, nil
	}
	return kl.HealthChecker.HealthCheck(container)
}
