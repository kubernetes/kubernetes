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
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/info"
	"gopkg.in/v1/yaml"
)

// State, sub object of the Docker JSON data
type State struct {
	Running bool
}

// The structured representation of the JSON object returned by Docker inspect
type DockerContainerData struct {
	state State
}

// Interface for testability
type DockerInterface interface {
	ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error)
	InspectContainer(id string) (*docker.Container, error)
	CreateContainer(docker.CreateContainerOptions) (*docker.Container, error)
	StartContainer(id string, hostConfig *docker.HostConfig) error
	StopContainer(id string, timeout uint) error
}

type CadvisorInterface interface {
	ContainerInfo(name string) (*info.ContainerInfo, error)
	MachineInfo() (*info.MachineInfo, error)
}

// The main kubelet implementation
type Kubelet struct {
	Hostname           string
	Client             util.EtcdClient
	DockerClient       DockerInterface
	CadvisorClient     CadvisorInterface
	FileCheckFrequency time.Duration
	SyncFrequency      time.Duration
	HTTPCheckFrequency time.Duration
	pullLock           sync.Mutex
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

// Starts background goroutines. If config_path, manifest_url, or address are empty,
// they are not watched. Never returns.
func (kl *Kubelet) RunKubelet(config_path, manifest_url, etcd_servers, address string, port uint) {
	updateChannel := make(chan manifestUpdate)
	if config_path != "" {
		go util.Forever(func() { kl.WatchFiles(config_path, updateChannel) }, kl.FileCheckFrequency)
	}
	if manifest_url != "" {
		go util.Forever(func() {
			if err := kl.extractFromHTTP(manifest_url, updateChannel); err != nil {
				log.Printf("Error syncing http: %#v", err)
			}
		}, kl.HTTPCheckFrequency)
	}
	if etcd_servers != "" {
		servers := []string{etcd_servers}
		log.Printf("Creating etcd client pointing to %v", servers)
		kl.Client = etcd.NewClient(servers)
		go util.Forever(func() { kl.SyncAndSetupEtcdWatch(updateChannel) }, 20*time.Second)
	}
	if address != "" {
		log.Printf("Starting to listen on %s:%d", address, port)
		handler := KubeletServer{
			Kubelet:       kl,
			UpdateChannel: updateChannel,
		}
		s := &http.Server{
			// TODO: This is broken if address is an ipv6 address.
			Addr:           fmt.Sprintf("%s:%d", address, port),
			Handler:        &handler,
			ReadTimeout:    10 * time.Second,
			WriteTimeout:   10 * time.Second,
			MaxHeaderBytes: 1 << 20,
		}
		go util.Forever(func() { s.ListenAndServe() }, 0)
	}
	kl.RunSyncLoop(updateChannel, kl)
}

// Interface implemented by Kubelet, for testability
type SyncHandler interface {
	SyncManifests([]api.ContainerManifest) error
}

// Log an event to the etcd backend.
func (kl *Kubelet) LogEvent(event *api.Event) error {
	if kl.Client == nil {
		return fmt.Errorf("no etcd client connection.")
	}
	event.Timestamp = time.Now().Unix()
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}

	var response *etcd.Response
	response, err = kl.Client.AddChild(fmt.Sprintf("/events/%s", event.Container.Name), string(data), 60*60*48 /* 2 days */)
	// TODO(bburns) : examine response here.
	if err != nil {
		log.Printf("Error writing event: %s\n", err)
		if response != nil {
			log.Printf("Response was: %#v\n", *response)
		}
	}
	return err
}

// Does this container exist on this host? Returns true if so, and the name under which the container is running.
// Returns an error if one occurs.
func (kl *Kubelet) ContainerExists(manifest *api.ContainerManifest, container *api.Container) (exists bool, foundName string, err error) {
	containers, err := kl.ListContainers()
	if err != nil {
		return false, "", err
	}
	for _, name := range containers {
		manifestId, containerName := dockerNameToManifestAndContainer(name)
		if manifestId == manifest.Id && containerName == container.Name {
			// TODO(bburns) : This leads to an extra list.  Convert this to use the returned ID and a straight call
			// to inspect
			data, err := kl.GetContainerByName(name)
			return data != nil, name, err
		}
	}
	return false, "", nil
}

// GetContainerID looks at the list of containers on the machine and returns the ID of the container whose name
// matches 'name'.  It returns the name of the container, or empty string, if the container isn't found.
// it returns true if the container is found, false otherwise, and any error that occurs.
func (kl *Kubelet) GetContainerID(name string) (string, bool, error) {
	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return "", false, err
	}
	for _, value := range containerList {
		if strings.Contains(value.Names[0], name) {
			return value.ID, true, nil
		}
	}
	return "", false, nil
}

// Get a container by name.
// returns the container data from Docker, or an error if one exists.
func (kl *Kubelet) GetContainerByName(name string) (*docker.Container, error) {
	id, found, err := kl.GetContainerID(name)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, nil
	}
	return kl.DockerClient.InspectContainer(id)
}

func (kl *Kubelet) ListContainers() ([]string, error) {
	result := []string{}
	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return result, err
	}
	for _, value := range containerList {
		result = append(result, value.Names[0])
	}
	return result, err
}

func (kl *Kubelet) pullImage(image string) error {
	kl.pullLock.Lock()
	defer kl.pullLock.Unlock()
	cmd := exec.Command("docker", "pull", image)
	err := cmd.Start()
	if err != nil {
		return err
	}
	return cmd.Wait()
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
func manifestAndContainerToDockerName(manifest *api.ContainerManifest, container *api.Container) string {
	// Note, manifest.Id could be blank.
	return fmt.Sprintf("%s--%s--%s--%x", containerNamePrefix, escapeDash(container.Name), escapeDash(manifest.Id), rand.Uint32())
}

// Upacks a container name, returning the manifest id and container name we would have used to
// construct the docker name. If the docker name isn't one we created, we may return empty strings.
func dockerNameToManifestAndContainer(name string) (manifestId, containerName string) {
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
		manifestId = unescapeDash(parts[2])
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

func makeVolumesAndBinds(container *api.Container) (map[string]struct{}, []string) {
	volumes := map[string]struct{}{}
	binds := []string{}
	for _, volume := range container.VolumeMounts {
		var basePath string
		if volume.MountType == "HOST" {
			// Host volumes are not Docker volumes and are directly mounted from the host.
			basePath = fmt.Sprintf("%s:%s", volume.MountPath, volume.MountPath)
		} else {
			volumes[volume.MountPath] = struct{}{}
			basePath = fmt.Sprintf("/exports/%s:%s", volume.Name, volume.MountPath)
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
		switch port.Protocol {
		case "udp":
			protocol = "/udp"
		case "tcp":
			protocol = "/tcp"
		default:
			if len(port.Protocol) != 0 {
				log.Printf("Unknown protocol: %s, defaulting to tcp.", port.Protocol)
			}
			protocol = "/tcp"
		}
		dockerPort := docker.Port(strconv.Itoa(interiorPort) + protocol)
		exposedPorts[dockerPort] = struct{}{}
		portBindings[dockerPort] = []docker.PortBinding{
			{
				HostPort: strconv.Itoa(exteriorPort),
			},
		}
	}
	return exposedPorts, portBindings
}

func (kl *Kubelet) RunContainer(manifest *api.ContainerManifest, container *api.Container, netMode string) (name string, err error) {
	name = manifestAndContainerToDockerName(manifest, container)

	envVariables := makeEnvironmentVariables(container)
	volumes, binds := makeVolumesAndBinds(container)
	exposedPorts, portBindings := makePortsAndBindings(container)

	opts := docker.CreateContainerOptions{
		Name: name,
		Config: &docker.Config{
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
	return name, kl.DockerClient.StartContainer(dockerContainer.ID, &docker.HostConfig{
		PortBindings: portBindings,
		Binds:        binds,
		NetworkMode:  netMode,
	})
}

func (kl *Kubelet) KillContainer(name string) error {
	id, found, err := kl.GetContainerID(name)
	if err != nil {
		return err
	}
	if !found {
		// This is weird, but not an error, so yell and then return nil
		log.Printf("Couldn't find container: %s", name)
		return nil
	}
	err = kl.DockerClient.StopContainer(id, 10)
	manifestId, containerName := dockerNameToManifestAndContainer(name)
	kl.LogEvent(&api.Event{
		Event: "STOP",
		Manifest: &api.ContainerManifest{
			Id: manifestId,
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
		log.Printf("Couldn't read from file: %v", err)
		return manifest, err
	}
	if err = kl.ExtractYAMLData(data, &manifest); err != nil {
		return manifest, err
	}
	return manifest, nil
}

func (kl *Kubelet) extractFromDir(name string) ([]api.ContainerManifest, error) {
	var manifests []api.ContainerManifest

	files, err := filepath.Glob(filepath.Join(name, "*"))
	if err != nil {
		return manifests, err
	}

	sort.Strings(files)

	for _, file := range files {
		manifest, err := kl.extractFromFile(file)
		if err != nil {
			log.Printf("Couldn't read from file %s: %v", file, err)
			return manifests, err
		}
		manifests = append(manifests, manifest)
	}
	return manifests, nil
}

func (kl *Kubelet) extractMultipleFromReader(reader io.Reader) ([]api.ContainerManifest, error) {
	var manifests []api.ContainerManifest
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		log.Printf("Couldn't read from reader: %v", err)
		return manifests, err
	}
	if err = kl.ExtractYAMLData(data, &manifests); err != nil {
		return manifests, err
	}
	return manifests, nil
}

func (kl *Kubelet) extractSingleFromReader(reader io.Reader) (api.ContainerManifest, error) {
	var manifest api.ContainerManifest
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		log.Printf("Couldn't read from reader: %v", err)
		return manifest, err
	}
	if err = kl.ExtractYAMLData(data, &manifest); err != nil {
		return manifest, err
	}
	return manifest, nil
}

// Watch a file or direcory of files for changes to the set of pods that
// should run on this Kubelet.
func (kl *Kubelet) WatchFiles(config_path string, updateChannel chan<- manifestUpdate) {
	var err error

	statInfo, err := os.Stat(config_path)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("Error accessing path: %#v", err)
		}
		return
	}
	if statInfo.Mode().IsDir() {
		manifests, err := kl.extractFromDir(config_path)
		if err != nil {
			log.Printf("Error polling dir: %#v", err)
			return
		}
		updateChannel <- manifestUpdate{fileSource, manifests}
	} else if statInfo.Mode().IsRegular() {
		manifest, err := kl.extractFromFile(config_path)
		if err != nil {
			log.Printf("Error polling file: %#v", err)
			return
		}
		updateChannel <- manifestUpdate{fileSource, []api.ContainerManifest{manifest}}
	} else {
		log.Printf("Error accessing config - not a directory or file")
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
	manifest, err := kl.extractSingleFromReader(response.Body)
	if err != nil {
		return err
	}
	updateChannel <- manifestUpdate{httpClientSource, []api.ContainerManifest{manifest}}
	return nil
}

// Take an etcd Response object, and turn it into a structured list of containers
// Return a list of containers, or an error if one occurs.
func (kl *Kubelet) ResponseToManifests(response *etcd.Response) ([]api.ContainerManifest, error) {
	if response.Node == nil || len(response.Node.Value) == 0 {
		return nil, fmt.Errorf("no nodes field: %#v", response)
	}
	var manifests []api.ContainerManifest
	err := kl.ExtractYAMLData([]byte(response.Node.Value), &manifests)
	return manifests, err
}

func (kl *Kubelet) getKubeletStateFromEtcd(key string, updateChannel chan<- manifestUpdate) error {
	response, err := kl.Client.Get(key+"/kubelet", true, false)
	if err != nil {
		if util.IsEtcdNotFound(err) {
			return nil
		}
		log.Printf("Error on etcd get of %s: %#v", key, err)
		return err
	}
	manifests, err := kl.ResponseToManifests(response)
	if err != nil {
		log.Printf("Error parsing response (%#v): %s", response, err)
		return err
	}
	log.Printf("Got state from etcd: %+v", manifests)
	updateChannel <- manifestUpdate{etcdSource, manifests}
	return nil
}

// Sync with etcd, and set up an etcd watch for new configurations
// The channel to send new configurations across
// This function loops forever and is intended to be run in a go routine.
func (kl *Kubelet) SyncAndSetupEtcdWatch(updateChannel chan<- manifestUpdate) {
	key := "/registry/hosts/" + strings.TrimSpace(kl.Hostname)
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
		log.Printf("Setting up a watch for configuration changes in etcd for %s", key)
		kl.Client.Watch(key, 0, true, watchChannel, done)
	}
}

// Timeout the watch after 30 seconds
func (kl *Kubelet) TimeoutWatch(done chan bool) {
	t := time.Tick(30 * time.Second)
	for _ = range t {
		done <- true
	}
}

// Extract data from YAML file into a list of containers.
func (kl *Kubelet) ExtractYAMLData(buf []byte, output interface{}) error {
	err := yaml.Unmarshal(buf, output)
	if err != nil {
		log.Printf("Couldn't unmarshal configuration: %v", err)
		return err
	}
	return nil
}

func (kl *Kubelet) extractFromEtcd(response *etcd.Response) ([]api.ContainerManifest, error) {
	var manifests []api.ContainerManifest
	if response.Node == nil || len(response.Node.Value) == 0 {
		return manifests, fmt.Errorf("no nodes field: %#v", response)
	}
	err := kl.ExtractYAMLData([]byte(response.Node.Value), &manifests)
	return manifests, err
}

// Watch etcd for changes, receives config objects from the etcd client watch.
// This function loops until the watchChannel is closed, and is intended to be run as a goroutine.
func (kl *Kubelet) WatchEtcd(watchChannel <-chan *etcd.Response, updateChannel chan<- manifestUpdate) {
	defer util.HandleCrash()
	for {
		watchResponse := <-watchChannel
		// This means the channel has been closed.
		if watchResponse == nil {
			return
		}
		log.Printf("Got etcd change: %#v", watchResponse)
		manifests, err := kl.extractFromEtcd(watchResponse)
		if err != nil {
			log.Printf("Error handling response from etcd: %#v", err)
			continue
		}
		log.Printf("manifests: %#v", manifests)
		// Ok, we have a valid configuration, send to channel for
		// rejiggering.
		updateChannel <- manifestUpdate{etcdSource, manifests}
	}
}

const networkContainerName = "net"

func (kl *Kubelet) networkContainerExists(manifest *api.ContainerManifest) (string, bool, error) {
	pods, err := kl.ListContainers()
	if err != nil {
		return "", false, err
	}
	for _, name := range pods {
		if strings.Contains(name, containerNamePrefix+"--"+networkContainerName+"--"+manifest.Id+"--") {
			return name, true, nil
		}
	}
	return "", false, nil
}

func (kl *Kubelet) createNetworkContainer(manifest *api.ContainerManifest) (string, error) {
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
	kl.pullImage("busybox")
	return kl.RunContainer(manifest, container, "")
}

// Sync the configured list of containers (desired state) with the host current state
func (kl *Kubelet) SyncManifests(config []api.ContainerManifest) error {
	log.Printf("Desired: %#v", config)
	var err error
	desired := map[string]bool{}
	for _, manifest := range config {
		netName, exists, err := kl.networkContainerExists(&manifest)
		if err != nil {
			log.Printf("Failed to introspect network container. (%#v)  Skipping container %s", err, manifest.Id)
			continue
		}
		if !exists {
			log.Printf("Network container doesn't exist, creating")
			netName, err = kl.createNetworkContainer(&manifest)
			if err != nil {
				log.Printf("Failed to create network container: %#v", err)
			}
			// Docker list prefixes '/' for some reason, so let's do that...
			netName = "/" + netName
		}
		desired[netName] = true
		for _, element := range manifest.Containers {
			var exists bool
			exists, actualName, err := kl.ContainerExists(&manifest, &element)
			if err != nil {
				log.Printf("Error detecting container: %#v skipping.", err)
				continue
			}
			if !exists {
				log.Printf("%#v doesn't exist, creating", element)
				err = kl.pullImage(element.Image)
				if err != nil {
					log.Printf("Error pulling container: %#v", err)
					continue
				}
				// netName has the '/' prefix, so slice it off
				networkContainer := netName[1:]
				actualName, err = kl.RunContainer(&manifest, &element, "container:"+networkContainer)
				// For some reason, list gives back names that start with '/'
				actualName = "/" + actualName

				if err != nil {
					// TODO(bburns) : Perhaps blacklist a container after N failures?
					log.Printf("Error creating container: %#v", err)
					desired[actualName] = true
					continue
				}
			} else {
				log.Printf("%#v exists as %v", element.Name, actualName)
			}
			desired[actualName] = true
		}
	}
	existingContainers, _ := kl.ListContainers()
	log.Printf("Existing: %#v Desired: %#v", existingContainers, desired)
	for _, container := range existingContainers {
		// Skip containers that we didn't create to allow users to manually
		// spin up their own containers if they want.
		if !strings.HasPrefix(container, "/"+containerNamePrefix+"--") {
			continue
		}
		if !desired[container] {
			log.Printf("Killing: %s", container)
			err = kl.KillContainer(container)
			if err != nil {
				log.Printf("Error killing container: %#v", err)
			}
		}
	}
	return err
}

// runSyncLoop is the main loop for processing changes. It watches for changes from
// four channels (file, etcd, server, and http) and creates a union of them. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync_frequency seconds.
// Never returns.
func (kl *Kubelet) RunSyncLoop(updateChannel <-chan manifestUpdate, handler SyncHandler) {
	last := make(map[string][]api.ContainerManifest)
	for {
		select {
		case u := <-updateChannel:
			log.Printf("Got configuration from %s: %#v", u.source, u.manifests)
			last[u.source] = u.manifests
		case <-time.After(kl.SyncFrequency):
		}

		manifests := []api.ContainerManifest{}
		for _, m := range last {
			manifests = append(manifests, m...)
		}

		err := handler.SyncManifests(manifests)
		if err != nil {
			log.Printf("Couldn't sync containers : %#v", err)
		}
	}
}

func (kl *Kubelet) GetContainerInfo(name string) (string, error) {
	info, err := kl.DockerClient.InspectContainer(name)
	if err != nil {
		return "{}", err
	}
	data, err := json.Marshal(info)
	return string(data), err
}

func (kl *Kubelet) GetContainerStats(name string) (*api.ContainerStats, error) {
	if kl.CadvisorClient == nil {
		return nil, nil
	}
	id, found, err := kl.GetContainerID(name)
	if err != nil || !found {
		return nil, err
	}

	info, err := kl.CadvisorClient.ContainerInfo(fmt.Sprintf("/docker/%v", id))

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
