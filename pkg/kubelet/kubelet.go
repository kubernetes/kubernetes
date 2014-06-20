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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/fsouza/go-dockerclient"
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

// The main kubelet implementation
type Kubelet struct {
	Hostname           string
	Client             util.EtcdClient
	DockerClient       DockerInterface
	FileCheckFrequency time.Duration
	SyncFrequency      time.Duration
	HTTPCheckFrequency time.Duration
	pullLock           sync.Mutex
}

// Starts background goroutines. If file, manifest_url, or address are empty,
// they are not watched. Never returns.
func (kl *Kubelet) RunKubelet(file, manifest_url, etcd_servers, address string, port uint) {
	fileChannel := make(chan api.ContainerManifest)
	etcdChannel := make(chan []api.ContainerManifest)
	httpChannel := make(chan api.ContainerManifest)
	serverChannel := make(chan api.ContainerManifest)

	go util.Forever(func() { kl.WatchFile(file, fileChannel) }, 20*time.Second)
	if manifest_url != "" {
		go util.Forever(func() { kl.WatchHTTP(manifest_url, httpChannel) }, 20*time.Second)
	}
	if etcd_servers != "" {
		servers := []string{etcd_servers}
		log.Printf("Creating etcd client pointing to %v", servers)
		kl.Client = etcd.NewClient(servers)
		go util.Forever(func() { kl.SyncAndSetupEtcdWatch(etcdChannel) }, 20*time.Second)
	}
	if address != "" {
		log.Printf("Starting to listen on %s:%d", address, port)
		handler := KubeletServer{
			Kubelet:       kl,
			UpdateChannel: serverChannel,
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
	kl.RunSyncLoop(etcdChannel, fileChannel, serverChannel, httpChannel, kl)
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

// Creates a name which can be reversed to identify both manifest id and container name.
func manifestAndContainerToDockerName(manifest *api.ContainerManifest, container *api.Container) string {
	// Note, manifest.Id could be blank.
	return fmt.Sprintf("%s--%s--%x", escapeDash(container.Name), escapeDash(manifest.Id), rand.Uint32())
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
	if len(parts) > 0 {
		containerName = unescapeDash(parts[0])
	}
	if len(parts) > 1 {
		manifestId = unescapeDash(parts[1])
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
		volumes[volume.MountPath] = struct{}{}
		basePath := "/exports/" + volume.Name + ":" + volume.MountPath
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

func (kl *Kubelet) RunContainer(manifest *api.ContainerManifest, container *api.Container) (name string, err error) {
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

func (kl *Kubelet) extractFromFile(lastData []byte, name string, changeChannel chan<- api.ContainerManifest) ([]byte, error) {
	var file *os.File
	var err error
	if file, err = os.Open(name); err != nil {
		return lastData, err
	}

	return kl.extractFromReader(lastData, file, changeChannel)
}

func (kl *Kubelet) extractFromReader(lastData []byte, reader io.Reader, changeChannel chan<- api.ContainerManifest) ([]byte, error) {
	var manifest api.ContainerManifest
	data, err := ioutil.ReadAll(reader)
	if err != nil {
		log.Printf("Couldn't read file: %v", err)
		return lastData, err
	}
	if err = kl.ExtractYAMLData(data, &manifest); err != nil {
		return lastData, err
	}
	if !bytes.Equal(lastData, data) {
		lastData = data
		// Ok, we have a valid configuration, send to channel for
		// rejiggering.
		changeChannel <- manifest
		return data, nil
	}
	return lastData, nil
}

// Watch a file for changes to the set of pods that should run on this Kubelet
// This function loops forever and is intended to be run as a goroutine
func (kl *Kubelet) WatchFile(file string, changeChannel chan<- api.ContainerManifest) {
	var lastData []byte
	for {
		var err error
		time.Sleep(kl.FileCheckFrequency)
		lastData, err = kl.extractFromFile(lastData, file, changeChannel)
		if err != nil {
			log.Printf("Error polling file: %#v", err)
		}
	}
}

func (kl *Kubelet) extractFromHTTP(lastData []byte, url string, changeChannel chan<- api.ContainerManifest) ([]byte, error) {
	client := &http.Client{}
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return lastData, err
	}
	response, err := client.Do(request)
	if err != nil {
		return lastData, err
	}
	defer response.Body.Close()
	return kl.extractFromReader(lastData, response.Body, changeChannel)
}

// Watch an HTTP endpoint for changes to the set of pods that should run on this Kubelet
// This function runs forever and is intended to be run as a goroutine
func (kl *Kubelet) WatchHTTP(url string, changeChannel chan<- api.ContainerManifest) {
	var lastData []byte
	for {
		var err error
		time.Sleep(kl.HTTPCheckFrequency)
		lastData, err = kl.extractFromHTTP(lastData, url, changeChannel)
		if err != nil {
			log.Printf("Error syncing http: %#v", err)
		}
	}
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

func (kl *Kubelet) getKubeletStateFromEtcd(key string, changeChannel chan<- []api.ContainerManifest) error {
	response, err := kl.Client.Get(key+"/kubelet", true, false)
	if err != nil {
		log.Printf("Error on get on %s: %#v", key, err)
		switch err.(type) {
		case *etcd.EtcdError:
			etcdError := err.(*etcd.EtcdError)
			if etcdError.ErrorCode == 100 {
				return nil
			}
		}
		return err
	}
	manifests, err := kl.ResponseToManifests(response)
	if err != nil {
		log.Printf("Error parsing response (%#v): %s", response, err)
		return err
	}
	log.Printf("Got initial state from etcd: %+v", manifests)
	changeChannel <- manifests
	return nil
}

// Sync with etcd, and set up an etcd watch for new configurations
// The channel to send new configurations across
// This function loops forever and is intended to be run in a go routine.
func (kl *Kubelet) SyncAndSetupEtcdWatch(changeChannel chan<- []api.ContainerManifest) {
	key := "/registry/hosts/" + strings.TrimSpace(kl.Hostname)
	// First fetch the initial configuration (watch only gives changes...)
	for {
		err := kl.getKubeletStateFromEtcd(key, changeChannel)
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
		go kl.WatchEtcd(watchChannel, changeChannel)

		kl.getKubeletStateFromEtcd(key, changeChannel)
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
func (kl *Kubelet) WatchEtcd(watchChannel <-chan *etcd.Response, changeChannel chan<- []api.ContainerManifest) {
	defer util.HandleCrash()
	for {
		watchResponse := <-watchChannel
		log.Printf("Got change: %#v", watchResponse)
		// This means the channel has been closed.
		if watchResponse == nil {
			return
		}
		manifests, err := kl.extractFromEtcd(watchResponse)
		if err != nil {
			log.Printf("Error handling response from etcd: %#v", err)
			continue
		}
		log.Printf("manifests: %#v", manifests)
		// Ok, we have a valid configuration, send to channel for
		// rejiggering.
		changeChannel <- manifests
	}
}

// Sync the configured list of containers (desired state) with the host current state
func (kl *Kubelet) SyncManifests(config []api.ContainerManifest) error {
	log.Printf("Desired:%#v", config)
	var err error
	desired := map[string]bool{}
	for _, manifest := range config {
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
				actualName, err = kl.RunContainer(&manifest, &element)
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
	log.Printf("Existing:\n%#v Desired: %#v", existingContainers, desired)
	for _, container := range existingContainers {
		// This is slightly hacky, but we ignore containers that lack '--' in their name
		// to allow users to manually spin up their own containers if they want.
		if !strings.Contains(container, "--") {
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
// four channels (file, etcd, server, and http) and creates a union of the two. For
// any new change seen, will run a sync against desired state and running state. If
// no changes are seen to the configuration, will synchronize the last known desired
// state every sync_frequency seconds.
// Never returns.
func (kl *Kubelet) RunSyncLoop(etcdChannel <-chan []api.ContainerManifest, fileChannel, serverChannel, httpChannel <-chan api.ContainerManifest, handler SyncHandler) {
	var lastFile, lastEtcd, lastHttp, lastServer []api.ContainerManifest
	for {
		select {
		case manifest := <-fileChannel:
			log.Printf("Got new manifest from file... %v", manifest)
			lastFile = []api.ContainerManifest{manifest}
		case manifests := <-etcdChannel:
			log.Printf("Got new configuration from etcd... %v", manifests)
			lastEtcd = manifests
		case manifest := <-httpChannel:
			log.Printf("Got new manifest from external http... %v", manifest)
			lastHttp = []api.ContainerManifest{manifest}
		case manifest := <-serverChannel:
			log.Printf("Got new manifest from our server... %v", manifest)
			lastServer = []api.ContainerManifest{manifest}
		case <-time.After(kl.SyncFrequency):
		}

		manifests := append([]api.ContainerManifest{}, lastFile...)
		manifests = append(manifests, lastEtcd...)
		manifests = append(manifests, lastHttp...)
		manifests = append(manifests, lastServer...)
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
