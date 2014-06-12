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

// Package kubelet is ...
package kubelet

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry"
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
	Client             registry.EtcdClient
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

func (kl *Kubelet) GetContainerID(name string) (string, error) {
	containerList, err := kl.DockerClient.ListContainers(docker.ListContainersOptions{})
	if err != nil {
		return "", err
	}
	for _, value := range containerList {
		if strings.Contains(value.Names[0], name) {
			return value.ID, nil
		}
	}
	return "", fmt.Errorf("couldn't find name: %s", name)
}

// Get a container by name.
// returns the container data from Docker, or an error if one exists.
func (kl *Kubelet) GetContainerByName(name string) (*docker.Container, error) {
	id, err := kl.GetContainerID(name)
	if err != nil {
		return nil, err
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

func (kl *Kubelet) RunContainer(manifest *api.ContainerManifest, container *api.Container) (name string, err error) {
	err = kl.pullImage(container.Image)
	if err != nil {
		return "", err
	}

	name = manifestAndContainerToDockerName(manifest, container)
	envVariables := []string{}
	for _, value := range container.Env {
		envVariables = append(envVariables, fmt.Sprintf("%s=%s", value.Name, value.Value))
	}

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

	exposedPorts := map[docker.Port]struct{}{}
	portBindings := map[docker.Port][]docker.PortBinding{}
	for _, port := range container.Ports {
		interiorPort := port.ContainerPort
		exteriorPort := port.HostPort
		// Some of this port stuff is under-documented voodoo.
		// See http://stackoverflow.com/questions/20428302/binding-a-port-to-a-host-interface-using-the-rest-api
		dockerPort := docker.Port(strconv.Itoa(interiorPort) + "/tcp")
		exposedPorts[dockerPort] = struct{}{}
		portBindings[dockerPort] = []docker.PortBinding{
			docker.PortBinding{
				HostPort: strconv.Itoa(exteriorPort),
			},
		}
	}
	var cmdList []string
	if len(container.Command) > 0 {
		cmdList = strings.Split(container.Command, " ")
	}
	opts := docker.CreateContainerOptions{
		Name: name,
		Config: &docker.Config{
			Image:        container.Image,
			ExposedPorts: exposedPorts,
			Env:          envVariables,
			Volumes:      volumes,
			WorkingDir:   container.WorkingDir,
			Cmd:          cmdList,
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
	id, err := kl.GetContainerID(name)
	if err != nil {
		return err
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

// Watch a file for changes to the set of pods that should run on this Kubelet
// This function loops forever and is intended to be run as a goroutine
func (kl *Kubelet) WatchFile(file string, changeChannel chan<- api.ContainerManifest) {
	var lastData []byte
	for {
		time.Sleep(kl.FileCheckFrequency)
		var manifest api.ContainerManifest
		data, err := ioutil.ReadFile(file)
		if err != nil {
			log.Printf("Couldn't read file: %s : %v", file, err)
			continue
		}
		if err = kl.ExtractYAMLData(data, &manifest); err != nil {
			continue
		}
		if !bytes.Equal(lastData, data) {
			lastData = data
			// Ok, we have a valid configuration, send to channel for
			// rejiggering.
			changeChannel <- manifest
			continue
		}
	}
}

// Watch an HTTP endpoint for changes to the set of pods that should run on this Kubelet
// This function runs forever and is intended to be run as a goroutine
func (kl *Kubelet) WatchHTTP(url string, changeChannel chan<- api.ContainerManifest) {
	var lastData []byte
	client := &http.Client{}
	for {
		time.Sleep(kl.HTTPCheckFrequency)
		var config api.ContainerManifest
		data, err := kl.SyncHTTP(client, url, &config)
		log.Printf("Containers: %#v", config)
		if err != nil {
			log.Printf("Error syncing HTTP: %#v", err)
			continue
		}
		if !bytes.Equal(lastData, data) {
			lastData = data
			changeChannel <- config
			continue
		}
	}
}

// SyncHTTP reads from url a yaml manifest and populates config. Returns the
// raw bytes, if something was read. Returns an error if something goes wrong.
// 'client' is used to execute the request, to allow caching of clients.
func (kl *Kubelet) SyncHTTP(client *http.Client, url string, config *api.ContainerManifest) ([]byte, error) {
	request, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if err = kl.ExtractYAMLData(body, &config); err != nil {
		return body, err
	}
	return body, nil
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

// Watch etcd for changes, receives config objects from the etcd client watch.
// This function loops forever and is intended to be run as a goroutine.
func (kl *Kubelet) WatchEtcd(watchChannel <-chan *etcd.Response, changeChannel chan<- []api.ContainerManifest) {
	defer util.HandleCrash()
	for {
		watchResponse := <-watchChannel
		log.Printf("Got change: %#v", watchResponse)

		// This means the channel has been closed.
		if watchResponse == nil {
			return
		}

		if watchResponse.Node == nil || len(watchResponse.Node.Value) == 0 {
			log.Printf("No nodes field: %#v", watchResponse)
			if watchResponse.Node != nil {
				log.Printf("Node: %#v", watchResponse.Node)
			}
		}
		log.Printf("Got data: %v", watchResponse.Node.Value)
		var manifests []api.ContainerManifest
		if err := kl.ExtractYAMLData([]byte(watchResponse.Node.Value), &manifests); err != nil {
			continue
		}
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
