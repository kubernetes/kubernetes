// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Manager of cAdvisor-monitored containers.
package manager

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/cadvisor/accelerators"
	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/collector"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/docker"
	"github.com/google/cadvisor/container/raw"
	"github.com/google/cadvisor/events"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/machine"
	"github.com/google/cadvisor/utils/oomparser"
	"github.com/google/cadvisor/utils/sysfs"
	"github.com/google/cadvisor/version"
	"github.com/google/cadvisor/watcher"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"k8s.io/klog"
	"k8s.io/utils/clock"
)

var globalHousekeepingInterval = flag.Duration("global_housekeeping_interval", 1*time.Minute, "Interval between global housekeepings")
var updateMachineInfoInterval = flag.Duration("update_machine_info_interval", 5*time.Minute, "Interval between machine info updates.")
var logCadvisorUsage = flag.Bool("log_cadvisor_usage", false, "Whether to log the usage of the cAdvisor container")
var eventStorageAgeLimit = flag.String("event_storage_age_limit", "default=24h", "Max length of time for which to store events (per type). Value is a comma separated list of key values, where the keys are event types (e.g.: creation, oom) or \"default\" and the value is a duration. Default is applied to all non-specified event types")
var eventStorageEventLimit = flag.String("event_storage_event_limit", "default=100000", "Max number of events to store (per type). Value is a comma separated list of key values, where the keys are event types (e.g.: creation, oom) or \"default\" and the value is an integer. Default is applied to all non-specified event types")
var applicationMetricsCountLimit = flag.Int("application_metrics_count_limit", 100, "Max number of application metrics to store (per container)")

// The Manager interface defines operations for starting a manager and getting
// container and machine information.
type Manager interface {
	// Start the manager. Calling other manager methods before this returns
	// may produce undefined behavior.
	Start() error

	// Stops the manager.
	Stop() error

	//  information about a container.
	GetContainerInfo(containerName string, query *info.ContainerInfoRequest) (*info.ContainerInfo, error)

	// Get V2 information about a container.
	// Recursive (subcontainer) requests are best-effort, and may return a partial result alongside an
	// error in the partial failure case.
	GetContainerInfoV2(containerName string, options v2.RequestOptions) (map[string]v2.ContainerInfo, error)

	// Get information about all subcontainers of the specified container (includes self).
	SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error)

	// Gets all the Docker containers. Return is a map from full container name to ContainerInfo.
	AllDockerContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error)

	// Gets information about a specific Docker container. The specified name is within the Docker namespace.
	DockerContainer(dockerName string, query *info.ContainerInfoRequest) (info.ContainerInfo, error)

	// Gets spec for all containers based on request options.
	GetContainerSpec(containerName string, options v2.RequestOptions) (map[string]v2.ContainerSpec, error)

	// Gets summary stats for all containers based on request options.
	GetDerivedStats(containerName string, options v2.RequestOptions) (map[string]v2.DerivedStats, error)

	// Get info for all requested containers based on the request options.
	GetRequestedContainersInfo(containerName string, options v2.RequestOptions) (map[string]*info.ContainerInfo, error)

	// Returns true if the named container exists.
	Exists(containerName string) bool

	// Get information about the machine.
	GetMachineInfo() (*info.MachineInfo, error)

	// Get version information about different components we depend on.
	GetVersionInfo() (*info.VersionInfo, error)

	// GetFsInfoByFsUUID returns the information of the device having the
	// specified filesystem uuid. If no such device with the UUID exists, this
	// function will return the fs.ErrNoSuchDevice error.
	GetFsInfoByFsUUID(uuid string) (v2.FsInfo, error)

	// Get filesystem information for the filesystem that contains the given directory
	GetDirFsInfo(dir string) (v2.FsInfo, error)

	// Get filesystem information for a given label.
	// Returns information for all global filesystems if label is empty.
	GetFsInfo(label string) ([]v2.FsInfo, error)

	// Get ps output for a container.
	GetProcessList(containerName string, options v2.RequestOptions) ([]v2.ProcessInfo, error)

	// Get events streamed through passedChannel that fit the request.
	WatchForEvents(request *events.Request) (*events.EventChannel, error)

	// Get past events that have been detected and that fit the request.
	GetPastEvents(request *events.Request) ([]*info.Event, error)

	CloseEventChannel(watch_id int)

	// Get status information about docker.
	DockerInfo() (info.DockerStatus, error)

	// Get details about interesting docker images.
	DockerImages() ([]info.DockerImage, error)

	// Returns debugging information. Map of lines per category.
	DebugInfo() map[string][]string
}

// New takes a memory storage and returns a new manager.
func New(memoryCache *memory.InMemoryCache, sysfs sysfs.SysFs, maxHousekeepingInterval time.Duration, allowDynamicHousekeeping bool, includedMetricsSet container.MetricSet, collectorHttpClient *http.Client, rawContainerCgroupPathPrefixWhiteList []string) (Manager, error) {
	if memoryCache == nil {
		return nil, fmt.Errorf("manager requires memory storage")
	}

	// Detect the container we are running on.
	selfContainer, err := cgroups.GetOwnCgroupPath("cpu")
	if err != nil {
		return nil, err
	}
	klog.V(2).Infof("cAdvisor running in container: %q", selfContainer)

	context := fs.Context{}

	if err := container.InitializeFSContext(&context); err != nil {
		return nil, err
	}

	fsInfo, err := fs.NewFsInfo(context)
	if err != nil {
		return nil, err
	}

	// If cAdvisor was started with host's rootfs mounted, assume that its running
	// in its own namespaces.
	inHostNamespace := false
	if _, err := os.Stat("/rootfs/proc"); os.IsNotExist(err) {
		inHostNamespace = true
	}

	// Register for new subcontainers.
	eventsChannel := make(chan watcher.ContainerEvent, 16)

	newManager := &manager{
		containers:                            make(map[namespacedContainerName]*containerData),
		quitChannels:                          make([]chan error, 0, 2),
		memoryCache:                           memoryCache,
		fsInfo:                                fsInfo,
		sysFs:                                 sysfs,
		cadvisorContainer:                     selfContainer,
		inHostNamespace:                       inHostNamespace,
		startupTime:                           time.Now(),
		maxHousekeepingInterval:               maxHousekeepingInterval,
		allowDynamicHousekeeping:              allowDynamicHousekeeping,
		includedMetrics:                       includedMetricsSet,
		containerWatchers:                     []watcher.ContainerWatcher{},
		eventsChannel:                         eventsChannel,
		collectorHttpClient:                   collectorHttpClient,
		nvidiaManager:                         &accelerators.NvidiaManager{},
		rawContainerCgroupPathPrefixWhiteList: rawContainerCgroupPathPrefixWhiteList,
	}

	machineInfo, err := machine.Info(sysfs, fsInfo, inHostNamespace)
	if err != nil {
		return nil, err
	}
	newManager.machineInfo = *machineInfo
	klog.V(1).Infof("Machine: %+v", newManager.machineInfo)

	versionInfo, err := getVersionInfo()
	if err != nil {
		return nil, err
	}
	klog.V(1).Infof("Version: %+v", *versionInfo)

	newManager.eventHandler = events.NewEventManager(parseEventsStoragePolicy())
	return newManager, nil
}

// A namespaced container name.
type namespacedContainerName struct {
	// The namespace of the container. Can be empty for the root namespace.
	Namespace string

	// The name of the container in this namespace.
	Name string
}

type manager struct {
	containers               map[namespacedContainerName]*containerData
	containersLock           sync.RWMutex
	memoryCache              *memory.InMemoryCache
	fsInfo                   fs.FsInfo
	sysFs                    sysfs.SysFs
	machineMu                sync.RWMutex // protects machineInfo
	machineInfo              info.MachineInfo
	quitChannels             []chan error
	cadvisorContainer        string
	inHostNamespace          bool
	eventHandler             events.EventManager
	startupTime              time.Time
	maxHousekeepingInterval  time.Duration
	allowDynamicHousekeeping bool
	includedMetrics          container.MetricSet
	containerWatchers        []watcher.ContainerWatcher
	eventsChannel            chan watcher.ContainerEvent
	collectorHttpClient      *http.Client
	nvidiaManager            accelerators.AcceleratorManager
	// List of raw container cgroup path prefix whitelist.
	rawContainerCgroupPathPrefixWhiteList []string
}

// Start the container manager.
func (self *manager) Start() error {
	self.containerWatchers = container.InitializePlugins(self, self.fsInfo, self.includedMetrics)

	err := raw.Register(self, self.fsInfo, self.includedMetrics, self.rawContainerCgroupPathPrefixWhiteList)
	if err != nil {
		klog.Errorf("Registration of the raw container factory failed: %v", err)
	}

	rawWatcher, err := raw.NewRawContainerWatcher()
	if err != nil {
		return err
	}
	self.containerWatchers = append(self.containerWatchers, rawWatcher)

	// Watch for OOMs.
	err = self.watchForNewOoms()
	if err != nil {
		klog.Warningf("Could not configure a source for OOM detection, disabling OOM events: %v", err)
	}

	// If there are no factories, don't start any housekeeping and serve the information we do have.
	if !container.HasFactories() {
		return nil
	}

	// Setup collection of nvidia GPU metrics if any of them are attached to the machine.
	self.nvidiaManager.Setup()

	// Create root and then recover all containers.
	err = self.createContainer("/", watcher.Raw)
	if err != nil {
		return err
	}
	klog.V(2).Infof("Starting recovery of all containers")
	err = self.detectSubcontainers("/")
	if err != nil {
		return err
	}
	klog.V(2).Infof("Recovery completed")

	// Watch for new container.
	quitWatcher := make(chan error)
	err = self.watchForNewContainers(quitWatcher)
	if err != nil {
		return err
	}
	self.quitChannels = append(self.quitChannels, quitWatcher)

	// Look for new containers in the main housekeeping thread.
	quitGlobalHousekeeping := make(chan error)
	self.quitChannels = append(self.quitChannels, quitGlobalHousekeeping)
	go self.globalHousekeeping(quitGlobalHousekeeping)

	quitUpdateMachineInfo := make(chan error)
	self.quitChannels = append(self.quitChannels, quitUpdateMachineInfo)
	go self.updateMachineInfo(quitUpdateMachineInfo)

	return nil
}

func (self *manager) Stop() error {
	defer self.nvidiaManager.Destroy()
	// Stop and wait on all quit channels.
	for i, c := range self.quitChannels {
		// Send the exit signal and wait on the thread to exit (by closing the channel).
		c <- nil
		err := <-c
		if err != nil {
			// Remove the channels that quit successfully.
			self.quitChannels = self.quitChannels[i:]
			return err
		}
	}
	self.quitChannels = make([]chan error, 0, 2)
	return nil
}

func (self *manager) updateMachineInfo(quit chan error) {
	ticker := time.NewTicker(*updateMachineInfoInterval)
	for {
		select {
		case <-ticker.C:
			info, err := machine.Info(self.sysFs, self.fsInfo, self.inHostNamespace)
			if err != nil {
				klog.Errorf("Could not get machine info: %v", err)
				break
			}
			self.machineMu.Lock()
			self.machineInfo = *info
			self.machineMu.Unlock()
			klog.V(5).Infof("Update machine info: %+v", *info)
		case <-quit:
			ticker.Stop()
			quit <- nil
			return
		}
	}
}

func (self *manager) globalHousekeeping(quit chan error) {
	// Long housekeeping is either 100ms or half of the housekeeping interval.
	longHousekeeping := 100 * time.Millisecond
	if *globalHousekeepingInterval/2 < longHousekeeping {
		longHousekeeping = *globalHousekeepingInterval / 2
	}

	ticker := time.Tick(*globalHousekeepingInterval)
	for {
		select {
		case t := <-ticker:
			start := time.Now()

			// Check for new containers.
			err := self.detectSubcontainers("/")
			if err != nil {
				klog.Errorf("Failed to detect containers: %s", err)
			}

			// Log if housekeeping took too long.
			duration := time.Since(start)
			if duration >= longHousekeeping {
				klog.V(3).Infof("Global Housekeeping(%d) took %s", t.Unix(), duration)
			}
		case <-quit:
			// Quit if asked to do so.
			quit <- nil
			klog.Infof("Exiting global housekeeping thread")
			return
		}
	}
}

func (self *manager) getContainerData(containerName string) (*containerData, error) {
	var cont *containerData
	var ok bool
	func() {
		self.containersLock.RLock()
		defer self.containersLock.RUnlock()

		// Ensure we have the container.
		cont, ok = self.containers[namespacedContainerName{
			Name: containerName,
		}]
	}()
	if !ok {
		return nil, fmt.Errorf("unknown container %q", containerName)
	}
	return cont, nil
}

func (self *manager) GetDerivedStats(containerName string, options v2.RequestOptions) (map[string]v2.DerivedStats, error) {
	conts, err := self.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	var errs partialFailure
	stats := make(map[string]v2.DerivedStats)
	for name, cont := range conts {
		d, err := cont.DerivedStats()
		if err != nil {
			errs.append(name, "DerivedStats", err)
		}
		stats[name] = d
	}
	return stats, errs.OrNil()
}

func (self *manager) GetContainerSpec(containerName string, options v2.RequestOptions) (map[string]v2.ContainerSpec, error) {
	conts, err := self.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	var errs partialFailure
	specs := make(map[string]v2.ContainerSpec)
	for name, cont := range conts {
		cinfo, err := cont.GetInfo(false)
		if err != nil {
			errs.append(name, "GetInfo", err)
		}
		spec := self.getV2Spec(cinfo)
		specs[name] = spec
	}
	return specs, errs.OrNil()
}

// Get V2 container spec from v1 container info.
func (self *manager) getV2Spec(cinfo *containerInfo) v2.ContainerSpec {
	spec := self.getAdjustedSpec(cinfo)
	return v2.ContainerSpecFromV1(&spec, cinfo.Aliases, cinfo.Namespace)
}

func (self *manager) getAdjustedSpec(cinfo *containerInfo) info.ContainerSpec {
	spec := cinfo.Spec

	// Set default value to an actual value
	if spec.HasMemory {
		// Memory.Limit is 0 means there's no limit
		if spec.Memory.Limit == 0 {
			self.machineMu.RLock()
			spec.Memory.Limit = uint64(self.machineInfo.MemoryCapacity)
			self.machineMu.RUnlock()
		}
	}
	return spec
}

func (self *manager) GetContainerInfo(containerName string, query *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	cont, err := self.getContainerData(containerName)
	if err != nil {
		return nil, err
	}
	return self.containerDataToContainerInfo(cont, query)
}

func (self *manager) GetContainerInfoV2(containerName string, options v2.RequestOptions) (map[string]v2.ContainerInfo, error) {
	containers, err := self.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}

	var errs partialFailure
	var nilTime time.Time // Ignored.

	infos := make(map[string]v2.ContainerInfo, len(containers))
	for name, container := range containers {
		result := v2.ContainerInfo{}
		cinfo, err := container.GetInfo(false)
		if err != nil {
			errs.append(name, "GetInfo", err)
			infos[name] = result
			continue
		}
		result.Spec = self.getV2Spec(cinfo)

		stats, err := self.memoryCache.RecentStats(name, nilTime, nilTime, options.Count)
		if err != nil {
			errs.append(name, "RecentStats", err)
			infos[name] = result
			continue
		}

		result.Stats = v2.ContainerStatsFromV1(containerName, &cinfo.Spec, stats)
		infos[name] = result
	}

	return infos, errs.OrNil()
}

func (self *manager) containerDataToContainerInfo(cont *containerData, query *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	// Get the info from the container.
	cinfo, err := cont.GetInfo(true)
	if err != nil {
		return nil, err
	}

	stats, err := self.memoryCache.RecentStats(cinfo.Name, query.Start, query.End, query.NumStats)
	if err != nil {
		return nil, err
	}

	// Make a copy of the info for the user.
	ret := &info.ContainerInfo{
		ContainerReference: cinfo.ContainerReference,
		Subcontainers:      cinfo.Subcontainers,
		Spec:               self.getAdjustedSpec(cinfo),
		Stats:              stats,
	}
	return ret, nil
}

func (self *manager) getContainer(containerName string) (*containerData, error) {
	self.containersLock.RLock()
	defer self.containersLock.RUnlock()
	cont, ok := self.containers[namespacedContainerName{Name: containerName}]
	if !ok {
		return nil, fmt.Errorf("unknown container %q", containerName)
	}
	return cont, nil
}

func (self *manager) getSubcontainers(containerName string) map[string]*containerData {
	self.containersLock.RLock()
	defer self.containersLock.RUnlock()
	containersMap := make(map[string]*containerData, len(self.containers))

	// Get all the unique subcontainers of the specified container
	matchedName := path.Join(containerName, "/")
	for i := range self.containers {
		name := self.containers[i].info.Name
		if name == containerName || strings.HasPrefix(name, matchedName) {
			containersMap[self.containers[i].info.Name] = self.containers[i]
		}
	}
	return containersMap
}

func (self *manager) SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	containersMap := self.getSubcontainers(containerName)

	containers := make([]*containerData, 0, len(containersMap))
	for _, cont := range containersMap {
		containers = append(containers, cont)
	}
	return self.containerDataSliceToContainerInfoSlice(containers, query)
}

func (self *manager) getAllDockerContainers() map[string]*containerData {
	self.containersLock.RLock()
	defer self.containersLock.RUnlock()
	containers := make(map[string]*containerData, len(self.containers))

	// Get containers in the Docker namespace.
	for name, cont := range self.containers {
		if name.Namespace == docker.DockerNamespace {
			containers[cont.info.Name] = cont
		}
	}
	return containers
}

func (self *manager) AllDockerContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error) {
	containers := self.getAllDockerContainers()

	output := make(map[string]info.ContainerInfo, len(containers))
	for name, cont := range containers {
		inf, err := self.containerDataToContainerInfo(cont, query)
		if err != nil {
			// Ignore the error because of race condition and return best-effort result.
			if err == memory.ErrDataNotFound {
				klog.Warningf("Error getting data for container %s because of race condition", name)
				continue
			}
			return nil, err
		}
		output[name] = *inf
	}
	return output, nil
}

func (self *manager) getDockerContainer(containerName string) (*containerData, error) {
	self.containersLock.RLock()
	defer self.containersLock.RUnlock()

	// Check for the container in the Docker container namespace.
	cont, ok := self.containers[namespacedContainerName{
		Namespace: docker.DockerNamespace,
		Name:      containerName,
	}]

	// Look for container by short prefix name if no exact match found.
	if !ok {
		for contName, c := range self.containers {
			if contName.Namespace == docker.DockerNamespace && strings.HasPrefix(contName.Name, containerName) {
				if cont == nil {
					cont = c
				} else {
					return nil, fmt.Errorf("unable to find container. Container %q is not unique", containerName)
				}
			}
		}

		if cont == nil {
			return nil, fmt.Errorf("unable to find Docker container %q", containerName)
		}
	}

	return cont, nil
}

func (self *manager) DockerContainer(containerName string, query *info.ContainerInfoRequest) (info.ContainerInfo, error) {
	container, err := self.getDockerContainer(containerName)
	if err != nil {
		return info.ContainerInfo{}, err
	}

	inf, err := self.containerDataToContainerInfo(container, query)
	if err != nil {
		return info.ContainerInfo{}, err
	}
	return *inf, nil
}

func (self *manager) containerDataSliceToContainerInfoSlice(containers []*containerData, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	if len(containers) == 0 {
		return nil, fmt.Errorf("no containers found")
	}

	// Get the info for each container.
	output := make([]*info.ContainerInfo, 0, len(containers))
	for i := range containers {
		cinfo, err := self.containerDataToContainerInfo(containers[i], query)
		if err != nil {
			// Skip containers with errors, we try to degrade gracefully.
			continue
		}
		output = append(output, cinfo)
	}

	return output, nil
}

func (self *manager) GetRequestedContainersInfo(containerName string, options v2.RequestOptions) (map[string]*info.ContainerInfo, error) {
	containers, err := self.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	var errs partialFailure
	containersMap := make(map[string]*info.ContainerInfo)
	query := info.ContainerInfoRequest{
		NumStats: options.Count,
	}
	for name, data := range containers {
		info, err := self.containerDataToContainerInfo(data, &query)
		if err != nil {
			errs.append(name, "containerDataToContainerInfo", err)
		}
		containersMap[name] = info
	}
	return containersMap, errs.OrNil()
}

func (self *manager) getRequestedContainers(containerName string, options v2.RequestOptions) (map[string]*containerData, error) {
	containersMap := make(map[string]*containerData)
	switch options.IdType {
	case v2.TypeName:
		if options.Recursive == false {
			cont, err := self.getContainer(containerName)
			if err != nil {
				return containersMap, err
			}
			containersMap[cont.info.Name] = cont
		} else {
			containersMap = self.getSubcontainers(containerName)
			if len(containersMap) == 0 {
				return containersMap, fmt.Errorf("unknown container: %q", containerName)
			}
		}
	case v2.TypeDocker:
		if options.Recursive == false {
			containerName = strings.TrimPrefix(containerName, "/")
			cont, err := self.getDockerContainer(containerName)
			if err != nil {
				return containersMap, err
			}
			containersMap[cont.info.Name] = cont
		} else {
			if containerName != "/" {
				return containersMap, fmt.Errorf("invalid request for docker container %q with subcontainers", containerName)
			}
			containersMap = self.getAllDockerContainers()
		}
	default:
		return containersMap, fmt.Errorf("invalid request type %q", options.IdType)
	}
	if options.MaxAge != nil {
		// update stats for all containers in containersMap
		var waitGroup sync.WaitGroup
		waitGroup.Add(len(containersMap))
		for _, container := range containersMap {
			go func(cont *containerData) {
				cont.OnDemandHousekeeping(*options.MaxAge)
				waitGroup.Done()
			}(container)
		}
		waitGroup.Wait()
	}
	return containersMap, nil
}

func (self *manager) GetDirFsInfo(dir string) (v2.FsInfo, error) {
	device, err := self.fsInfo.GetDirFsDevice(dir)
	if err != nil {
		return v2.FsInfo{}, fmt.Errorf("failed to get device for dir %q: %v", dir, err)
	}
	return self.getFsInfoByDeviceName(device.Device)
}

func (self *manager) GetFsInfoByFsUUID(uuid string) (v2.FsInfo, error) {
	device, err := self.fsInfo.GetDeviceInfoByFsUUID(uuid)
	if err != nil {
		return v2.FsInfo{}, err
	}
	return self.getFsInfoByDeviceName(device.Device)
}

func (self *manager) GetFsInfo(label string) ([]v2.FsInfo, error) {
	var empty time.Time
	// Get latest data from filesystems hanging off root container.
	stats, err := self.memoryCache.RecentStats("/", empty, empty, 1)
	if err != nil {
		return nil, err
	}
	dev := ""
	if len(label) != 0 {
		dev, err = self.fsInfo.GetDeviceForLabel(label)
		if err != nil {
			return nil, err
		}
	}
	fsInfo := []v2.FsInfo{}
	for i := range stats[0].Filesystem {
		fs := stats[0].Filesystem[i]
		if len(label) != 0 && fs.Device != dev {
			continue
		}
		mountpoint, err := self.fsInfo.GetMountpointForDevice(fs.Device)
		if err != nil {
			return nil, err
		}
		labels, err := self.fsInfo.GetLabelsForDevice(fs.Device)
		if err != nil {
			return nil, err
		}

		fi := v2.FsInfo{
			Timestamp:  stats[0].Timestamp,
			Device:     fs.Device,
			Mountpoint: mountpoint,
			Capacity:   fs.Limit,
			Usage:      fs.Usage,
			Available:  fs.Available,
			Labels:     labels,
		}
		if fs.HasInodes {
			fi.Inodes = &fs.Inodes
			fi.InodesFree = &fs.InodesFree
		}
		fsInfo = append(fsInfo, fi)
	}
	return fsInfo, nil
}

func (m *manager) GetMachineInfo() (*info.MachineInfo, error) {
	m.machineMu.RLock()
	defer m.machineMu.RUnlock()
	// Copy and return the MachineInfo.
	return &m.machineInfo, nil
}

func (m *manager) GetVersionInfo() (*info.VersionInfo, error) {
	// TODO: Consider caching this and periodically updating.  The VersionInfo may change if
	// the docker daemon is started after the cAdvisor client is created.  Caching the value
	// would be helpful so we would be able to return the last known docker version if
	// docker was down at the time of a query.
	return getVersionInfo()
}

func (m *manager) Exists(containerName string) bool {
	m.containersLock.Lock()
	defer m.containersLock.Unlock()

	namespacedName := namespacedContainerName{
		Name: containerName,
	}

	_, ok := m.containers[namespacedName]
	if ok {
		return true
	}
	return false
}

func (m *manager) GetProcessList(containerName string, options v2.RequestOptions) ([]v2.ProcessInfo, error) {
	// override recursive. Only support single container listing.
	options.Recursive = false
	// override MaxAge.  ProcessList does not require updated stats.
	options.MaxAge = nil
	conts, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	if len(conts) != 1 {
		return nil, fmt.Errorf("Expected the request to match only one container")
	}
	// TODO(rjnagal): handle count? Only if we can do count by type (eg. top 5 cpu users)
	ps := []v2.ProcessInfo{}
	for _, cont := range conts {
		ps, err = cont.GetProcessList(m.cadvisorContainer, m.inHostNamespace)
		if err != nil {
			return nil, err
		}
	}
	return ps, nil
}

func (m *manager) registerCollectors(collectorConfigs map[string]string, cont *containerData) error {
	for k, v := range collectorConfigs {
		configFile, err := cont.ReadFile(v, m.inHostNamespace)
		if err != nil {
			return fmt.Errorf("failed to read config file %q for config %q, container %q: %v", k, v, cont.info.Name, err)
		}
		klog.V(4).Infof("Got config from %q: %q", v, configFile)

		if strings.HasPrefix(k, "prometheus") || strings.HasPrefix(k, "Prometheus") {
			newCollector, err := collector.NewPrometheusCollector(k, configFile, *applicationMetricsCountLimit, cont.handler, m.collectorHttpClient)
			if err != nil {
				return fmt.Errorf("failed to create collector for container %q, config %q: %v", cont.info.Name, k, err)
			}
			err = cont.collectorManager.RegisterCollector(newCollector)
			if err != nil {
				return fmt.Errorf("failed to register collector for container %q, config %q: %v", cont.info.Name, k, err)
			}
		} else {
			newCollector, err := collector.NewCollector(k, configFile, *applicationMetricsCountLimit, cont.handler, m.collectorHttpClient)
			if err != nil {
				return fmt.Errorf("failed to create collector for container %q, config %q: %v", cont.info.Name, k, err)
			}
			err = cont.collectorManager.RegisterCollector(newCollector)
			if err != nil {
				return fmt.Errorf("failed to register collector for container %q, config %q: %v", cont.info.Name, k, err)
			}
		}
	}
	return nil
}

// Enables overwriting an existing containerData/Handler object for a given containerName.
// Can't use createContainer as it just returns if a given containerName has a handler already.
// Ex: rkt handler will want to take priority over the raw handler, but the raw handler might be created first.

// Only allow raw handler to be overridden
func (m *manager) overrideContainer(containerName string, watchSource watcher.ContainerWatchSource) error {
	m.containersLock.Lock()
	defer m.containersLock.Unlock()

	namespacedName := namespacedContainerName{
		Name: containerName,
	}

	if _, ok := m.containers[namespacedName]; ok {
		containerData := m.containers[namespacedName]

		if containerData.handler.Type() != container.ContainerTypeRaw {
			return nil
		}

		err := m.destroyContainerLocked(containerName)
		if err != nil {
			return fmt.Errorf("overrideContainer: failed to destroy containerData/handler for %v: %v", containerName, err)
		}
	}

	return m.createContainerLocked(containerName, watchSource)
}

// Create a container.
func (m *manager) createContainer(containerName string, watchSource watcher.ContainerWatchSource) error {
	m.containersLock.Lock()
	defer m.containersLock.Unlock()

	return m.createContainerLocked(containerName, watchSource)
}

func (m *manager) createContainerLocked(containerName string, watchSource watcher.ContainerWatchSource) error {
	namespacedName := namespacedContainerName{
		Name: containerName,
	}

	// Check that the container didn't already exist.
	if _, ok := m.containers[namespacedName]; ok {
		return nil
	}

	handler, accept, err := container.NewContainerHandler(containerName, watchSource, m.inHostNamespace)
	if err != nil {
		return err
	}
	if !accept {
		// ignoring this container.
		klog.V(4).Infof("ignoring container %q", containerName)
		return nil
	}
	collectorManager, err := collector.NewCollectorManager()
	if err != nil {
		return err
	}

	logUsage := *logCadvisorUsage && containerName == m.cadvisorContainer
	cont, err := newContainerData(containerName, m.memoryCache, handler, logUsage, collectorManager, m.maxHousekeepingInterval, m.allowDynamicHousekeeping, clock.RealClock{})
	if err != nil {
		return err
	}
	if !cgroups.IsCgroup2UnifiedMode() {
		devicesCgroupPath, err := handler.GetCgroupPath("devices")
		if err != nil {
			klog.Warningf("Error getting devices cgroup path: %v", err)
		} else {
			cont.nvidiaCollector, err = m.nvidiaManager.GetCollector(devicesCgroupPath)
			if err != nil {
				klog.V(4).Infof("GPU metrics may be unavailable/incomplete for container %q: %v", cont.info.Name, err)
			}
		}
	}

	// Add collectors
	labels := handler.GetContainerLabels()
	collectorConfigs := collector.GetCollectorConfigs(labels)
	err = m.registerCollectors(collectorConfigs, cont)
	if err != nil {
		klog.Warningf("Failed to register collectors for %q: %v", containerName, err)
	}

	// Add the container name and all its aliases. The aliases must be within the namespace of the factory.
	m.containers[namespacedName] = cont
	for _, alias := range cont.info.Aliases {
		m.containers[namespacedContainerName{
			Namespace: cont.info.Namespace,
			Name:      alias,
		}] = cont
	}

	klog.V(3).Infof("Added container: %q (aliases: %v, namespace: %q)", containerName, cont.info.Aliases, cont.info.Namespace)

	contSpec, err := cont.handler.GetSpec()
	if err != nil {
		return err
	}

	contRef, err := cont.handler.ContainerReference()
	if err != nil {
		return err
	}

	newEvent := &info.Event{
		ContainerName: contRef.Name,
		Timestamp:     contSpec.CreationTime,
		EventType:     info.EventContainerCreation,
	}
	err = m.eventHandler.AddEvent(newEvent)
	if err != nil {
		return err
	}

	// Start the container's housekeeping.
	return cont.Start()
}

func (m *manager) destroyContainer(containerName string) error {
	m.containersLock.Lock()
	defer m.containersLock.Unlock()

	return m.destroyContainerLocked(containerName)
}

func (m *manager) destroyContainerLocked(containerName string) error {
	namespacedName := namespacedContainerName{
		Name: containerName,
	}
	cont, ok := m.containers[namespacedName]
	if !ok {
		// Already destroyed, done.
		return nil
	}

	// Tell the container to stop.
	err := cont.Stop()
	if err != nil {
		return err
	}

	// Remove the container from our records (and all its aliases).
	delete(m.containers, namespacedName)
	for _, alias := range cont.info.Aliases {
		delete(m.containers, namespacedContainerName{
			Namespace: cont.info.Namespace,
			Name:      alias,
		})
	}
	klog.V(3).Infof("Destroyed container: %q (aliases: %v, namespace: %q)", containerName, cont.info.Aliases, cont.info.Namespace)

	contRef, err := cont.handler.ContainerReference()
	if err != nil {
		return err
	}

	newEvent := &info.Event{
		ContainerName: contRef.Name,
		Timestamp:     time.Now(),
		EventType:     info.EventContainerDeletion,
	}
	err = m.eventHandler.AddEvent(newEvent)
	if err != nil {
		return err
	}
	return nil
}

// Detect all containers that have been added or deleted from the specified container.
func (m *manager) getContainersDiff(containerName string) (added []info.ContainerReference, removed []info.ContainerReference, err error) {
	// Get all subcontainers recursively.
	m.containersLock.RLock()
	cont, ok := m.containers[namespacedContainerName{
		Name: containerName,
	}]
	m.containersLock.RUnlock()
	if !ok {
		return nil, nil, fmt.Errorf("failed to find container %q while checking for new containers", containerName)
	}
	allContainers, err := cont.handler.ListContainers(container.ListRecursive)

	if err != nil {
		return nil, nil, err
	}
	allContainers = append(allContainers, info.ContainerReference{Name: containerName})

	m.containersLock.RLock()
	defer m.containersLock.RUnlock()

	// Determine which were added and which were removed.
	allContainersSet := make(map[string]*containerData)
	for name, d := range m.containers {
		// Only add the canonical name.
		if d.info.Name == name.Name {
			allContainersSet[name.Name] = d
		}
	}

	// Added containers
	for _, c := range allContainers {
		delete(allContainersSet, c.Name)
		_, ok := m.containers[namespacedContainerName{
			Name: c.Name,
		}]
		if !ok {
			added = append(added, c)
		}
	}

	// Removed ones are no longer in the container listing.
	for _, d := range allContainersSet {
		removed = append(removed, d.info.ContainerReference)
	}

	return
}

// Detect the existing subcontainers and reflect the setup here.
func (m *manager) detectSubcontainers(containerName string) error {
	added, removed, err := m.getContainersDiff(containerName)
	if err != nil {
		return err
	}

	// Add the new containers.
	for _, cont := range added {
		err = m.createContainer(cont.Name, watcher.Raw)
		if err != nil {
			klog.Errorf("Failed to create existing container: %s: %s", cont.Name, err)
		}
	}

	// Remove the old containers.
	for _, cont := range removed {
		err = m.destroyContainer(cont.Name)
		if err != nil {
			klog.Errorf("Failed to destroy existing container: %s: %s", cont.Name, err)
		}
	}

	return nil
}

// Watches for new containers started in the system. Runs forever unless there is a setup error.
func (self *manager) watchForNewContainers(quit chan error) error {
	for _, watcher := range self.containerWatchers {
		err := watcher.Start(self.eventsChannel)
		if err != nil {
			return err
		}
	}

	// There is a race between starting the watch and new container creation so we do a detection before we read new containers.
	err := self.detectSubcontainers("/")
	if err != nil {
		return err
	}

	// Listen to events from the container handler.
	go func() {
		for {
			select {
			case event := <-self.eventsChannel:
				switch {
				case event.EventType == watcher.ContainerAdd:
					switch event.WatchSource {
					default:
						err = self.createContainer(event.Name, event.WatchSource)
					}
				case event.EventType == watcher.ContainerDelete:
					err = self.destroyContainer(event.Name)
				}
				if err != nil {
					klog.Warningf("Failed to process watch event %+v: %v", event, err)
				}
			case <-quit:
				var errs partialFailure

				// Stop processing events if asked to quit.
				for i, watcher := range self.containerWatchers {
					err := watcher.Stop()
					if err != nil {
						errs.append(fmt.Sprintf("watcher %d", i), "Stop", err)
					}
				}

				if len(errs) > 0 {
					quit <- errs
				} else {
					quit <- nil
					klog.Infof("Exiting thread watching subcontainers")
					return
				}
			}
		}
	}()
	return nil
}

func (self *manager) watchForNewOoms() error {
	klog.V(2).Infof("Started watching for new ooms in manager")
	outStream := make(chan *oomparser.OomInstance, 10)
	oomLog, err := oomparser.New()
	if err != nil {
		return err
	}
	go oomLog.StreamOoms(outStream)

	go func() {
		for oomInstance := range outStream {
			// Surface OOM and OOM kill events.
			newEvent := &info.Event{
				ContainerName: oomInstance.ContainerName,
				Timestamp:     oomInstance.TimeOfDeath,
				EventType:     info.EventOom,
			}
			err := self.eventHandler.AddEvent(newEvent)
			if err != nil {
				klog.Errorf("failed to add OOM event for %q: %v", oomInstance.ContainerName, err)
			}
			klog.V(3).Infof("Created an OOM event in container %q at %v", oomInstance.ContainerName, oomInstance.TimeOfDeath)

			newEvent = &info.Event{
				ContainerName: oomInstance.VictimContainerName,
				Timestamp:     oomInstance.TimeOfDeath,
				EventType:     info.EventOomKill,
				EventData: info.EventData{
					OomKill: &info.OomKillEventData{
						Pid:         oomInstance.Pid,
						ProcessName: oomInstance.ProcessName,
					},
				},
			}
			err = self.eventHandler.AddEvent(newEvent)
			if err != nil {
				klog.Errorf("failed to add OOM kill event for %q: %v", oomInstance.ContainerName, err)
			}
		}
	}()
	return nil
}

// can be called by the api which will take events returned on the channel
func (self *manager) WatchForEvents(request *events.Request) (*events.EventChannel, error) {
	return self.eventHandler.WatchEvents(request)
}

// can be called by the api which will return all events satisfying the request
func (self *manager) GetPastEvents(request *events.Request) ([]*info.Event, error) {
	return self.eventHandler.GetEvents(request)
}

// called by the api when a client is no longer listening to the channel
func (self *manager) CloseEventChannel(watch_id int) {
	self.eventHandler.StopWatch(watch_id)
}

// Parses the events StoragePolicy from the flags.
func parseEventsStoragePolicy() events.StoragePolicy {
	policy := events.DefaultStoragePolicy()

	// Parse max age.
	parts := strings.Split(*eventStorageAgeLimit, ",")
	for _, part := range parts {
		items := strings.Split(part, "=")
		if len(items) != 2 {
			klog.Warningf("Unknown event storage policy %q when parsing max age", part)
			continue
		}
		dur, err := time.ParseDuration(items[1])
		if err != nil {
			klog.Warningf("Unable to parse event max age duration %q: %v", items[1], err)
			continue
		}
		if items[0] == "default" {
			policy.DefaultMaxAge = dur
			continue
		}
		policy.PerTypeMaxAge[info.EventType(items[0])] = dur
	}

	// Parse max number.
	parts = strings.Split(*eventStorageEventLimit, ",")
	for _, part := range parts {
		items := strings.Split(part, "=")
		if len(items) != 2 {
			klog.Warningf("Unknown event storage policy %q when parsing max event limit", part)
			continue
		}
		val, err := strconv.Atoi(items[1])
		if err != nil {
			klog.Warningf("Unable to parse integer from %q: %v", items[1], err)
			continue
		}
		if items[0] == "default" {
			policy.DefaultMaxNumEvents = val
			continue
		}
		policy.PerTypeMaxNumEvents[info.EventType(items[0])] = val
	}

	return policy
}

func (m *manager) DockerImages() ([]info.DockerImage, error) {
	return docker.Images()
}

func (m *manager) DockerInfo() (info.DockerStatus, error) {
	return docker.Status()
}

func (m *manager) DebugInfo() map[string][]string {
	debugInfo := container.DebugInfo()

	// Get unique containers.
	var conts map[*containerData]struct{}
	func() {
		m.containersLock.RLock()
		defer m.containersLock.RUnlock()

		conts = make(map[*containerData]struct{}, len(m.containers))
		for _, c := range m.containers {
			conts[c] = struct{}{}
		}
	}()

	// List containers.
	lines := make([]string, 0, len(conts))
	for cont := range conts {
		lines = append(lines, cont.info.Name)
		if cont.info.Namespace != "" {
			lines = append(lines, fmt.Sprintf("\tNamespace: %s", cont.info.Namespace))
		}

		if len(cont.info.Aliases) != 0 {
			lines = append(lines, "\tAliases:")
			for _, alias := range cont.info.Aliases {
				lines = append(lines, fmt.Sprintf("\t\t%s", alias))
			}
		}
	}

	debugInfo["Managed containers"] = lines
	return debugInfo
}

func (self *manager) getFsInfoByDeviceName(deviceName string) (v2.FsInfo, error) {
	mountPoint, err := self.fsInfo.GetMountpointForDevice(deviceName)
	if err != nil {
		return v2.FsInfo{}, fmt.Errorf("failed to get mount point for device %q: %v", deviceName, err)
	}
	infos, err := self.GetFsInfo("")
	if err != nil {
		return v2.FsInfo{}, err
	}
	for _, info := range infos {
		if info.Mountpoint == mountPoint {
			return info, nil
		}
	}
	return v2.FsInfo{}, fmt.Errorf("cannot find filesystem info for device %q", deviceName)
}

func getVersionInfo() (*info.VersionInfo, error) {

	kernel_version := machine.KernelVersion()
	container_os := machine.ContainerOsVersion()
	docker_version, err := docker.VersionString()
	if err != nil {
		return nil, err
	}
	docker_api_version, err := docker.APIVersionString()
	if err != nil {
		return nil, err
	}

	return &info.VersionInfo{
		KernelVersion:      kernel_version,
		ContainerOsVersion: container_os,
		DockerVersion:      docker_version,
		DockerAPIVersion:   docker_api_version,
		CadvisorVersion:    version.Info["version"],
		CadvisorRevision:   version.Info["revision"],
	}, nil
}

// Helper for accumulating partial failures.
type partialFailure []string

func (f *partialFailure) append(id, operation string, err error) {
	*f = append(*f, fmt.Sprintf("[%q: %s: %s]", id, operation, err))
}

func (f partialFailure) Error() string {
	return fmt.Sprintf("partial failures: %s", strings.Join(f, ", "))
}

func (f partialFailure) OrNil() error {
	if len(f) == 0 {
		return nil
	}
	return f
}
