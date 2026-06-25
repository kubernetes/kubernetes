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

//go:build linux

// Manager of cAdvisor-monitored containers.
package manager

import (
	"flag"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/cadvisor/lib/cache/memory"
	"github.com/google/cadvisor/lib/container"
	"github.com/google/cadvisor/lib/container/raw"
	"github.com/google/cadvisor/lib/fs"
	"github.com/google/cadvisor/lib/machine"
	info "github.com/google/cadvisor/lib/model"
	"github.com/google/cadvisor/lib/stats"
	"github.com/google/cadvisor/lib/utils/oomparser"
	"github.com/google/cadvisor/lib/utils/sysfs"
	"github.com/google/cadvisor/lib/version"
	"github.com/google/cadvisor/lib/watcher"

	"github.com/opencontainers/cgroups"

	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
)

var globalHousekeepingInterval = flag.Duration("global_housekeeping_interval", 1*time.Minute, "Interval between global housekeepings")
var updateMachineInfoInterval = flag.Duration("update_machine_info_interval", 5*time.Minute, "Interval between machine info updates.")

// log_cadvisor_usage is retained as a registered no-op: the per-container usage
// logging path was removed (C6), but the kubelet pins this flag by name in
// cmd/kubelet/app/options/globalflags_linux.go, so it must keep resolving.
var _ = flag.Bool("log_cadvisor_usage", false, "Whether to log the usage of the cAdvisor container")
var eventStorageAgeLimit = flag.String("event_storage_age_limit", "default=24h", "Max length of time for which to store events (per type). Value is a comma separated list of key values, where the keys are event types (e.g.: creation, oom) or \"default\" and the value is a duration. Default is applied to all non-specified event types")
var eventStorageEventLimit = flag.String("event_storage_event_limit", "default=100000", "Max number of events to store (per type). Value is a comma separated list of key values, where the keys are event types (e.g.: creation, oom) or \"default\" and the value is an integer. Default is applied to all non-specified event types")

// application_metrics_count_limit is registered here so the kubelet's global
// flag of the same name resolves at startup. The full binary's collector
// factory reads its value via ApplicationMetricsCountLimit; the kubelet runs no
// collectors, so it does not.
var applicationMetricsCountLimit = flag.Int("application_metrics_count_limit", 100, "Max number of application metrics to store (per container)")

// ApplicationMetricsCountLimit exposes the application_metrics_count_limit flag
// value so the root binary can build its collectors without re-registering the
// flag.
func ApplicationMetricsCountLimit() int { return *applicationMetricsCountLimit }

// The namespace under which aliases are unique.
const (
	DockerNamespace = "docker"
	PodmanNamespace = "podman"
)

var HousekeepingConfigFlags = HousekeepingConfig{
	flag.Duration("max_housekeeping_interval", 60*time.Second, "Largest interval to allow between container housekeepings"),
	flag.Bool("allow_dynamic_housekeeping", true, "Whether to allow the housekeeping interval to be dynamic"),
}

// The Manager interface defines operations for starting a manager and getting
// container and machine information.
type Manager interface {
	// Start the manager. Calling other manager methods before this returns
	// may produce undefined behavior.
	Start() error

	// Stops the manager.
	Stop() error

	// Get V2 information about a container.
	// Recursive (subcontainer) requests are best-effort, and may return a partial result alongside an
	// error in the partial failure case.
	GetContainerInfoV2(containerName string, options info.RequestOptions) (map[string]info.ContainerInfo, error)

	// Get info for all requested containers based on the request options.
	GetRequestedContainersInfo(containerName string, options info.RequestOptions) (map[string]*info.ContainerInfo, error)

	// Get information about the machine.
	GetMachineInfo() (*info.MachineInfo, error)

	// Get version information about different components we depend on.
	GetVersionInfo() (*info.VersionInfo, error)

	// Get filesystem information for the filesystem that contains the given directory
	GetDirFsInfo(dir string) (info.FsInfo, error)

	// Get filesystem information for a given label.
	// Returns information for all global filesystems if label is empty.
	GetFsInfo(label string) ([]info.FsInfo, error)

	// DebugInfo returns debug information about the manager's tracked containers.
	DebugInfo() map[string][]string

	// The methods below back the full cAdvisor binary's v1/v2 REST API and web
	// UI. They are pure queries over the in-memory container registry and add no
	// dependencies; the kubelet does not call them. See queries.go.

	// Get information about a container.
	GetContainerInfo(containerName string, query *info.ContainerInfoRequest) (*info.ContainerInfo, error)

	// Get information about all subcontainers of the specified container (includes self).
	SubcontainersInfo(containerName string, query *info.ContainerInfoRequest) ([]*info.ContainerInfo, error)

	// Get information about the docker containers (by container name) of an instance.
	AllDockerContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error)

	// Get information about the docker container with the specified name.
	DockerContainer(dockerName string, query *info.ContainerInfoRequest) (info.ContainerInfo, error)

	// Get information about the podman containers (by container name) of an instance.
	AllPodmanContainers(query *info.ContainerInfoRequest) (map[string]info.ContainerInfo, error)

	// Get information about the podman container with the specified name.
	PodmanContainer(containerName string, query *info.ContainerInfoRequest) (info.ContainerInfo, error)

	// Get the specs for a container, possibly with subcontainers.
	GetContainerSpec(containerName string, options info.RequestOptions) (map[string]info.ContainerSpec, error)

	// Get filesystem information for the filesystem identified by the given UUID.
	GetFsInfoByFsUUID(uuid string) (info.FsInfo, error)

	// Returns true if the named container exists.
	Exists(containerName string) bool

	// Get the derived (rolling-window percentile) stats for a container.
	GetDerivedStats(containerName string, options info.RequestOptions) (map[string]info.DerivedStats, error)

	// Get the list of processes running in a container.
	GetProcessList(containerName string, options info.RequestOptions) ([]info.ProcessInfo, error)

	// SetEventSink wires the sink that receives container lifecycle and OOM
	// events. The full binary injects its events manager; the kubelet does not
	// call this (no events are emitted). See events.go.
	SetEventSink(sink EventSink)
}

// Housekeeping configuration for the manager
type HousekeepingConfig = struct {
	Interval     *time.Duration
	AllowDynamic *bool
}

// New takes a memory storage and returns a new manager.
func New(memoryCache *memory.InMemoryCache, sysfs sysfs.SysFs, HousekeepingConfig HousekeepingConfig, includedMetricsSet container.MetricSet, rawContainerCgroupPathPrefixWhiteList, containerEnvMetadataWhiteList []string, perfEventsFile string, resctrlInterval time.Duration) (Manager, error) {
	if memoryCache == nil {
		return nil, fmt.Errorf("manager requires memory storage")
	}

	// Detect the container we are running on.
	selfContainer := "/"
	var err error
	// Avoid using GetOwnCgroupPath on cgroup v2 as it is not supported by libcontainer
	if !cgroups.IsCgroup2UnifiedMode() {
		selfContainer, err = cgroups.GetOwnCgroup("cpu")
		if err != nil {
			return nil, err
		}
		klog.V(2).Infof("cAdvisor running in container: %q", selfContainer)
	}

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
		quitChannels:                          make([]chan error, 0, 2),
		memoryCache:                           memoryCache,
		fsInfo:                                fsInfo,
		sysFs:                                 sysfs,
		cadvisorContainer:                     selfContainer,
		inHostNamespace:                       inHostNamespace,
		startupTime:                           time.Now(),
		maxHousekeepingInterval:               *HousekeepingConfig.Interval,
		allowDynamicHousekeeping:              *HousekeepingConfig.AllowDynamic,
		includedMetrics:                       includedMetricsSet,
		containerWatchers:                     []watcher.ContainerWatcher{},
		eventsChannel:                         eventsChannel,
		rawContainerCgroupPathPrefixWhiteList: rawContainerCgroupPathPrefixWhiteList,
		containerEnvMetadataWhiteList:         containerEnvMetadataWhiteList,
	}

	machineInfo, err := machine.Info(sysfs, fsInfo, inHostNamespace)
	if err != nil {
		return nil, err
	}
	newManager.machineInfo = *machineInfo
	klog.V(1).Infof("Machine: %+v", newManager.machineInfo)

	// Collector managers default to Noop; the full cAdvisor binary registers
	// real perf/resctrl factories via plugins.go. The kubelet leaves them nil.
	newManager.perfManager = &stats.NoopManager{}
	if PerfManagerFactory != nil {
		newManager.perfManager, err = PerfManagerFactory(perfEventsFile, machineInfo.Topology)
		if err != nil {
			return nil, err
		}
	}
	newManager.resctrlManager = &stats.NoopResctrlManager{}
	if ResctrlManagerFactory != nil {
		if rm, rerr := ResctrlManagerFactory(resctrlInterval, machineInfo.CPUVendorID, inHostNamespace); rerr != nil {
			klog.V(4).Infof("Cannot gather resctrl metrics: %v", rerr)
		} else {
			newManager.resctrlManager = rm
		}
	}

	versionInfo, err := getVersionInfo()
	if err != nil {
		return nil, err
	}
	klog.V(1).Infof("Version: %+v", *versionInfo)

	return newManager, nil
}

// A namespaced container name.
type namespacedContainerName struct {
	// The namespace of the container. Can be empty for the root namespace.
	Namespace string

	// The name of the container in this namespace.
	Name string
}

// containerMap is a type-safe wrapper around sync.Map for storing containerData
// keyed by namespacedContainerName.
type containerMap struct {
	m sync.Map
}

// Load returns the containerData for the given name, or nil if not found.
func (c *containerMap) Load(name namespacedContainerName) (*containerData, bool) {
	v, ok := c.m.Load(name)
	if !ok {
		return nil, false
	}
	return v.(*containerData), true
}

// Store stores the containerData for the given name.
func (c *containerMap) Store(name namespacedContainerName, data *containerData) {
	c.m.Store(name, data)
}

// Delete removes the containerData for the given name.
func (c *containerMap) Delete(name namespacedContainerName) {
	c.m.Delete(name)
}

// Range calls f for each container in the map. If f returns false, iteration stops.
func (c *containerMap) Range(f func(name namespacedContainerName, data *containerData) bool) {
	c.m.Range(func(key, value any) bool {
		return f(key.(namespacedContainerName), value.(*containerData))
	})
}

type manager struct {
	containers               containerMap
	memoryCache              *memory.InMemoryCache
	fsInfo                   fs.FsInfo
	sysFs                    sysfs.SysFs
	machineMu                sync.RWMutex // protects machineInfo
	machineInfo              info.MachineInfo
	quitChannels             []chan error
	cadvisorContainer        string
	inHostNamespace          bool
	startupTime              time.Time
	maxHousekeepingInterval  time.Duration
	allowDynamicHousekeeping bool
	includedMetrics          container.MetricSet
	containerWatchers        []watcher.ContainerWatcher
	eventsChannel            chan watcher.ContainerEvent
	// List of raw container cgroup path prefix whitelist.
	rawContainerCgroupPathPrefixWhiteList []string
	// List of container env prefix whitelist, the matched container envs would be collected into metrics as extra labels.
	containerEnvMetadataWhiteList []string

	// Collector managers for perf_event / resctrl. Default to Noop; the full
	// cAdvisor binary injects real implementations via plugins.go.
	perfManager    stats.Manager
	resctrlManager stats.ResctrlManager

	// eventSink, if set by the full binary via SetEventSink, receives container
	// lifecycle and OOM events. nil for the kubelet (no event machinery). See
	// events.go.
	eventSink EventSink
}

// Start the container manager.
func (m *manager) Start() error {
	m.containerWatchers = container.InitializePlugins(m, m.fsInfo, m.includedMetrics)

	err := raw.Register(m, m.fsInfo, m.includedMetrics, m.rawContainerCgroupPathPrefixWhiteList)
	if err != nil {
		klog.Errorf("Registration of the raw container factory failed: %v", err)
	}

	rawWatcher, err := raw.NewRawContainerWatcher(m.includedMetrics)
	if err != nil {
		return err
	}
	m.containerWatchers = append(m.containerWatchers, rawWatcher)

	// Watch for OOMs.
	if err := m.watchForNewOoms(); err != nil {
		klog.Warningf("Could not configure a source for OOM detection, disabling OOM events: %v", err)
	}

	// If there are no factories, don't start any housekeeping and serve the information we do have.
	if !container.HasFactories() {
		return nil
	}

	// Create root and then recover all containers.
	err = m.createContainer("/", watcher.Raw)
	if err != nil {
		return err
	}
	klog.V(2).Infof("Starting recovery of all containers")
	err = m.detectSubcontainers("/")
	if err != nil {
		return err
	}
	klog.V(2).Infof("Recovery completed")

	// Watch for new container.
	quitWatcher := make(chan error)
	err = m.watchForNewContainers(quitWatcher)
	if err != nil {
		return err
	}
	m.quitChannels = append(m.quitChannels, quitWatcher)

	// Look for new containers in the main housekeeping thread.
	quitGlobalHousekeeping := make(chan error)
	m.quitChannels = append(m.quitChannels, quitGlobalHousekeeping)
	go m.globalHousekeeping(quitGlobalHousekeeping)

	quitUpdateMachineInfo := make(chan error)
	m.quitChannels = append(m.quitChannels, quitUpdateMachineInfo)
	go m.updateMachineInfo(quitUpdateMachineInfo)

	return nil
}

func (m *manager) Stop() error {
	defer m.destroyCollectors()
	// Stop and wait on all quit channels.
	for i, c := range m.quitChannels {
		// Send the exit signal and wait on the thread to exit (by closing the channel).
		c <- nil
		err := <-c
		if err != nil {
			// Remove the channels that quit successfully.
			m.quitChannels = m.quitChannels[i:]
			return err
		}
	}
	m.quitChannels = make([]chan error, 0, 2)
	return nil
}

func (m *manager) destroyCollectors() {
	m.containers.Range(func(_ namespacedContainerName, container *containerData) bool {
		if container == nil {
			return true
		}
		container.perfCollector.Destroy()
		container.resctrlCollector.Destroy()
		return true
	})
}

func (m *manager) updateMachineInfo(quit chan error) {
	ticker := time.NewTicker(*updateMachineInfoInterval)
	for {
		select {
		case <-ticker.C:
			info, err := machine.Info(m.sysFs, m.fsInfo, m.inHostNamespace)
			if err != nil {
				klog.Errorf("Could not get machine info: %v", err)
				break
			}
			m.machineMu.Lock()
			m.machineInfo = *info
			m.machineMu.Unlock()
			klog.V(5).Infof("Update machine info: %+v", *info)
		case <-quit:
			ticker.Stop()
			quit <- nil
			return
		}
	}
}

func (m *manager) globalHousekeeping(quit chan error) {
	// Long housekeeping is either 100ms or half of the housekeeping interval.
	longHousekeeping := 100 * time.Millisecond
	if *globalHousekeepingInterval/2 < longHousekeeping {
		longHousekeeping = *globalHousekeepingInterval / 2
	}

	ticker := time.NewTicker(*globalHousekeepingInterval)
	for {
		select {
		case t := <-ticker.C:
			start := time.Now()

			// Check for new containers.
			err := m.detectSubcontainers("/")
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

func (m *manager) getAdjustedSpec(cinfo *containerInfo) info.ContainerSpec {
	spec := cinfo.Spec

	// Set default value to an actual value
	if spec.HasMemory {
		// Memory.Limit is 0 means there's no limit
		if spec.Memory.Limit == 0 {
			m.machineMu.RLock()
			spec.Memory.Limit = uint64(m.machineInfo.MemoryCapacity)
			m.machineMu.RUnlock()
		}
	}
	return spec
}

func (m *manager) GetContainerInfoV2(containerName string, options info.RequestOptions) (map[string]info.ContainerInfo, error) {
	containers, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}

	var errs partialFailure
	var nilTime time.Time // Ignored.

	infos := make(map[string]info.ContainerInfo, len(containers))
	for name, container := range containers {
		result := info.ContainerInfo{}
		cinfo, err := container.GetInfo(false)
		if err != nil {
			errs.append(name, "GetInfo", err)
			infos[name] = result
			continue
		}
		result.Spec = m.getAdjustedSpec(cinfo)
		result.ContainerReference = cinfo.ContainerReference

		stats, err := m.memoryCache.RecentStats(name, nilTime, nilTime, options.Count)
		if err != nil {
			errs.append(name, "RecentStats", err)
			infos[name] = result
			continue
		}

		statsOut := make([]*info.ContainerStats, len(stats))
		var lastStat *info.ContainerStats
		for i, s := range stats {
			cp := *s
			if cinfo.Spec.HasCpu {
				if ci, err := info.InstCpuStats(lastStat, s); err == nil {
					cp.CpuInst = ci
				}
			}
			statsOut[i] = &cp
			lastStat = s
		}
		result.Stats = statsOut
		infos[name] = result
	}

	return infos, errs.OrNil()
}

func (m *manager) containerDataToContainerInfo(cont *containerData, query *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	// Get the info from the container.
	cinfo, err := cont.GetInfo(true)
	if err != nil {
		return nil, err
	}

	stats, err := m.memoryCache.RecentStats(cinfo.Name, query.Start, query.End, query.NumStats)
	if err != nil {
		return nil, err
	}

	// Make a copy of the info for the user.
	ret := &info.ContainerInfo{
		ContainerReference: cinfo.ContainerReference,
		Subcontainers:      cinfo.Subcontainers,
		Spec:               m.getAdjustedSpec(cinfo),
		Stats:              stats,
	}
	return ret, nil
}

func (m *manager) getContainer(containerName string) (*containerData, error) {
	cont, ok := m.containers.Load(namespacedContainerName{Name: containerName})
	if !ok {
		return nil, fmt.Errorf("unknown container %q", containerName)
	}
	return cont, nil
}

func (m *manager) getSubcontainers(containerName string) map[string]*containerData {
	matchedName := path.Join(containerName, "/")
	containersMap := make(map[string]*containerData)

	// Get all the unique subcontainers of the specified container
	m.containers.Range(func(_ namespacedContainerName, cont *containerData) bool {
		if cont == nil {
			return true
		}
		name := cont.info.Name
		if name == containerName || strings.HasPrefix(name, matchedName) {
			containersMap[name] = cont
		}
		return true
	})
	return containersMap
}

func (m *manager) getAllNamespacedContainers(ns string) map[string]*containerData {
	containers := make(map[string]*containerData)

	// Get containers in a namespace.
	m.containers.Range(func(name namespacedContainerName, cont *containerData) bool {
		if cont == nil {
			return true
		}
		if name.Namespace == ns {
			containers[cont.info.Name] = cont
		}
		return true
	})
	return containers
}

func (m *manager) namespacedContainer(containerName string, ns string) (*containerData, error) {
	// Check for the container in the namespace.
	if cont, ok := m.containers.Load(namespacedContainerName{Namespace: ns, Name: containerName}); ok {
		return cont, nil
	}

	// Look for container by short prefix name if no exact match found.
	var cont *containerData
	var err error
	m.containers.Range(func(name namespacedContainerName, c *containerData) bool {
		if name.Namespace == ns && strings.HasPrefix(name.Name, containerName) {
			if cont == nil {
				cont = c
			} else {
				err = fmt.Errorf("unable to find container in %q namespace. Container %q is not unique", ns, containerName)
				return false // stop iteration
			}
		}
		return true
	})

	if err != nil {
		return nil, err
	}

	if cont == nil {
		return nil, fmt.Errorf("unable to find container %q in %q namespace", containerName, ns)
	}

	return cont, nil
}

func (m *manager) GetRequestedContainersInfo(containerName string, options info.RequestOptions) (map[string]*info.ContainerInfo, error) {
	containers, err := m.getRequestedContainers(containerName, options)
	if err != nil {
		return nil, err
	}
	var errs partialFailure
	containersMap := make(map[string]*info.ContainerInfo)
	query := info.ContainerInfoRequest{
		NumStats: options.Count,
	}
	for name, data := range containers {
		info, err := m.containerDataToContainerInfo(data, &query)
		if err != nil {
			if err == memory.ErrDataNotFound {
				klog.V(4).Infof("Error getting data for container %s because of race condition", name)
				continue
			}
			errs.append(name, "containerDataToContainerInfo", err)
		}
		containersMap[name] = info
	}
	return containersMap, errs.OrNil()
}

// watchForNewOoms feeds a per-container OOM-kill counter from the kernel log via
// oomparser. It is an engine-level async router (NOT a per-tick StatsAugmenter):
// OOM events arrive keyed by container name and are matched to the registry here.
// No event stream — only the counter that backs container_oom_events_total.
func (m *manager) watchForNewOoms() error {
	outStream := make(chan *oomparser.OomInstance, 10)
	oomLog, err := oomparser.New()
	if err != nil {
		return err
	}
	go oomLog.StreamOoms(outStream)

	go func() {
		for oomInstance := range outStream {
			// Surface OOM and OOM-kill events to the sink (no-op if unset).
			m.addEvent(&info.Event{
				ContainerName: oomInstance.ContainerName,
				Timestamp:     oomInstance.TimeOfDeath,
				EventType:     info.EventOom,
			})
			m.addEvent(&info.Event{
				ContainerName: oomInstance.VictimContainerName,
				Timestamp:     oomInstance.TimeOfDeath,
				EventType:     info.EventOomKill,
				EventData: info.EventData{
					OomKill: &info.OomKillEventData{
						Pid:         oomInstance.Pid,
						ProcessName: oomInstance.ProcessName,
						Constraint:  oomInstance.Constraint,
					},
				},
			})

			// Count OOM events for later collection by prometheus.
			conts, err := m.getRequestedContainers(oomInstance.ContainerName, info.RequestOptions{IdType: info.TypeName, Count: 1})
			if err != nil || len(conts) != 1 {
				continue
			}
			for _, cont := range conts {
				atomic.AddUint64(&cont.oomEvents, 1)
			}
		}
	}()
	return nil
}

func (m *manager) getRequestedContainers(containerName string, options info.RequestOptions) (map[string]*containerData, error) {
	containersMap := make(map[string]*containerData)
	switch options.IdType {
	case info.TypeName:
		if !options.Recursive {
			cont, err := m.getContainer(containerName)
			if err != nil {
				return containersMap, err
			}
			containersMap[cont.info.Name] = cont
		} else {
			containersMap = m.getSubcontainers(containerName)
			if len(containersMap) == 0 {
				return containersMap, fmt.Errorf("unknown container: %q", containerName)
			}
		}
	case info.TypeDocker, info.TypePodman:
		namespace := map[string]string{
			info.TypeDocker: DockerNamespace,
			info.TypePodman: PodmanNamespace,
		}[options.IdType]
		if !options.Recursive {
			containerName = strings.TrimPrefix(containerName, "/")
			cont, err := m.namespacedContainer(containerName, namespace)
			if err != nil {
				return containersMap, err
			}
			containersMap[cont.info.Name] = cont
		} else {
			if containerName != "/" {
				return containersMap, fmt.Errorf("invalid request for %s container %q with subcontainers", options.IdType, containerName)
			}
			containersMap = m.getAllNamespacedContainers(namespace)
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

func (m *manager) GetDirFsInfo(dir string) (info.FsInfo, error) {
	device, err := m.fsInfo.GetDirFsDevice(dir)
	if err != nil {
		return info.FsInfo{}, fmt.Errorf("failed to get device for dir %q: %v", dir, err)
	}
	return m.getFsInfoByDeviceName(device.Device)
}

func (m *manager) GetFsInfo(label string) ([]info.FsInfo, error) {
	var empty time.Time
	// Get latest data from filesystems hanging off root container.
	stats, err := m.memoryCache.RecentStats("/", empty, empty, 1)
	if err != nil {
		return nil, err
	}
	dev := ""
	if len(label) != 0 {
		dev, err = m.fsInfo.GetDeviceForLabel(label)
		if err != nil {
			return nil, err
		}
	}
	fsInfo := []info.FsInfo{}
	for i := range stats[0].Filesystem {
		fs := stats[0].Filesystem[i]
		if len(label) != 0 && fs.Device != dev {
			continue
		}
		mountpoint, err := m.fsInfo.GetMountpointForDevice(fs.Device)
		if err != nil {
			return nil, err
		}
		labels, err := m.fsInfo.GetLabelsForDevice(fs.Device)
		if err != nil {
			return nil, err
		}

		fi := info.FsInfo{
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
	return m.machineInfo.Clone(), nil
}

func (m *manager) DebugInfo() map[string][]string {
	debugInfo := container.DebugInfo()

	// Get unique containers.
	conts := make(map[*containerData]struct{})
	m.containers.Range(func(_ namespacedContainerName, cont *containerData) bool {
		if cont != nil {
			conts[cont] = struct{}{}
		}
		return true
	})

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

func (m *manager) GetVersionInfo() (*info.VersionInfo, error) {
	// TODO: Consider caching this and periodically updating.  The VersionInfo may change if
	// the docker daemon is started after the cAdvisor client is created.  Caching the value
	// would be helpful so we would be able to return the last known docker version if
	// docker was down at the time of a query.
	return getVersionInfo()
}

// Create a container.
func (m *manager) createContainer(containerName string, watchSource watcher.ContainerWatchSource) error {
	namespacedName := namespacedContainerName{
		Name: containerName,
	}

	// Check that the container didn't already exist.
	if _, ok := m.containers.Load(namespacedName); ok {
		return nil
	}

	handler, accept, err := container.NewContainerHandler(containerName, watchSource, m.containerEnvMetadataWhiteList, m.inHostNamespace)
	if err != nil {
		return err
	}
	if !accept {
		// ignoring this container.
		klog.V(4).Infof("ignoring container %q", containerName)
		return nil
	}
	cont, err := newContainerData(containerName, m.memoryCache, handler, m.maxHousekeepingInterval, m.allowDynamicHousekeeping, clock.RealClock{})
	if err != nil {
		return err
	}

	if m.includedMetrics.Has(container.PerfMetrics) {
		perfCgroupPath, err := handler.GetCgroupPath("perf_event")
		if err != nil {
			klog.Warningf("Error getting perf_event cgroup path: %q", err)
		} else {
			cont.perfCollector, err = m.perfManager.GetCollector(perfCgroupPath)
			if err != nil {
				klog.Errorf("Perf event metrics will not be available for container %q: %v", containerName, err)
			}
		}
	}
	if m.includedMetrics.Has(container.ResctrlMetrics) {
		m.machineMu.RLock()
		noOfNUMA := len(m.machineInfo.Topology)
		m.machineMu.RUnlock()
		cont.resctrlCollector, err = m.resctrlManager.GetCollector(containerName, func() ([]string, error) {
			pids, perr := handler.ListProcesses(container.ListSelf)
			if perr != nil {
				return nil, perr
			}
			ss := make([]string, len(pids))
			for i, p := range pids {
				ss[i] = strconv.Itoa(p)
			}
			return ss, nil
		}, noOfNUMA)
		if err != nil {
			klog.V(4).Infof("resctrl metrics will not be available for container %s: %s", cont.info.Name, err)
		}
	}

	// Application-metrics collectors (binary-only; the kubelet leaves the factory
	// nil). The readFile closure lets the collector factory read config files
	// (declared via container labels) from inside the container.
	if CollectorManagerFactory != nil {
		cm, cerr := CollectorManagerFactory(handler, func(p string) ([]byte, error) {
			return cont.ReadFile(p, m.inHostNamespace)
		})
		if cerr != nil {
			klog.V(4).Infof("Failed to set up application-metrics collectors for %q: %v", containerName, cerr)
		} else if cm != nil {
			cont.collectorManager = cm
			// Refresh the spec so custom-metric definitions appear in it.
			if specErr := cont.updateSpec(); specErr != nil {
				klog.V(4).Infof("Failed to refresh spec after wiring collectors for %q: %v", containerName, specErr)
			}
		}
	}

	// Add the container name and all its aliases. The aliases must be within the namespace of the factory.
	m.containers.Store(namespacedName, cont)
	for _, alias := range cont.info.Aliases {
		m.containers.Store(namespacedContainerName{
			Namespace: cont.info.Namespace,
			Name:      alias,
		}, cont)
	}

	klog.V(3).Infof("Added container: %q (aliases: %v, namespace: %q)", containerName, cont.info.Aliases, cont.info.Namespace)

	// Surface the container-creation event to the sink (no-op if unset).
	if m.eventSink != nil {
		var creationTime time.Time
		if spec, specErr := cont.handler.GetSpec(); specErr == nil {
			creationTime = spec.CreationTime
		}
		m.addEvent(&info.Event{
			ContainerName: cont.info.Name,
			Timestamp:     creationTime,
			EventType:     info.EventContainerCreation,
		})
	}

	// Start the container's housekeeping.
	return cont.Start()
}

func (m *manager) destroyContainer(containerName string) error {
	namespacedName := namespacedContainerName{
		Name: containerName,
	}
	cont, ok := m.containers.Load(namespacedName)
	if !ok {
		// Already destroyed, done.
		return nil
	}

	exitCode, err := cont.handler.GetExitCode()
	if err != nil {
		klog.V(4).Infof("Could not retrieve exit code for container %q: %v (using -1)", containerName, err)
		exitCode = -1
	}

	err = cont.Stop()
	if err != nil {
		return err
	}

	// Remove the container from our records (and all its aliases).
	m.containers.Delete(namespacedName)
	for _, alias := range cont.info.Aliases {
		m.containers.Delete(namespacedContainerName{
			Namespace: cont.info.Namespace,
			Name:      alias,
		})
	}
	klog.V(3).Infof("Destroyed container: %q (aliases: %v, namespace: %q, exit_code: %d)", containerName, cont.info.Aliases, cont.info.Namespace, exitCode)

	// Surface the container-deletion event to the sink (no-op if unset),
	// carrying the exit code (the /events API and its tests rely on it).
	m.addEvent(&info.Event{
		ContainerName: cont.info.Name,
		Timestamp:     time.Now(),
		EventType:     info.EventContainerDeletion,
		EventData: info.EventData{
			ContainerDeletion: &info.ContainerDeletionEventData{
				ExitCode: exitCode,
			},
		},
	})

	return nil
}

// Detect all containers that have been added or deleted from the specified container.
func (m *manager) getContainersDiff(containerName string) (added []info.ContainerReference, removed []info.ContainerReference, err error) {
	// Get all subcontainers recursively.
	cont, ok := m.containers.Load(namespacedContainerName{Name: containerName})
	if !ok {
		return nil, nil, fmt.Errorf("failed to find container %q while checking for new containers", containerName)
	}
	allContainers, err := cont.handler.ListContainers(container.ListRecursive)

	if err != nil {
		return nil, nil, err
	}
	allContainers = append(allContainers, info.ContainerReference{Name: containerName})

	// Determine which were added and which were removed.
	allContainersSet := make(map[string]*containerData)
	m.containers.Range(func(name namespacedContainerName, cont *containerData) bool {
		if cont == nil {
			return true
		}
		// Only add the canonical name.
		if cont.info.Name == name.Name {
			allContainersSet[name.Name] = cont
		}
		return true
	})

	// Added containers
	for _, c := range allContainers {
		delete(allContainersSet, c.Name)
		_, ok := m.containers.Load(namespacedContainerName{Name: c.Name})
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
func (m *manager) watchForNewContainers(quit chan error) error {
	watched := make([]watcher.ContainerWatcher, 0)
	for _, watcher := range m.containerWatchers {
		err := watcher.Start(m.eventsChannel)
		if err != nil {
			for _, w := range watched {
				stopErr := w.Stop()
				if stopErr != nil {
					klog.Warningf("Failed to stop wacher %v with error: %v", w, stopErr)
				}
			}
			return err
		}
		watched = append(watched, watcher)
	}

	// There is a race between starting the watch and new container creation so we do a detection before we read new containers.
	err := m.detectSubcontainers("/")
	if err != nil {
		return err
	}

	// Listen to events from the container handler.
	go func() {
		for {
			select {
			case event := <-m.eventsChannel:
				switch {
				case event.EventType == watcher.ContainerAdd:
					switch event.WatchSource {
					default:
						err = m.createContainer(event.Name, event.WatchSource)
					}
				case event.EventType == watcher.ContainerDelete:
					err = m.destroyContainer(event.Name)
				}
				if err != nil {
					klog.Warningf("Failed to process watch event %+v: %v", event, err)
				}
			case <-quit:
				var errs partialFailure

				// Stop processing events if asked to quit.
				for i, watcher := range m.containerWatchers {
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

func (m *manager) getFsInfoByDeviceName(deviceName string) (info.FsInfo, error) {
	mountPoint, err := m.fsInfo.GetMountpointForDevice(deviceName)
	if err != nil {
		return info.FsInfo{}, fmt.Errorf("failed to get mount point for device %q: %v", deviceName, err)
	}
	infos, err := m.GetFsInfo("")
	if err != nil {
		return info.FsInfo{}, err
	}
	for _, info := range infos {
		if info.Mountpoint == mountPoint {
			return info, nil
		}
	}
	return info.FsInfo{}, fmt.Errorf("cannot find filesystem info for device %q", deviceName)
}

func getVersionInfo() (*info.VersionInfo, error) {

	kernelVersion := machine.KernelVersion()
	osVersion := machine.ContainerOsVersion()

	return &info.VersionInfo{
		KernelVersion:      kernelVersion,
		ContainerOsVersion: osVersion,
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
