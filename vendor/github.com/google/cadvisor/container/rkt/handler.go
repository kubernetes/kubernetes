// Copyright 2016 Google Inc. All Rights Reserved.
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

// Handler for "rkt" containers.
package rkt

import (
	"fmt"
	"os"
	"time"

	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"golang.org/x/net/context"

	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
)

type rktContainerHandler struct {
	rktClient rktapi.PublicAPIClient
	// Name of the container for this handler.
	name               string
	cgroupSubsystems   *libcontainer.CgroupSubsystems
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	// Manager of this container's cgroups.
	cgroupManager cgroups.Manager

	// Whether this container has network isolation enabled.
	hasNetwork bool

	fsInfo fs.FsInfo

	rootFs string

	isPod bool

	aliases []string

	pid int

	rootfsStorageDir string

	labels map[string]string

	// Filesystem handler.
	fsHandler common.FsHandler

	ignoreMetrics container.MetricSet

	apiPod *rktapi.Pod
}

func newRktContainerHandler(name string, rktClient rktapi.PublicAPIClient, rktPath string, cgroupSubsystems *libcontainer.CgroupSubsystems, machineInfoFactory info.MachineInfoFactory, fsInfo fs.FsInfo, rootFs string, ignoreMetrics container.MetricSet) (container.ContainerHandler, error) {
	aliases := make([]string, 1)
	isPod := false

	apiPod := &rktapi.Pod{}

	parsed, err := parseName(name)
	if err != nil {
		return nil, fmt.Errorf("this should be impossible!, new handler failing, but factory allowed, name = %s", name)
	}

	//rktnetes uses containerID: rkt://fff40827-b994-4e3a-8f88-6427c2c8a5ac:nginx
	if parsed.Container == "" {
		isPod = true
		aliases = append(aliases, "rkt://"+parsed.Pod)
	} else {
		aliases = append(aliases, "rkt://"+parsed.Pod+":"+parsed.Container)
	}

	pid := os.Getpid()
	labels := make(map[string]string)
	resp, err := rktClient.InspectPod(context.Background(), &rktapi.InspectPodRequest{
		Id: parsed.Pod,
	})
	if err != nil {
		return nil, err
	}
	annotations := resp.Pod.Annotations
	if parsed.Container != "" { // As not empty string, an App container
		if contAnnotations, ok := findAnnotations(resp.Pod.Apps, parsed.Container); !ok {
			glog.Warningf("couldn't find app %v in pod", parsed.Container)
		} else {
			annotations = append(annotations, contAnnotations...)
		}
	} else { // The Pod container
		pid = int(resp.Pod.Pid)
		apiPod = resp.Pod
	}
	labels = createLabels(annotations)

	cgroupPaths := common.MakeCgroupPaths(cgroupSubsystems.MountPoints, name)

	// Generate the equivalent cgroup manager for this container.
	cgroupManager := &cgroupfs.Manager{
		Cgroups: &configs.Cgroup{
			Name: name,
		},
		Paths: cgroupPaths,
	}

	hasNetwork := false
	if isPod {
		hasNetwork = true
	}

	rootfsStorageDir := getRootFs(rktPath, parsed)

	handler := &rktContainerHandler{
		name:               name,
		rktClient:          rktClient,
		cgroupSubsystems:   cgroupSubsystems,
		machineInfoFactory: machineInfoFactory,
		cgroupPaths:        cgroupPaths,
		cgroupManager:      cgroupManager,
		fsInfo:             fsInfo,
		hasNetwork:         hasNetwork,
		rootFs:             rootFs,
		isPod:              isPod,
		aliases:            aliases,
		pid:                pid,
		labels:             labels,
		rootfsStorageDir:   rootfsStorageDir,
		ignoreMetrics:      ignoreMetrics,
		apiPod:             apiPod,
	}

	if !ignoreMetrics.Has(container.DiskUsageMetrics) {
		handler.fsHandler = common.NewFsHandler(time.Minute, rootfsStorageDir, "", fsInfo)
	}

	return handler, nil
}

func findAnnotations(apps []*rktapi.App, container string) ([]*rktapi.KeyValue, bool) {
	for _, app := range apps {
		if app.Name == container {
			return app.Annotations, true
		}
	}
	return nil, false
}

func createLabels(annotations []*rktapi.KeyValue) map[string]string {
	labels := make(map[string]string)
	for _, kv := range annotations {
		labels[kv.Key] = kv.Value
	}

	return labels
}

func (handler *rktContainerHandler) ContainerReference() (info.ContainerReference, error) {
	return info.ContainerReference{
		Name:      handler.name,
		Aliases:   handler.aliases,
		Namespace: RktNamespace,
		Labels:    handler.labels,
	}, nil
}

func (handler *rktContainerHandler) Start() {
	handler.fsHandler.Start()
}

func (handler *rktContainerHandler) Cleanup() {
	handler.fsHandler.Stop()
}

func (handler *rktContainerHandler) GetSpec() (info.ContainerSpec, error) {
	hasNetwork := handler.hasNetwork && !handler.ignoreMetrics.Has(container.NetworkUsageMetrics)
	hasFilesystem := !handler.ignoreMetrics.Has(container.DiskUsageMetrics)

	spec, err := common.GetSpec(handler.cgroupPaths, handler.machineInfoFactory, hasNetwork, hasFilesystem)

	spec.Labels = handler.labels

	return spec, err
}

func (handler *rktContainerHandler) getFsStats(stats *info.ContainerStats) error {
	if handler.ignoreMetrics.Has(container.DiskUsageMetrics) {
		return nil
	}

	deviceInfo, err := handler.fsInfo.GetDirFsDevice(handler.rootfsStorageDir)
	if err != nil {
		return err
	}

	mi, err := handler.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}
	var limit uint64 = 0

	// Use capacity as limit.
	for _, fs := range mi.Filesystems {
		if fs.Device == deviceInfo.Device {
			limit = fs.Capacity
			break
		}
	}

	fsStat := info.FsStats{Device: deviceInfo.Device, Limit: limit}

	fsStat.BaseUsage, fsStat.Usage = handler.fsHandler.Usage()

	stats.Filesystem = append(stats.Filesystem, fsStat)

	return nil
}

func (handler *rktContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := libcontainer.GetStats(handler.cgroupManager, handler.rootFs, handler.pid, handler.ignoreMetrics)
	if err != nil {
		return stats, err
	}

	// Get filesystem stats.
	err = handler.getFsStats(stats)
	if err != nil {
		return stats, err
	}

	return stats, nil
}

func (self *rktContainerHandler) GetContainerIPAddress() string {
	// attempt to return the ip address of the pod
	// if a specific ip address of the pod could not be determined, return the system ip address
	if self.isPod && len(self.apiPod.Networks) > 0 {
		address := self.apiPod.Networks[0].Ipv4
		if address != "" {
			return address
		} else {
			return self.apiPod.Networks[0].Ipv6
		}
	} else {
		return "127.0.0.1"
	}
}

func (handler *rktContainerHandler) GetCgroupPath(resource string) (string, error) {
	path, ok := handler.cgroupPaths[resource]
	if !ok {
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, handler.name)
	}
	return path, nil
}

func (handler *rktContainerHandler) GetContainerLabels() map[string]string {
	return handler.labels
}

func (handler *rktContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return common.ListContainers(handler.name, handler.cgroupPaths, listType)
}

func (handler *rktContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return libcontainer.GetProcesses(handler.cgroupManager)
}

func (handler *rktContainerHandler) Exists() bool {
	return common.CgroupExists(handler.cgroupPaths)
}

func (handler *rktContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeRkt
}
