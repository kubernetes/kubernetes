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

	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/container/common"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"golang.org/x/net/context"

	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
	"k8s.io/klog"
)

type rktContainerHandler struct {
	machineInfoFactory info.MachineInfoFactory

	// Absolute path to the cgroup hierarchies of this container.
	// (e.g.: "cpu" -> "/sys/fs/cgroup/cpu/test")
	cgroupPaths map[string]string

	fsInfo fs.FsInfo

	isPod bool

	rootfsStorageDir string

	// Filesystem handler.
	fsHandler common.FsHandler

	includedMetrics container.MetricSet

	apiPod *rktapi.Pod

	labels map[string]string

	reference info.ContainerReference

	libcontainerHandler *libcontainer.Handler
}

func newRktContainerHandler(name string, rktClient rktapi.PublicAPIClient, rktPath string, cgroupSubsystems *libcontainer.CgroupSubsystems, machineInfoFactory info.MachineInfoFactory, fsInfo fs.FsInfo, rootFs string, includedMetrics container.MetricSet) (container.ContainerHandler, error) {
	aliases := make([]string, 1)
	isPod := false

	apiPod := &rktapi.Pod{}

	parsed, err := parseName(name)
	if err != nil {
		return nil, fmt.Errorf("this should be impossible!, new handler failing, but factory allowed, name = %s", name)
	}

	// rktnetes uses containerID: rkt://fff40827-b994-4e3a-8f88-6427c2c8a5ac:nginx
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
			klog.Warningf("couldn't find app %v in pod", parsed.Container)
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

	libcontainerHandler := libcontainer.NewHandler(cgroupManager, rootFs, pid, includedMetrics)

	rootfsStorageDir := getRootFs(rktPath, parsed)

	containerReference := info.ContainerReference{
		Name:      name,
		Aliases:   aliases,
		Namespace: RktNamespace,
	}

	handler := &rktContainerHandler{
		machineInfoFactory:  machineInfoFactory,
		cgroupPaths:         cgroupPaths,
		fsInfo:              fsInfo,
		isPod:               isPod,
		rootfsStorageDir:    rootfsStorageDir,
		includedMetrics:     includedMetrics,
		apiPod:              apiPod,
		labels:              labels,
		reference:           containerReference,
		libcontainerHandler: libcontainerHandler,
	}

	if includedMetrics.Has(container.DiskUsageMetrics) {
		handler.fsHandler = common.NewFsHandler(common.DefaultPeriod, rootfsStorageDir, "", fsInfo)
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
	return handler.reference, nil
}

func (handler *rktContainerHandler) Start() {
	handler.fsHandler.Start()
}

func (handler *rktContainerHandler) Cleanup() {
	handler.fsHandler.Stop()
}

func (handler *rktContainerHandler) GetSpec() (info.ContainerSpec, error) {
	hasNetwork := handler.isPod && handler.includedMetrics.Has(container.NetworkUsageMetrics)
	hasFilesystem := handler.includedMetrics.Has(container.DiskUsageMetrics)

	spec, err := common.GetSpec(handler.cgroupPaths, handler.machineInfoFactory, hasNetwork, hasFilesystem)

	spec.Labels = handler.labels

	return spec, err
}

func (handler *rktContainerHandler) getFsStats(stats *info.ContainerStats) error {
	mi, err := handler.machineInfoFactory.GetMachineInfo()
	if err != nil {
		return err
	}

	if handler.includedMetrics.Has(container.DiskIOMetrics) {
		common.AssignDeviceNamesToDiskStats((*common.MachineInfoNamer)(mi), &stats.DiskIo)
	}

	if !handler.includedMetrics.Has(container.DiskUsageMetrics) {
		return nil
	}

	deviceInfo, err := handler.fsInfo.GetDirFsDevice(handler.rootfsStorageDir)
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

	usage := handler.fsHandler.Usage()
	fsStat.BaseUsage = usage.BaseUsageBytes
	fsStat.Usage = usage.TotalUsageBytes
	fsStat.Inodes = usage.InodeUsage

	stats.Filesystem = append(stats.Filesystem, fsStat)

	return nil
}

func (handler *rktContainerHandler) GetStats() (*info.ContainerStats, error) {
	stats, err := handler.libcontainerHandler.GetStats()
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
		return "", fmt.Errorf("could not find path for resource %q for container %q\n", resource, handler.reference.Name)
	}
	return path, nil
}

func (handler *rktContainerHandler) GetContainerLabels() map[string]string {
	return handler.labels
}

func (handler *rktContainerHandler) ListContainers(listType container.ListType) ([]info.ContainerReference, error) {
	return common.ListContainers(handler.reference.Name, handler.cgroupPaths, listType)
}

func (handler *rktContainerHandler) ListProcesses(listType container.ListType) ([]int, error) {
	return handler.libcontainerHandler.GetProcesses()
}

func (handler *rktContainerHandler) Exists() bool {
	return common.CgroupExists(handler.cgroupPaths)
}

func (handler *rktContainerHandler) Type() container.ContainerType {
	return container.ContainerTypeRkt
}
