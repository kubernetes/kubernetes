// Copyright 2015 Google Inc. All Rights Reserved.
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

package api

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/golang/glog"
	"github.com/google/cadvisor/events"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/manager"
)

const (
	containersApi    = "containers"
	subcontainersApi = "subcontainers"
	machineApi       = "machine"
	dockerApi        = "docker"
	summaryApi       = "summary"
	statsApi         = "stats"
	specApi          = "spec"
	eventsApi        = "events"
	storageApi       = "storage"
)

// Interface for a cAdvisor API version
type ApiVersion interface {
	// Returns the version string.
	Version() string

	// List of supported API endpoints.
	SupportedRequestTypes() []string

	// Handles a request. The second argument is the parameters after /api/<version>/<endpoint>
	HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error
}

// Gets all supported API versions.
func getApiVersions() []ApiVersion {
	v1_0 := &version1_0{}
	v1_1 := newVersion1_1(v1_0)
	v1_2 := newVersion1_2(v1_1)
	v1_3 := newVersion1_3(v1_2)
	v2_0 := newVersion2_0(v1_3)

	return []ApiVersion{v1_0, v1_1, v1_2, v1_3, v2_0}

}

// API v1.0

type version1_0 struct {
}

func (self *version1_0) Version() string {
	return "v1.0"
}

func (self *version1_0) SupportedRequestTypes() []string {
	return []string{containersApi, machineApi}
}

func (self *version1_0) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	switch requestType {
	case machineApi:
		glog.V(2).Infof("Api - Machine")

		// Get the MachineInfo
		machineInfo, err := m.GetMachineInfo()
		if err != nil {
			return err
		}

		err = writeResult(machineInfo, w)
		if err != nil {
			return err
		}
	case containersApi:
		containerName := getContainerName(request)
		glog.V(2).Infof("Api - Container(%s)", containerName)

		// Get the query request.
		query, err := getContainerInfoRequest(r.Body)
		if err != nil {
			return err
		}

		// Get the container.
		cont, err := m.GetContainerInfo(containerName, query)
		if err != nil {
			return fmt.Errorf("failed to get container %q with error: %s", containerName, err)
		}

		// Only output the container as JSON.
		err = writeResult(cont, w)
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown request type %q", requestType)
	}
	return nil
}

// API v1.1

type version1_1 struct {
	baseVersion *version1_0
}

// v1.1 builds on v1.0.
func newVersion1_1(v *version1_0) *version1_1 {
	return &version1_1{
		baseVersion: v,
	}
}

func (self *version1_1) Version() string {
	return "v1.1"
}

func (self *version1_1) SupportedRequestTypes() []string {
	return append(self.baseVersion.SupportedRequestTypes(), subcontainersApi)
}

func (self *version1_1) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	switch requestType {
	case subcontainersApi:
		containerName := getContainerName(request)
		glog.V(2).Infof("Api - Subcontainers(%s)", containerName)

		// Get the query request.
		query, err := getContainerInfoRequest(r.Body)
		if err != nil {
			return err
		}

		// Get the subcontainers.
		containers, err := m.SubcontainersInfo(containerName, query)
		if err != nil {
			return fmt.Errorf("failed to get subcontainers for container %q with error: %s", containerName, err)
		}

		// Only output the containers as JSON.
		err = writeResult(containers, w)
		if err != nil {
			return err
		}
		return nil
	default:
		return self.baseVersion.HandleRequest(requestType, request, m, w, r)
	}
}

// API v1.2

type version1_2 struct {
	baseVersion *version1_1
}

// v1.2 builds on v1.1.
func newVersion1_2(v *version1_1) *version1_2 {
	return &version1_2{
		baseVersion: v,
	}
}

func (self *version1_2) Version() string {
	return "v1.2"
}

func (self *version1_2) SupportedRequestTypes() []string {
	return append(self.baseVersion.SupportedRequestTypes(), dockerApi)
}

func (self *version1_2) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	switch requestType {
	case dockerApi:
		glog.V(2).Infof("Api - Docker(%v)", request)

		// Get the query request.
		query, err := getContainerInfoRequest(r.Body)
		if err != nil {
			return err
		}

		var containers map[string]info.ContainerInfo
		// map requests for "docker/" to "docker"
		if len(request) == 1 && len(request[0]) == 0 {
			request = request[:0]
		}
		switch len(request) {
		case 0:
			// Get all Docker containers.
			containers, err = m.AllDockerContainers(query)
			if err != nil {
				return fmt.Errorf("failed to get all Docker containers with error: %v", err)
			}
		case 1:
			// Get one Docker container.
			var cont info.ContainerInfo
			cont, err = m.DockerContainer(request[0], query)
			if err != nil {
				return fmt.Errorf("failed to get Docker container %q with error: %v", request[0], err)
			}
			containers = map[string]info.ContainerInfo{
				cont.Name: cont,
			}
		default:
			return fmt.Errorf("unknown request for Docker container %v", request)
		}

		// Only output the containers as JSON.
		err = writeResult(containers, w)
		if err != nil {
			return err
		}
		return nil
	default:
		return self.baseVersion.HandleRequest(requestType, request, m, w, r)
	}
}

// API v1.3

type version1_3 struct {
	baseVersion *version1_2
}

// v1.3 builds on v1.2.
func newVersion1_3(v *version1_2) *version1_3 {
	return &version1_3{
		baseVersion: v,
	}
}

func (self *version1_3) Version() string {
	return "v1.3"
}

func (self *version1_3) SupportedRequestTypes() []string {
	return append(self.baseVersion.SupportedRequestTypes(), eventsApi)
}

func (self *version1_3) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	switch requestType {
	case eventsApi:
		query, eventsFromAllTime, err := getEventRequest(r)
		if err != nil {
			return err
		}
		glog.V(2).Infof("Api - Events(%v)", query)
		if eventsFromAllTime {
			pastEvents, err := m.GetPastEvents(query)
			if err != nil {
				return err
			}
			return writeResult(pastEvents, w)
		}
		eventsChannel := make(chan *events.Event, 10)
		err = m.WatchForEvents(query, eventsChannel)
		if err != nil {
			return err
		}
		return streamResults(eventsChannel, w, r)
	default:
		return self.baseVersion.HandleRequest(requestType, request, m, w, r)
	}
}

// API v2.0

// v2.0 builds on v1.3
type version2_0 struct {
	baseVersion *version1_3
}

func newVersion2_0(v *version1_3) *version2_0 {
	return &version2_0{
		baseVersion: v,
	}
}

func (self *version2_0) Version() string {
	return "v2.0"
}

func (self *version2_0) SupportedRequestTypes() []string {
	return append(self.baseVersion.SupportedRequestTypes(), summaryApi)
}

func (self *version2_0) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	switch requestType {
	case summaryApi:
		containerName := getContainerName(request)
		glog.V(2).Infof("Api - Summary(%v)", containerName)

		stats, err := m.GetContainerDerivedStats(containerName)
		if err != nil {
			return err
		}

		return writeResult(stats, w)
	case statsApi:
		name := getContainerName(request)
		sr, err := getStatsRequest(name, r)
		if err != nil {
			return err
		}
		glog.V(2).Infof("Api - Stats: Looking for stats for container %q, options %+v", name, sr)
		query := info.ContainerInfoRequest{
			NumStats: sr.Count,
		}
		cont, err := m.GetContainerInfo(name, &query)
		if err != nil {
			return fmt.Errorf("failed to get container %q: %v", name, err)
		}
		contStats := convertStats(cont)
		return writeResult(contStats, w)
	case specApi:
		containerName := getContainerName(request)
		glog.V(2).Infof("Api - Spec(%v)", containerName)
		spec, err := m.GetContainerSpec(containerName)
		if err != nil {
			return err
		}
		specV2 := convertSpec(spec)
		return writeResult(specV2, w)
	case storageApi:
		var err error
		fi := []v2.FsInfo{}
		label := r.URL.Query().Get("label")
		if len(label) == 0 {
			// Get all global filesystems info.
			fi, err = m.GetFsInfo("")
			if err != nil {
				return err
			}
		} else {
			// Get a specific label.
			fi, err = m.GetFsInfo(label)
			if err != nil {
				return err
			}
		}
		return writeResult(fi, w)
	default:
		return self.baseVersion.HandleRequest(requestType, request, m, w, r)
	}
}

// Convert container spec from v1 to v2.
func convertSpec(specV1 info.ContainerSpec) v2.ContainerSpec {
	specV2 := v2.ContainerSpec{
		CreationTime: specV1.CreationTime,
		HasCpu:       specV1.HasCpu,
		HasMemory:    specV1.HasMemory,
	}
	if specV1.HasCpu {
		specV2.Cpu.Limit = specV1.Cpu.Limit
		specV2.Cpu.MaxLimit = specV1.Cpu.MaxLimit
		specV2.Cpu.Mask = specV1.Cpu.Mask
	}
	if specV1.HasMemory {
		specV2.Memory.Limit = specV1.Memory.Limit
		specV2.Memory.Reservation = specV1.Memory.Reservation
		specV2.Memory.SwapLimit = specV1.Memory.SwapLimit
	}
	return specV2
}

func convertStats(cont *info.ContainerInfo) []v2.ContainerStats {
	stats := []v2.ContainerStats{}
	for _, val := range cont.Stats {
		stat := v2.ContainerStats{
			Timestamp:     val.Timestamp,
			HasCpu:        cont.Spec.HasCpu,
			HasMemory:     cont.Spec.HasMemory,
			HasNetwork:    cont.Spec.HasNetwork,
			HasFilesystem: cont.Spec.HasFilesystem,
			HasDiskIo:     cont.Spec.HasDiskIo,
		}
		if stat.HasCpu {
			stat.Cpu = val.Cpu
		}
		if stat.HasMemory {
			stat.Memory = val.Memory
		}
		if stat.HasNetwork {
			// TODO(rjnagal): Return stats about all network interfaces.
			stat.Network = append(stat.Network, val.Network)
		}
		if stat.HasFilesystem {
			stat.Filesystem = val.Filesystem
		}
		if stat.HasDiskIo {
			stat.DiskIo = val.DiskIo
		}
		// TODO(rjnagal): Handle load stats.
		stats = append(stats, stat)
	}
	return stats
}

func getStatsRequest(id string, r *http.Request) (v2.StatsRequest, error) {
	// fill in the defaults.
	sr := v2.StatsRequest{
		IdType:    "name",
		Count:     64,
		Recursive: false,
	}
	idType := r.URL.Query().Get("type")
	if len(idType) != 0 && idType != "name" {
		return sr, fmt.Errorf("unknown 'type' %q for container name %q", idType, id)
	}
	count := r.URL.Query().Get("count")
	if len(count) != 0 {
		n, err := strconv.ParseUint(count, 10, 32)
		if err != nil {
			return sr, fmt.Errorf("failed to parse 'count' option: %v", count)
		}
		sr.Count = int(n)
	}
	// TODO(rjnagal): Add option to specify recursive.
	return sr, nil
}
