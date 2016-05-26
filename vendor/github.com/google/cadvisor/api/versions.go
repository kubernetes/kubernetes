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
	"path"
	"strconv"

	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/manager"

	"github.com/golang/glog"
)

const (
	containersApi    = "containers"
	subcontainersApi = "subcontainers"
	machineApi       = "machine"
	machineStatsApi  = "machinestats"
	dockerApi        = "docker"
	summaryApi       = "summary"
	statsApi         = "stats"
	specApi          = "spec"
	eventsApi        = "events"
	storageApi       = "storage"
	attributesApi    = "attributes"
	versionApi       = "version"
	psApi            = "ps"
	customMetricsApi = "appmetrics"
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
	v2_0 := newVersion2_0()
	v2_1 := newVersion2_1(v2_0)

	return []ApiVersion{v1_0, v1_1, v1_2, v1_3, v2_0, v2_1}

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
		glog.V(4).Infof("Api - Machine")

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
		glog.V(4).Infof("Api - Container(%s)", containerName)

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
		glog.V(4).Infof("Api - Subcontainers(%s)", containerName)

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
		glog.V(4).Infof("Api - Docker(%v)", request)

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
		return handleEventRequest(request, m, w, r)
	default:
		return self.baseVersion.HandleRequest(requestType, request, m, w, r)
	}
}

func handleEventRequest(request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	query, stream, err := getEventRequest(r)
	if err != nil {
		return err
	}
	query.ContainerName = path.Join("/", getContainerName(request))
	glog.V(4).Infof("Api - Events(%v)", query)
	if !stream {
		pastEvents, err := m.GetPastEvents(query)
		if err != nil {
			return err
		}
		return writeResult(pastEvents, w)
	}
	eventChannel, err := m.WatchForEvents(query)
	if err != nil {
		return err
	}
	return streamResults(eventChannel, w, r, m)

}

// API v2.0

type version2_0 struct {
}

func newVersion2_0() *version2_0 {
	return &version2_0{}
}

func (self *version2_0) Version() string {
	return "v2.0"
}

func (self *version2_0) SupportedRequestTypes() []string {
	return []string{versionApi, attributesApi, eventsApi, machineApi, summaryApi, statsApi, specApi, storageApi, psApi, customMetricsApi}
}

func (self *version2_0) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	opt, err := getRequestOptions(r)
	if err != nil {
		return err
	}
	switch requestType {
	case versionApi:
		glog.V(4).Infof("Api - Version")
		versionInfo, err := m.GetVersionInfo()
		if err != nil {
			return err
		}
		return writeResult(versionInfo.CadvisorVersion, w)
	case attributesApi:
		glog.V(4).Info("Api - Attributes")

		machineInfo, err := m.GetMachineInfo()
		if err != nil {
			return err
		}
		versionInfo, err := m.GetVersionInfo()
		if err != nil {
			return err
		}
		info := v2.GetAttributes(machineInfo, versionInfo)
		return writeResult(info, w)
	case machineApi:
		glog.V(4).Info("Api - Machine")

		// TODO(rjnagal): Move machineInfo from v1.
		machineInfo, err := m.GetMachineInfo()
		if err != nil {
			return err
		}
		return writeResult(machineInfo, w)
	case summaryApi:
		containerName := getContainerName(request)
		glog.V(4).Infof("Api - Summary for container %q, options %+v", containerName, opt)

		stats, err := m.GetDerivedStats(containerName, opt)
		if err != nil {
			return err
		}
		return writeResult(stats, w)
	case statsApi:
		name := getContainerName(request)
		glog.V(4).Infof("Api - Stats: Looking for stats for container %q, options %+v", name, opt)
		infos, err := m.GetRequestedContainersInfo(name, opt)
		if err != nil {
			if len(infos) == 0 {
				return err
			}
			glog.Errorf("Error calling GetRequestedContainersInfo: %v", err)
		}
		contStats := make(map[string][]v2.DeprecatedContainerStats, 0)
		for name, cinfo := range infos {
			contStats[name] = v2.DeprecatedStatsFromV1(cinfo)
		}
		return writeResult(contStats, w)
	case customMetricsApi:
		containerName := getContainerName(request)
		glog.V(4).Infof("Api - Custom Metrics: Looking for metrics for container %q, options %+v", containerName, opt)
		infos, err := m.GetContainerInfoV2(containerName, opt)
		if err != nil {
			return err
		}
		contMetrics := make(map[string]map[string]map[string][]info.MetricValBasic, 0)
		for _, cinfo := range infos {
			metrics := make(map[string]map[string][]info.MetricValBasic, 0)
			for _, contStat := range cinfo.Stats {
				if len(contStat.CustomMetrics) == 0 {
					continue
				}
				for name, allLabels := range contStat.CustomMetrics {
					metricLabels := make(map[string][]info.MetricValBasic, 0)
					for _, metric := range allLabels {
						if !metric.Timestamp.IsZero() {
							metVal := info.MetricValBasic{
								Timestamp:  metric.Timestamp,
								IntValue:   metric.IntValue,
								FloatValue: metric.FloatValue,
							}
							labels := metrics[name]
							if labels != nil {
								values := labels[metric.Label]
								values = append(values, metVal)
								labels[metric.Label] = values
								metrics[name] = labels
							} else {
								metricLabels[metric.Label] = []info.MetricValBasic{metVal}
								metrics[name] = metricLabels
							}
						}
					}
				}
			}
			contMetrics[containerName] = metrics
		}
		return writeResult(contMetrics, w)
	case specApi:
		containerName := getContainerName(request)
		glog.V(4).Infof("Api - Spec for container %q, options %+v", containerName, opt)
		specs, err := m.GetContainerSpec(containerName, opt)
		if err != nil {
			return err
		}
		return writeResult(specs, w)
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
	case eventsApi:
		return handleEventRequest(request, m, w, r)
	case psApi:
		// reuse container type from request.
		// ignore recursive.
		// TODO(rjnagal): consider count to limit ps output.
		name := getContainerName(request)
		glog.V(4).Infof("Api - Spec for container %q, options %+v", name, opt)
		ps, err := m.GetProcessList(name, opt)
		if err != nil {
			return fmt.Errorf("process listing failed: %v", err)
		}
		return writeResult(ps, w)
	default:
		return fmt.Errorf("unknown request type %q", requestType)
	}
}

type version2_1 struct {
	baseVersion *version2_0
}

func newVersion2_1(v *version2_0) *version2_1 {
	return &version2_1{
		baseVersion: v,
	}
}

func (self *version2_1) Version() string {
	return "v2.1"
}

func (self *version2_1) SupportedRequestTypes() []string {
	return append([]string{machineStatsApi}, self.baseVersion.SupportedRequestTypes()...)
}

func (self *version2_1) HandleRequest(requestType string, request []string, m manager.Manager, w http.ResponseWriter, r *http.Request) error {
	// Get the query request.
	opt, err := getRequestOptions(r)
	if err != nil {
		return err
	}

	switch requestType {
	case machineStatsApi:
		glog.V(4).Infof("Api - MachineStats(%v)", request)
		cont, err := m.GetRequestedContainersInfo("/", opt)
		if err != nil {
			if len(cont) == 0 {
				return err
			}
			glog.Errorf("Error calling GetRequestedContainersInfo: %v", err)
		}
		return writeResult(v2.MachineStatsFromV1(cont["/"]), w)
	case statsApi:
		name := getContainerName(request)
		glog.V(4).Infof("Api - Stats: Looking for stats for container %q, options %+v", name, opt)
		conts, err := m.GetRequestedContainersInfo(name, opt)
		if err != nil {
			if len(conts) == 0 {
				return err
			}
			glog.Errorf("Error calling GetRequestedContainersInfo: %v", err)
		}
		contStats := make(map[string]v2.ContainerInfo, len(conts))
		for name, cont := range conts {
			if name == "/" {
				// Root cgroup stats should be exposed as machine stats
				continue
			}
			contStats[name] = v2.ContainerInfo{
				Spec:  v2.ContainerSpecFromV1(&cont.Spec, cont.Aliases, cont.Namespace),
				Stats: v2.ContainerStatsFromV1(&cont.Spec, cont.Stats),
			}
		}
		return writeResult(contStats, w)
	default:
		return self.baseVersion.HandleRequest(requestType, request, m, w, r)
	}
}

func getRequestOptions(r *http.Request) (v2.RequestOptions, error) {
	supportedTypes := map[string]bool{
		v2.TypeName:   true,
		v2.TypeDocker: true,
	}
	// fill in the defaults.
	opt := v2.RequestOptions{
		IdType:    v2.TypeName,
		Count:     64,
		Recursive: false,
	}
	idType := r.URL.Query().Get("type")
	if len(idType) != 0 {
		if !supportedTypes[idType] {
			return opt, fmt.Errorf("unknown 'type' %q", idType)
		}
		opt.IdType = idType
	}
	count := r.URL.Query().Get("count")
	if len(count) != 0 {
		n, err := strconv.ParseUint(count, 10, 32)
		if err != nil {
			return opt, fmt.Errorf("failed to parse 'count' option: %v", count)
		}
		opt.Count = int(n)
	}
	recursive := r.URL.Query().Get("recursive")
	if recursive == "true" {
		opt.Recursive = true
	}
	return opt, nil
}
