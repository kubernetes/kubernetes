// +build cgo,linux

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package cadvisor

import (
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/events"
	cadvisorhttp "github.com/google/cadvisor/http"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager"
	"github.com/google/cadvisor/utils/sysfs"
	"k8s.io/kubernetes/pkg/kubelet/collector"
	"k8s.io/kubernetes/pkg/util"
)

type cadvisorCollector struct {
	client ClientInterface

	// Whether or not to use built-in cadvisor
	useBuiltin bool
}

var _ collector.Interface = new(cadvisorCollector)

// TODO(vmarmol): Make configurable.
// The amount of time for which to keep stats in memory.
const statsCacheDuration = 2 * time.Minute
const maxHousekeepingInterval = 15 * time.Second
const allowDynamicHousekeeping = true

func NewCadvisorCollector(connectionURL string) (*cadvisorCollector, error) {
	cc := &cadvisorCollector{}

	// Checks if only <port> is specified in connectionURL
	matched := regexp.MustCompile(`^\d+$`).MatchString(connectionURL)
	if matched {
		// Creates a cAdvisor and exports its API on the specified port if port > 0.
		port, err := strconv.ParseUint(connectionURL, 10, 32)
		if err != nil {
			return cc, err
		}

		sysFs, err := sysfs.NewRealSysFs()
		if err != nil {
			return cc, err
		}

		// Create and start the cAdvisor container manager.
		m, err := manager.New(memory.New(statsCacheDuration, nil), sysFs, maxHousekeepingInterval, allowDynamicHousekeeping)
		if err != nil {
			return cc, err
		}
		cc.client = m

		err = cc.exportHTTP(uint(port))
		if err != nil {
			return cc, err
		}
		cc.useBuiltin = true

	} else {

		// Checks if [<hostname>|<ip>]:<port> is specified in connectionURL
		matchedVars := regexp.MustCompile(`^https*://(?P<host>[0-9a-zA-Z\.]+):(?P<port>\d+)$`).FindStringSubmatch(connectionURL)
		if matchedVars != nil {
			var err error
			cc.client, err = NewClient(connectionURL)
			if err != nil {
				return cc, err
			}
			cc.useBuiltin = false

		} else {
			// connectionURL does not match any patterns cAdvisor driver supports
			return cc, fmt.Errorf("Cannot parse connection URL (%s)", connectionURL)
		}
	}
	return cc, nil
}

func (cc *cadvisorCollector) Start() error {
	if cc.useBuiltin {
		return cc.Start()
	} else {
		return nil
	}
}

func (cc *cadvisorCollector) exportHTTP(port uint) error {
	// Register the handlers regardless as this registers the prometheus
	// collector properly.
	mux := http.NewServeMux()
	err := cadvisorhttp.RegisterHandlers(mux, cc.client.(manager.Manager), "", "", "", "")
	if err != nil {
		return err
	}

	re := regexp.MustCompile(`^k8s_(?P<kubernetes_container_name>[^_\.]+)[^_]+_(?P<kubernetes_pod_name>[^_]+)_(?P<kubernetes_namespace>[^_]+)`)
	reCaptureNames := re.SubexpNames()
	cadvisorhttp.RegisterPrometheusHandler(mux, cc.client.(manager.Manager), "/metrics", func(name string) map[string]string {
		extraLabels := map[string]string{}
		matches := re.FindStringSubmatch(name)
		for i, match := range matches {
			if len(reCaptureNames[i]) > 0 {
				extraLabels[re.SubexpNames()[i]] = match
			}
		}
		return extraLabels
	})

	// Only start the http server if port > 0
	if port > 0 {
		serv := &http.Server{
			Addr:    fmt.Sprintf(":%d", port),
			Handler: mux,
		}

		// TODO(vmarmol): Remove this when the cAdvisor port is once again free.
		// If export failed, retry in the background until we are able to bind.
		// This allows an existing cAdvisor to be killed before this one registers.
		go func() {
			defer util.HandleCrash()

			err := serv.ListenAndServe()
			for err != nil {
				glog.Infof("Failed to register cAdvisor on port %d, retrying. Error: %v", port, err)
				time.Sleep(time.Minute)
				err = serv.ListenAndServe()
			}
		}()
	}

	return nil
}

func (cc *cadvisorCollector) MachineInfo() (*collector.MachineInfo, error) {
	machineInfo, err := cc.client.GetMachineInfo()
	if err != nil {
		return nil, err
	}

	ret := collector.MachineInfo{
		NumCores:       machineInfo.NumCores,
		MemoryCapacity: machineInfo.MemoryCapacity,
		MachineID:      machineInfo.MachineID,
		SystemUUID:     machineInfo.SystemUUID,
		BootID:         machineInfo.BootID,
	}

	return &ret, nil
}

func (cc *cadvisorCollector) VersionInfo() (*collector.VersionInfo, error) {
	versionInfo, err := cc.client.GetVersionInfo()
	if err != nil {
		return nil, err
	}

	ret := collector.VersionInfo{
		KernelVersion:      versionInfo.KernelVersion,
		ContainerOsVersion: versionInfo.ContainerOsVersion,
		DockerVersion:      versionInfo.DockerVersion,
		CollectorVersion:   versionInfo.CadvisorVersion,
	}

	return &ret, nil
}

func (cc *cadvisorCollector) FsInfo(fsLabel string) (*collector.FsInfo, error) {
	fsInfo, err := cc.client.GetFsInfo(fsLabel)
	if err != nil {
		return nil, err
	}
	if len(fsInfo) == 0 {
		return nil, fmt.Errorf("failed to find information for the filesystem labeled %q", fsLabel)
	}
	// TODO(vmarmol): Handle this better when a label has more than one image filesystem.
	if len(fsInfo) > 1 {
		glog.Warningf("More than one filesystem labeled %q: %#v. Only using the first one", fsLabel, fsInfo)
	}

	//return fsInfo[0], nil
	ret := collector.FsInfo{
		Device:     fsInfo[0].Device,
		Mountpoint: fsInfo[0].Mountpoint,
		Capacity:   fsInfo[0].Capacity,
		Available:  fsInfo[0].Available,
		Usage:      fsInfo[0].Usage,
		Labels:     fsInfo[0].Labels,
	}
	return &ret, nil
}

func (cc *cadvisorCollector) WatchEvents(request *collector.Request) (chan *collector.Event, error) {

	cadvisorReq := events.Request{
		ContainerName:        request.ContainerName,
		EventType:            map[cadvisorapi.EventType]bool{},
		IncludeSubcontainers: request.IncludeSubcontainers,
		MaxEventsReturned:    10,
	}

	for k := range request.EventType {
		switch k {
		case collector.EventOom:
			cadvisorReq.EventType[cadvisorapi.EventOom] = true
		case collector.EventOomKill:
			cadvisorReq.EventType[cadvisorapi.EventOomKill] = true
		case collector.EventContainerCreation:
			cadvisorReq.EventType[cadvisorapi.EventContainerCreation] = true
		case collector.EventContainerDeletion:
			cadvisorReq.EventType[cadvisorapi.EventContainerDeletion] = true
		default:
			return nil, fmt.Errorf("Request EventType (%v) not recognized", k)
		}
	}

	cadvisorChan, err := cc.client.WatchForEvents(&cadvisorReq)
	if err != nil {
		return nil, err
	}

	retChan := make(chan *collector.Event)
	go func() {
		defer util.HandleCrash()

		for cadvisorEvent := range cadvisorChan.GetChannel() {
			event := collector.Event{
				ContainerName: cadvisorEvent.ContainerName,
				Timestamp:     cadvisorEvent.Timestamp,
			}
			switch cadvisorEvent.EventType {
			case cadvisorapi.EventOom:
				event.EventType = collector.EventOom
			case cadvisorapi.EventOomKill:
				event.EventType = collector.EventOomKill
			case cadvisorapi.EventContainerCreation:
				event.EventType = collector.EventContainerCreation
			case cadvisorapi.EventContainerDeletion:
				event.EventType = collector.EventContainerDeletion
			default:
				glog.Warningf("EventType (%v) not recognized", cadvisorEvent.EventType)
			}
			retChan <- &event
		}
	}()
	return retChan, nil
}

func (cc *cadvisorCollector) ContainerInfo(containerName string, req *collector.ContainerInfoRequest, subcontainers bool, isRawContainer bool) (map[string]interface{}, error) {

	cadvisorReq := &cadvisorapi.ContainerInfoRequest{
		NumStats: req.NumStats,
		Start:    req.Start,
		End:      req.End,
	}

	if isRawContainer {
		if subcontainers {
			infos, err := cc.client.SubcontainersInfo(containerName, cadvisorReq)
			if err != nil {
				return nil, err
			}

			result := make(map[string]interface{}, len(infos))
			for _, info := range infos {
				result[info.Name] = info
			}
			return result, nil

		} else {
			containerInfo, err := cc.client.GetContainerInfo(containerName, cadvisorReq)
			if err != nil {
				return nil, err
			}

			return map[string]interface{}{
				containerInfo.Name: containerInfo,
			}, nil
		}

	} else {
		containerInfo, err := cc.client.DockerContainer(containerName, cadvisorReq)
		if err != nil {
			return nil, err
		}

		return map[string]interface{}{
			containerInfo.Name: containerInfo,
		}, nil
	}
}
