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

// This file is dedicated to heapster running outside of kubernetes. Heapster
// will poll a file to get the hosts that it needs to monitor and will collect
// stats from cadvisor running on those hosts.

package sources

import (
	"fmt"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/extpoints"
	"k8s.io/heapster/sinks/cache"
	"k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/datasource"
	"k8s.io/heapster/sources/nodes"
)

const (
	defaultCadvisorPort = 8080
	defaultStandalone   = false
	defaultHostsFile    = "/var/run/heapster/hosts"
)

var defaultFleetEndpoints = []string{"http://127.0.0.1:4001"}

type cadvisorSource struct {
	cadvisorPort int
	cadvisorApi  datasource.Cadvisor
	nodesApi     nodes.NodesApi
}

func init() {
	extpoints.SourceFactories.Register(NewCadvisorSources, "cadvisor")
}

func (self *cadvisorSource) GetInfo(start, end time.Time) (api.AggregateData, error) {
	var (
		lock sync.Mutex
		wg   sync.WaitGroup
	)
	nodeList, err := self.nodesApi.List()
	if err != nil {
		return api.AggregateData{}, err
	}

	result := api.AggregateData{}
	for hostname, info := range nodeList.Items {
		wg.Add(1)
		go func(hostname string, info nodes.Info) {
			defer wg.Done()
			host := datasource.Host{
				IP:   info.InternalIP,
				Port: self.cadvisorPort,
			}
			rawSubcontainers, node, err := self.cadvisorApi.GetAllContainers(host, start, end)
			if err != nil {
				glog.Error(err)
				return
			}
			subcontainers := []api.Container{}
			for _, cont := range rawSubcontainers {
				if cont != nil {
					cont.Hostname = hostname
					subcontainers = append(subcontainers, *cont)
				}
			}
			lock.Lock()
			defer lock.Unlock()
			result.Containers = append(result.Containers, subcontainers...)
			if node != nil {
				node.Hostname = hostname
				result.Machine = append(result.Machine, *node)
			}
		}(string(hostname), info)
	}
	wg.Wait()

	return result, nil
}

func (self *cadvisorSource) DebugInfo() string {
	desc := "Source type: Cadvisor\n"
	// TODO(rjnagal): Cache config?
	nodeList, err := self.nodesApi.List()
	if err != nil {
		desc += fmt.Sprintf("\tFailed to read host config: %s", err)
	}
	desc += fmt.Sprintf("\tNodeList: %+v\n", *nodeList)
	desc += fmt.Sprintf("\t%s\n", self.nodesApi.DebugInfo())
	desc += "\n"
	return desc
}

func (cs *cadvisorSource) Name() string {
	return "Cadvisor Source"
}

func NewCadvisorSources(uri *url.URL, _ cache.Cache) ([]api.Source, error) {
	switch uri.Path {
	case "coreos", "fleet":
		return newCoreosSources(uri.Query())
	case "external":
		return newExternalSources(uri.Query())
	default:
		return nil, fmt.Errorf("Unknown cadvisor source: %s", uri.Path)
	}
}

func newExternalSources(options map[string][]string) ([]api.Source, error) {
	hostsFile := defaultHostsFile
	if len(options["hostsFile"]) > 0 {
		hostsFile = options["hostsFile"][0]
	}
	standalone := defaultStandalone
	if len(options["standalone"]) > 0 {
		standaloneOption, err := strconv.ParseBool(options["standalone"][0])
		if err != nil {
			return nil, err
		}
		standalone = standaloneOption
	}

	nodesApi, err := nodes.NewExternalNodes(standalone, hostsFile)
	if err != nil {
		return nil, err
	}

	cadvisorPort := defaultCadvisorPort
	if len(options["cadvisorPort"]) > 0 {
		cadvisorPort, err = strconv.Atoi(options["cadvisorPort"][0])
		if err != nil {
			return nil, err
		}
	}

	return []api.Source{
		&cadvisorSource{
			cadvisorApi:  datasource.NewCadvisor(),
			nodesApi:     nodesApi,
			cadvisorPort: cadvisorPort,
		},
	}, nil
}

func newCoreosSources(options map[string][]string) ([]api.Source, error) {
	fleetEndpoints := defaultFleetEndpoints
	if len(options["fleetEndpoint"]) > 0 {
		fleetEndpoints = options["fleetEndpoint"]
	}

	nodesApi, err := nodes.NewCoreOSNodes(fleetEndpoints)
	if err != nil {
		return nil, err
	}

	cadvisorPort := defaultCadvisorPort
	if len(options["cadvisorPort"]) > 0 {
		cadvisorPort, err = strconv.Atoi(options["cadvisorPort"][0])
		if err != nil {
			return nil, err
		}
	}

	return []api.Source{
		&cadvisorSource{
			cadvisorApi:  datasource.NewCadvisor(),
			nodesApi:     nodesApi,
			cadvisorPort: cadvisorPort,
		},
	}, nil
}
