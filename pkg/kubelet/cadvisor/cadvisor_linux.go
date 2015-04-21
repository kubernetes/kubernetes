// +build cgo,linux

/*
Copyright 2015 Google Inc. All rights reserved.

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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/google/cadvisor/events"
	cadvisorFs "github.com/google/cadvisor/fs"
	cadvisorHttp "github.com/google/cadvisor/http"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiV2 "github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/manager"
	"github.com/google/cadvisor/storage/memory"
	"github.com/google/cadvisor/utils/sysfs"
)

type cadvisorClient struct {
	manager.Manager
}

var _ Interface = new(cadvisorClient)

// TODO(vmarmol): Make configurable.
// The number of stats to keep in memory.
const statsToCache = 60

// Creates a cAdvisor and exports its API on the specified port if port > 0.
func New(port uint) (Interface, error) {
	sysFs, err := sysfs.NewRealSysFs()
	if err != nil {
		return nil, err
	}

	// Create and start the cAdvisor container manager.
	m, err := manager.New(memory.New(statsToCache, nil), sysFs)
	if err != nil {
		return nil, err
	}
	err = m.Start()
	if err != nil {
		return nil, err
	}

	cadvisorClient := &cadvisorClient{
		Manager: m,
	}

	// Export the HTTP endpoint if a port was specified.
	if port > 0 {
		err = cadvisorClient.exportHTTP(port)
		if err != nil {
			return nil, err
		}
	}

	return cadvisorClient, nil
}

func (cc *cadvisorClient) exportHTTP(port uint) error {
	mux := http.NewServeMux()
	err := cadvisorHttp.RegisterHandlers(mux, cc, "", "", "", "", "/metrics")
	if err != nil {
		return err
	}

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

	return nil
}

func (cc *cadvisorClient) ContainerInfo(name string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	return cc.GetContainerInfo(name, req)
}

func (cc *cadvisorClient) VersionInfo() (*cadvisorApi.VersionInfo, error) {
	return cc.GetVersionInfo()
}

func (cc *cadvisorClient) MachineInfo() (*cadvisorApi.MachineInfo, error) {
	return cc.GetMachineInfo()
}

func (cc *cadvisorClient) DockerImagesFsInfo() (cadvisorApiV2.FsInfo, error) {
	res, err := cc.GetFsInfo(cadvisorFs.LabelDockerImages)
	if err != nil {
		return cadvisorApiV2.FsInfo{}, err
	}
	if len(res) == 0 {
		return cadvisorApiV2.FsInfo{}, fmt.Errorf("failed to find information for the filesystem containing Docker images")
	}
	// TODO(vmarmol): Handle this better when Docker has more than one image filesystem.
	if len(res) > 1 {
		glog.Warningf("More than one Docker images filesystem: %#v. Only using the first one", res)
	}

	return res[0], nil
}

func (cc *cadvisorClient) GetPastEvents(request *events.Request) ([]*cadvisorApi.Event, error) {
	return cc.GetPastEvents(request)
}
