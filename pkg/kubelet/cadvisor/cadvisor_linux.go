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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/google/cadvisor/cache/memory"
	"github.com/google/cadvisor/events"
	cadvisorFs "github.com/google/cadvisor/fs"
	cadvisorHttp "github.com/google/cadvisor/http"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiV2 "github.com/google/cadvisor/info/v2"
	"github.com/google/cadvisor/manager"
	"github.com/google/cadvisor/utils/sysfs"
)

type cadvisorClient struct {
	manager.Manager
}

var _ Interface = new(cadvisorClient)

// TODO(vmarmol): Make configurable.
// The amount of time for which to keep stats in memory.
const statsCacheDuration = 2 * time.Minute

// Creates a cAdvisor and exports its API on the specified port if port > 0.
func New(port uint) (Interface, error) {
	sysFs, err := sysfs.NewRealSysFs()
	if err != nil {
		return nil, err
	}

	// Create and start the cAdvisor container manager.
	m, err := manager.New(memory.New(statsCacheDuration, nil), sysFs)
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

func (cc *cadvisorClient) Start() error {
	return cc.Manager.Start()
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

func (cc *cadvisorClient) SubcontainerInfo(name string, req *cadvisorApi.ContainerInfoRequest) (map[string]*cadvisorApi.ContainerInfo, error) {
	infos, err := cc.SubcontainersInfo(name, req)
	if err != nil {
		return nil, err
	}

	result := make(map[string]*cadvisorApi.ContainerInfo, len(infos))
	for _, info := range infos {
		result[info.Name] = info
	}
	return result, nil
}

func (cc *cadvisorClient) MachineInfo() (*cadvisorApi.MachineInfo, error) {
	return cc.GetMachineInfo()
}

func (cc *cadvisorClient) DockerImagesFsInfo() (cadvisorApiV2.FsInfo, error) {
	return cc.getFsInfo(cadvisorFs.LabelDockerImages)
}

func (cc *cadvisorClient) RootFsInfo() (cadvisorApiV2.FsInfo, error) {
	return cc.getFsInfo(cadvisorFs.LabelSystemRoot)
}

func (cc *cadvisorClient) getFsInfo(label string) (cadvisorApiV2.FsInfo, error) {
	res, err := cc.GetFsInfo(label)
	if err != nil {
		return cadvisorApiV2.FsInfo{}, err
	}
	if len(res) == 0 {
		return cadvisorApiV2.FsInfo{}, fmt.Errorf("failed to find information for the filesystem labeled %q", label)
	}
	// TODO(vmarmol): Handle this better when a label has more than one image filesystem.
	if len(res) > 1 {
		glog.Warningf("More than one filesystem labeled %q: %#v. Only using the first one", label, res)
	}

	return res[0], nil
}

func (cc *cadvisorClient) WatchEvents(request *events.Request) (*events.EventChannel, error) {
	return cc.WatchForEvents(request)
}
