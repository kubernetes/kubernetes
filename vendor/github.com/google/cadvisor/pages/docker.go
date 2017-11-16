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

package pages

import (
	"fmt"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"time"

	"github.com/google/cadvisor/container/docker"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager"

	"github.com/golang/glog"
)

const DockerPage = "/docker/"

func toStatusKV(status info.DockerStatus) ([]keyVal, []keyVal) {
	ds := []keyVal{
		{Key: "Driver", Value: status.Driver},
	}
	for k, v := range status.DriverStatus {
		ds = append(ds, keyVal{Key: k, Value: v})
	}
	return []keyVal{
		{Key: "Docker Version", Value: status.Version},
		{Key: "Docker API Version", Value: status.APIVersion},
		{Key: "Kernel Version", Value: status.KernelVersion},
		{Key: "OS Version", Value: status.OS},
		{Key: "Host Name", Value: status.Hostname},
		{Key: "Docker Root Directory", Value: status.RootDir},
		{Key: "Execution  Driver", Value: status.ExecDriver},
		{Key: "Number of Images", Value: strconv.Itoa(status.NumImages)},
		{Key: "Number of Containers", Value: strconv.Itoa(status.NumContainers)},
	}, ds
}

func serveDockerPage(m manager.Manager, w http.ResponseWriter, u *url.URL) {
	start := time.Now()

	// The container name is the path after the handler
	containerName := u.Path[len(DockerPage)-1:]
	rootDir := getRootDir(containerName)

	var data *pageData
	if containerName == "/" {
		// Get the containers.
		reqParams := info.ContainerInfoRequest{
			NumStats: 0,
		}
		conts, err := m.AllDockerContainers(&reqParams)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get container %q with error: %v", containerName, err), http.StatusNotFound)
			return
		}
		subcontainers := make([]link, 0, len(conts))
		for _, cont := range conts {
			subcontainers = append(subcontainers, link{
				Text: getContainerDisplayName(cont.ContainerReference),
				Link: path.Join(rootDir, DockerPage, docker.ContainerNameToDockerId(cont.ContainerReference.Name)),
			})
		}

		// Get Docker status
		status, err := m.DockerInfo()
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get docker info: %v", err), http.StatusInternalServerError)
			return
		}

		dockerStatus, driverStatus := toStatusKV(status)
		// Get Docker Images
		images, err := m.DockerImages()
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get docker images: %v", err), http.StatusInternalServerError)
			return
		}

		dockerContainersText := "Docker Containers"
		data = &pageData{
			DisplayName: dockerContainersText,
			ParentContainers: []link{
				{
					Text: dockerContainersText,
					Link: path.Join(rootDir, DockerPage),
				}},
			Subcontainers:      subcontainers,
			Root:               rootDir,
			DockerStatus:       dockerStatus,
			DockerDriverStatus: driverStatus,
			DockerImages:       images,
		}
	} else {
		// Get the container.
		reqParams := info.ContainerInfoRequest{
			NumStats: 60,
		}
		cont, err := m.DockerContainer(containerName[1:], &reqParams)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get container %q with error: %v", containerName, err), http.StatusNotFound)
			return
		}
		displayName := getContainerDisplayName(cont.ContainerReference)

		// Make a list of the parent containers and their links
		var parentContainers []link
		parentContainers = append(parentContainers, link{
			Text: "Docker Containers",
			Link: path.Join(rootDir, DockerPage),
		})
		parentContainers = append(parentContainers, link{
			Text: displayName,
			Link: path.Join(rootDir, DockerPage, docker.ContainerNameToDockerId(cont.Name)),
		})

		// Get the MachineInfo
		machineInfo, err := m.GetMachineInfo()
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to get machine info: %v", err), http.StatusInternalServerError)
			return
		}
		data = &pageData{
			DisplayName:            displayName,
			ContainerName:          escapeContainerName(cont.Name),
			ParentContainers:       parentContainers,
			Spec:                   cont.Spec,
			Stats:                  cont.Stats,
			MachineInfo:            machineInfo,
			ResourcesAvailable:     cont.Spec.HasCpu || cont.Spec.HasMemory || cont.Spec.HasNetwork,
			CpuAvailable:           cont.Spec.HasCpu,
			MemoryAvailable:        cont.Spec.HasMemory,
			NetworkAvailable:       cont.Spec.HasNetwork,
			FsAvailable:            cont.Spec.HasFilesystem,
			CustomMetricsAvailable: cont.Spec.HasCustomMetrics,
			Root: rootDir,
		}
	}

	err := pageTemplate.Execute(w, data)
	if err != nil {
		glog.Errorf("Failed to apply template: %s", err)
	}

	glog.V(5).Infof("Request took %s", time.Since(start))
	return
}
