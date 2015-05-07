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
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/container/docker"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager"
)

const DockerPage = "/docker/"

func serveDockerPage(m manager.Manager, w http.ResponseWriter, u *url.URL) error {
	start := time.Now()

	// The container name is the path after the handler
	containerName := u.Path[len(DockerPage):]
	rootDir := getRootDir(u.Path)

	var data *pageData
	if containerName == "" {
		// Get the containers.
		reqParams := info.ContainerInfoRequest{
			NumStats: 0,
		}
		conts, err := m.AllDockerContainers(&reqParams)
		if err != nil {
			return fmt.Errorf("failed to get container %q with error: %v", containerName, err)
		}
		subcontainers := make([]link, 0, len(conts))
		for _, cont := range conts {
			subcontainers = append(subcontainers, link{
				Text: getContainerDisplayName(cont.ContainerReference),
				Link: path.Join("/docker", docker.ContainerNameToDockerId(cont.ContainerReference.Name)),
			})
		}

		dockerContainersText := "Docker Containers"
		data = &pageData{
			DisplayName: dockerContainersText,
			ParentContainers: []link{
				{
					Text: dockerContainersText,
					Link: DockerPage,
				}},
			Subcontainers: subcontainers,
			Root:          rootDir,
		}
	} else {
		// Get the container.
		reqParams := info.ContainerInfoRequest{
			NumStats: 60,
		}
		cont, err := m.DockerContainer(containerName, &reqParams)
		if err != nil {
			return fmt.Errorf("failed to get container %q with error: %v", containerName, err)
		}
		displayName := getContainerDisplayName(cont.ContainerReference)

		// Make a list of the parent containers and their links
		var parentContainers []link
		parentContainers = append(parentContainers, link{
			Text: "Docker containers",
			Link: DockerPage,
		})
		parentContainers = append(parentContainers, link{
			Text: displayName,
			Link: path.Join(DockerPage, docker.ContainerNameToDockerId(cont.Name)),
		})

		// Get the MachineInfo
		machineInfo, err := m.GetMachineInfo()
		if err != nil {
			return err
		}

		data = &pageData{
			DisplayName:        displayName,
			ContainerName:      cont.Name,
			ParentContainers:   parentContainers,
			Spec:               cont.Spec,
			Stats:              cont.Stats,
			MachineInfo:        machineInfo,
			ResourcesAvailable: cont.Spec.HasCpu || cont.Spec.HasMemory || cont.Spec.HasNetwork,
			CpuAvailable:       cont.Spec.HasCpu,
			MemoryAvailable:    cont.Spec.HasMemory,
			NetworkAvailable:   cont.Spec.HasNetwork,
			FsAvailable:        cont.Spec.HasFilesystem,
			Root:               rootDir,
		}
	}

	err := pageTemplate.Execute(w, data)
	if err != nil {
		glog.Errorf("Failed to apply template: %s", err)
	}

	glog.V(5).Infof("Request took %s", time.Since(start))
	return nil
}
