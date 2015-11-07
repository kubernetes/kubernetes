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

// Page for /containers/
package pages

import (
	"fmt"
	"html/template"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/manager"
)

const ContainersPage = "/containers/"

// from http://golang.org/doc/effective_go.html#constants
type ByteSize float64

const (
	_ = iota
	// KB - kilobyte
	KB ByteSize = 1 << (10 * iota)
	// MB - megabyte
	MB
	// GB - gigabyte
	GB
	// TB - terabyte
	TB
	// PB - petabyte
	PB
	// EB - exabyte
	EB
	// ZB - zettabyte
	ZB
	// YB - yottabyte
	YB
)

func (b ByteSize) Size() string {
	for _, i := range [...]ByteSize{YB, ZB, EB, PB, TB, GB, MB, KB} {
		if b >= i {
			return fmt.Sprintf("%.2f", b/i)
		}
	}
	return fmt.Sprintf("%.2f", b)
}

func (b ByteSize) Unit() string {
	switch {
	case b >= YB:
		return "YB"
	case b >= ZB:
		return "ZB"
	case b >= EB:
		return "EB"
	case b >= PB:
		return "PB"
	case b >= TB:
		return "TB"
	case b >= GB:
		return "GB"
	case b >= MB:
		return "MB"
	case b >= KB:
		return "KB"
	}
	return "B"
}

var funcMap = template.FuncMap{
	"printMask":   printMask,
	"printCores":  printCores,
	"printShares": printShares,
	"printSize":   printSize,
	"printUnit":   printUnit,
}

func printMask(mask string, numCores int) interface{} {
	masks := make([]string, numCores)
	activeCores := getActiveCores(mask)
	for i := 0; i < numCores; i++ {
		coreClass := "inactive-cpu"
		if activeCores[i] {
			coreClass = "active-cpu"
		}
		masks[i] = fmt.Sprintf("<span class=\"%s\">%d</span>", coreClass, i)
	}
	return template.HTML(strings.Join(masks, "&nbsp;"))
}

func getActiveCores(mask string) map[int]bool {
	activeCores := make(map[int]bool)
	for _, corebits := range strings.Split(mask, ",") {
		cores := strings.Split(corebits, "-")
		if len(cores) == 1 {
			index, err := strconv.Atoi(cores[0])
			if err != nil {
				// Ignore malformed strings.
				continue
			}
			activeCores[index] = true
		} else if len(cores) == 2 {
			start, err := strconv.Atoi(cores[0])
			if err != nil {
				continue
			}
			end, err := strconv.Atoi(cores[1])
			if err != nil {
				continue
			}
			for i := start; i <= end; i++ {
				activeCores[i] = true
			}
		}
	}
	return activeCores
}

func printCores(millicores *uint64) string {
	cores := float64(*millicores) / 1000
	return strconv.FormatFloat(cores, 'f', 3, 64)
}

func printShares(shares *uint64) string {
	return fmt.Sprintf("%d", *shares)
}

func toMegabytes(bytes uint64) float64 {
	return float64(bytes) / (1 << 20)
}

// Size after which we consider memory to be "unlimited". This is not
// MaxInt64 due to rounding by the kernel.
const maxMemorySize = uint64(1 << 62)

func printSize(bytes uint64) string {
	if bytes >= maxMemorySize {
		return "unlimited"
	}
	return ByteSize(bytes).Size()
}

func printUnit(bytes uint64) string {
	if bytes >= maxMemorySize {
		return ""
	}
	return ByteSize(bytes).Unit()
}

func toMemoryPercent(usage uint64, spec *info.ContainerSpec, machine *info.MachineInfo) int {
	// Saturate limit to the machine size.
	limit := uint64(spec.Memory.Limit)
	if limit > uint64(machine.MemoryCapacity) {
		limit = uint64(machine.MemoryCapacity)
	}

	return int((usage * 100) / limit)
}

func serveContainersPage(m manager.Manager, w http.ResponseWriter, u *url.URL) error {
	start := time.Now()

	// The container name is the path after the handler
	containerName := u.Path[len(ContainersPage)-1:]

	// Get the container.
	reqParams := info.ContainerInfoRequest{
		NumStats: 60,
	}
	cont, err := m.GetContainerInfo(containerName, &reqParams)
	if err != nil {
		return fmt.Errorf("failed to get container %q with error: %v", containerName, err)
	}
	displayName := getContainerDisplayName(cont.ContainerReference)

	// Get the MachineInfo
	machineInfo, err := m.GetMachineInfo()
	if err != nil {
		return err
	}

	rootDir := getRootDir(containerName)

	// Make a list of the parent containers and their links
	pathParts := strings.Split(string(cont.Name), "/")
	parentContainers := make([]link, 0, len(pathParts))
	parentContainers = append(parentContainers, link{
		Text: "root",
		Link: path.Join(rootDir, ContainersPage),
	})
	for i := 1; i < len(pathParts); i++ {
		// Skip empty parts.
		if pathParts[i] == "" {
			continue
		}
		parentContainers = append(parentContainers, link{
			Text: pathParts[i],
			Link: path.Join(rootDir, ContainersPage, path.Join(pathParts[1:i+1]...)),
		})
	}

	// Build the links for the subcontainers.
	subcontainerLinks := make([]link, 0, len(cont.Subcontainers))
	for _, sub := range cont.Subcontainers {
		if !m.Exists(sub.Name) {
			continue
		}
		subcontainerLinks = append(subcontainerLinks, link{
			Text: getContainerDisplayName(sub),
			Link: path.Join(rootDir, ContainersPage, sub.Name),
		})
	}

	data := &pageData{
		DisplayName:            displayName,
		ContainerName:          escapeContainerName(cont.Name),
		ParentContainers:       parentContainers,
		Subcontainers:          subcontainerLinks,
		Spec:                   cont.Spec,
		Stats:                  cont.Stats,
		MachineInfo:            machineInfo,
		IsRoot:                 cont.Name == "/",
		ResourcesAvailable:     cont.Spec.HasCpu || cont.Spec.HasMemory || cont.Spec.HasNetwork || cont.Spec.HasFilesystem,
		CpuAvailable:           cont.Spec.HasCpu,
		MemoryAvailable:        cont.Spec.HasMemory,
		NetworkAvailable:       cont.Spec.HasNetwork,
		FsAvailable:            cont.Spec.HasFilesystem,
		CustomMetricsAvailable: cont.Spec.HasCustomMetrics,
		Root: rootDir,
	}
	err = pageTemplate.Execute(w, data)
	if err != nil {
		glog.Errorf("Failed to apply template: %s", err)
	}

	glog.V(5).Infof("Request took %s", time.Since(start))
	return nil
}

// Build a relative path to the root of the container page.
func getRootDir(containerName string) string {
	// The root is at: container depth
	levels := (strings.Count(containerName, "/"))
	return strings.Repeat("../", levels)
}
