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
	"log"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/manager"
)

const ContainersPage = "/containers/"

var funcMap = template.FuncMap{
	"containerLink":         containerLink,
	"printMask":             printMask,
	"printCores":            printCores,
	"printMegabytes":        printMegabytes,
	"getMemoryUsage":        getMemoryUsage,
	"getMemoryUsagePercent": getMemoryUsagePercent,
	"getHotMemoryPercent":   getHotMemoryPercent,
	"getColdMemoryPercent":  getColdMemoryPercent,
}

// TODO(vmarmol): Consider housekeeping Spec too so we can show changes through time. We probably don't need it ever second though.

var pageTemplate *template.Template

type pageData struct {
	ContainerName      string
	ParentContainers   []info.ContainerReference
	Subcontainers      []info.ContainerReference
	Spec               *info.ContainerSpec
	Stats              []*info.ContainerStats
	MachineInfo        *info.MachineInfo
	ResourcesAvailable bool
	CpuAvailable       bool
	MemoryAvailable    bool
}

func init() {
	pageTemplate = template.New("containersTemplate").Funcs(funcMap)
	_, err := pageTemplate.Parse(containersHtmlTemplate)
	if err != nil {
		log.Fatalf("Failed to parse template: %s", err)
	}
}

// TODO(vmarmol): Escape this correctly.
func containerLink(container info.ContainerReference, basenameOnly bool, cssClasses string) interface{} {
	var displayName string
	containerName := container.Name
	if len(container.Aliases) > 0 {
		displayName = container.Aliases[0]
	} else if basenameOnly {
		displayName = path.Base(string(container.Name))
	} else {
		displayName = string(container.Name)
	}
	if container.Name == "root" {
		containerName = "/"
	} else if strings.Contains(container.Name, " ") {
		// If it has a space, it is an a.k.a, so keep the base-name
		containerName = container.Name[:strings.Index(container.Name, " ")]
	}
	return template.HTML(fmt.Sprintf("<a class=\"%s\" href=\"%s%s\">%s</a>", cssClasses, ContainersPage[:len(ContainersPage)-1], containerName, displayName))
}

func printMask(mask *info.CpuSpecMask, numCores int) interface{} {
	// TODO(vmarmol): Detect this correctly.
	// TODO(vmarmol): Support more than 64 cores.
	rawMask := uint64(0)
	if len(mask.Data) > 0 {
		rawMask = mask.Data[0]
	}
	masks := make([]string, numCores)
	for i := uint(0); i < uint(numCores); i++ {
		coreClass := "inactive-cpu"
		// by default, all cores are active
		if ((0x1<<i)&rawMask) != 0 || len(mask.Data) == 0 {
			coreClass = "active-cpu"
		}
		masks[i] = fmt.Sprintf("<span class=\"%s\">%d</span>", coreClass, i)
	}
	return template.HTML(strings.Join(masks, "&nbsp;"))
}

func printCores(millicores *uint64) string {
	// TODO(vmarmol): Detect this correctly
	if *millicores > 1024*1000 {
		return "unlimited"
	}
	cores := float64(*millicores) / 1000
	return strconv.FormatFloat(cores, 'f', 3, 64)
}

func toMegabytes(bytes uint64) float64 {
	return float64(bytes) / (1 << 20)
}

func printMegabytes(bytes uint64) string {
	// TODO(vmarmol): Detect this correctly
	if bytes > (100 << 30) {
		return "unlimited"
	}
	megabytes := toMegabytes(bytes)
	return strconv.FormatFloat(megabytes, 'f', 3, 64)
}

func toMemoryPercent(usage uint64, spec *info.ContainerSpec, machine *info.MachineInfo) int {
	// Saturate limit to the machine size.
	limit := uint64(spec.Memory.Limit)
	if limit > uint64(machine.MemoryCapacity) {
		limit = uint64(machine.MemoryCapacity)
	}

	return int((usage * 100) / limit)
}

func getMemoryUsage(stats []*info.ContainerStats) string {
	return strconv.FormatFloat(toMegabytes((stats[len(stats)-1].Memory.Usage)), 'f', 2, 64)
}

func getMemoryUsagePercent(spec *info.ContainerSpec, stats []*info.ContainerStats, machine *info.MachineInfo) int {
	return toMemoryPercent((stats[len(stats)-1].Memory.Usage), spec, machine)
}

func getHotMemoryPercent(spec *info.ContainerSpec, stats []*info.ContainerStats, machine *info.MachineInfo) int {
	return toMemoryPercent((stats[len(stats)-1].Memory.WorkingSet), spec, machine)
}

func getColdMemoryPercent(spec *info.ContainerSpec, stats []*info.ContainerStats, machine *info.MachineInfo) int {
	latestStats := stats[len(stats)-1].Memory
	return toMemoryPercent((latestStats.Usage)-(latestStats.WorkingSet), spec, machine)
}

func ServerContainersPage(m manager.Manager, w http.ResponseWriter, u *url.URL) error {
	start := time.Now()

	// The container name is the path after the handler
	containerName := u.Path[len(ContainersPage)-1:]

	// Get the container.
	cont, err := m.GetContainerInfo(containerName)
	if err != nil {
		return fmt.Errorf("Failed to get container \"%s\" with error: %s", containerName, err)
	}

	// Get the MachineInfo
	machineInfo, err := m.GetMachineInfo()
	if err != nil {
		return err
	}

	// Make a list of the parent containers and their links
	var parentContainers []info.ContainerReference
	parentContainers = append(parentContainers, info.ContainerReference{Name: "root"})
	parentName := ""
	for _, part := range strings.Split(string(cont.Name), "/") {
		if part == "" {
			continue
		}
		parentName += "/" + part
		parentContainers = append(parentContainers, info.ContainerReference{Name: parentName})
	}

	// Pick the shortest name of the container as the display name.
	displayName := cont.Name
	for _, alias := range cont.Aliases {
		if len(displayName) >= len(alias) {
			displayName = alias
		}
	}

	// Replace the last part of the parent containers with the displayName.
	if displayName != cont.Name {
		parentContainers[len(parentContainers)-1] = info.ContainerReference{
			Name: fmt.Sprintf("%s (%s)", displayName, path.Base(cont.Name)),
		}
	}

	data := &pageData{
		ContainerName: displayName,
		// TODO(vmarmol): Only use strings for this.
		ParentContainers:   parentContainers,
		Subcontainers:      cont.Subcontainers,
		Spec:               cont.Spec,
		Stats:              cont.Stats,
		MachineInfo:        machineInfo,
		ResourcesAvailable: cont.Spec.Cpu != nil || cont.Spec.Memory != nil,
		CpuAvailable:       cont.Spec.Cpu != nil,
		MemoryAvailable:    cont.Spec.Memory != nil,
	}
	err = pageTemplate.Execute(w, data)
	if err != nil {
		log.Printf("Failed to apply template: %s", err)
	}

	log.Printf("Request took %s", time.Since(start))
	return nil
}
