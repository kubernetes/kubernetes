/*
   Copyright The containerd Authors.

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

package cgroups

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	v1 "github.com/containerd/cgroups/stats/v1"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

type rdmaController struct {
	root string
}

func (p *rdmaController) Name() Name {
	return Rdma
}

func (p *rdmaController) Path(path string) string {
	return filepath.Join(p.root, path)
}

func NewRdma(root string) *rdmaController {
	return &rdmaController{
		root: filepath.Join(root, string(Rdma)),
	}
}

func createCmdString(device string, limits *specs.LinuxRdma) string {
	var cmdString string

	cmdString = device
	if limits.HcaHandles != nil {
		cmdString = cmdString + " " + "hca_handle=" + strconv.FormatUint(uint64(*limits.HcaHandles), 10)
	}

	if limits.HcaObjects != nil {
		cmdString = cmdString + " " + "hca_object=" + strconv.FormatUint(uint64(*limits.HcaObjects), 10)
	}
	return cmdString
}

func (p *rdmaController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(p.Path(path), defaultDirPerm); err != nil {
		return err
	}

	for device, limit := range resources.Rdma {
		if device != "" && (limit.HcaHandles != nil || limit.HcaObjects != nil) {
			limit := limit
			return retryingWriteFile(
				filepath.Join(p.Path(path), "rdma.max"),
				[]byte(createCmdString(device, &limit)),
				defaultFilePerm,
			)
		}
	}
	return nil
}

func (p *rdmaController) Update(path string, resources *specs.LinuxResources) error {
	return p.Create(path, resources)
}

func parseRdmaKV(raw string, entry *v1.RdmaEntry) {
	var value uint64
	var err error

	parts := strings.Split(raw, "=")
	switch len(parts) {
	case 2:
		if parts[1] == "max" {
			value = math.MaxUint32
		} else {
			value, err = parseUint(parts[1], 10, 32)
			if err != nil {
				return
			}
		}
		if parts[0] == "hca_handle" {
			entry.HcaHandles = uint32(value)
		} else if parts[0] == "hca_object" {
			entry.HcaObjects = uint32(value)
		}
	}
}

func toRdmaEntry(strEntries []string) []*v1.RdmaEntry {
	var rdmaEntries []*v1.RdmaEntry
	for i := range strEntries {
		parts := strings.Fields(strEntries[i])
		switch len(parts) {
		case 3:
			entry := new(v1.RdmaEntry)
			entry.Device = parts[0]
			parseRdmaKV(parts[1], entry)
			parseRdmaKV(parts[2], entry)

			rdmaEntries = append(rdmaEntries, entry)
		default:
			continue
		}
	}
	return rdmaEntries
}

func (p *rdmaController) Stat(path string, stats *v1.Metrics) error {

	currentData, err := os.ReadFile(filepath.Join(p.Path(path), "rdma.current"))
	if err != nil {
		return err
	}
	currentPerDevices := strings.Split(string(currentData), "\n")

	maxData, err := os.ReadFile(filepath.Join(p.Path(path), "rdma.max"))
	if err != nil {
		return err
	}
	maxPerDevices := strings.Split(string(maxData), "\n")

	// If device got removed between reading two files, ignore returning
	// stats.
	if len(currentPerDevices) != len(maxPerDevices) {
		return nil
	}

	currentEntries := toRdmaEntry(currentPerDevices)
	maxEntries := toRdmaEntry(maxPerDevices)

	stats.Rdma = &v1.RdmaStat{
		Current: currentEntries,
		Limit:   maxEntries,
	}
	return nil
}
