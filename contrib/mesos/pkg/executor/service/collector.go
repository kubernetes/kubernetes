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

package service

import (
	"fmt"

	"k8s.io/kubernetes/pkg/kubelet/collector"
	"k8s.io/kubernetes/pkg/kubelet/collector/cadvisor"
)

type MesosCollector struct {
	collector.Interface
	cores int
	mem   int64
}

func NewMesosCollector(cores int, mem int64, collector string, collectorURL string) (*MesosCollector, error) {
	switch collector {
	case "cadvisor":
		c, err := cadvisor.NewCadvisorCollector(collectorURL)
		if err != nil {
			return nil, err
		}
		return &MesosCollector{c, cores, mem}, nil
	default:
		return nil, fmt.Errorf("Unrecognized collector type: %s", collector)
	}
}

func (mc *MesosCollector) MachineInfo() (*collector.MachineInfo, error) {
	mi, err := mc.Interface.MachineInfo()
	if err != nil {
		return nil, err
	}

	// set Mesos provided values
	mesosMi := *mi
	mesosMi.NumCores = mc.cores
	mesosMi.MemoryCapacity = mc.mem

	return &mesosMi, nil
}
