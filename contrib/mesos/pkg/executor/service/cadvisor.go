/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"

	cadvisorapi "github.com/google/cadvisor/info/v1"
)

type MesosCadvisor struct {
	cadvisor.Interface
	cores int
	mem   uint64
}

func NewMesosCadvisor(cores int, mem uint64, port uint, runtime string) (*MesosCadvisor, error) {
	c, err := cadvisor.New(port, runtime)
	if err != nil {
		return nil, err
	}
	return &MesosCadvisor{c, cores, mem}, nil
}

func (mc *MesosCadvisor) MachineInfo() (*cadvisorapi.MachineInfo, error) {
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
