/*
Copyright 2017 The Kubernetes Authors.

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

package state

import (
	"encoding/json"
	"github.com/golang/glog"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"os"
	"sync"
)

type stateData struct {
	DefaultCpuSet string            `json:"defaultCpuSet"`
	Entries       map[string]string `json:"reservedList,omitempty"`
}

type stateFile struct {
	sync.RWMutex
	stateFilePath string
}

func NewStateFileBackend() *stateFile {
	return &stateFile{
		stateFilePath: "/var/lib/kubelet/cpu_manager_state",
	}
}

func (sf *stateFile) tryRestoreState() (defaultCPUSet cpuset.CPUSet, assignments ContainerCpuAssignment, err error) {
	sf.RLock()
	defer sf.RUnlock()

	// init return values
	defaultCPUSet = cpuset.NewCPUSet()
	assignments = make(ContainerCpuAssignment)
	err = nil

	// used when all parsing is ok
	tmpAssignments := make(ContainerCpuAssignment)
	tmpDefaultCPUSet := cpuset.NewCPUSet()
	tmpContainerCpuSet := cpuset.NewCPUSet()

	var content []byte

	if content, err = ioutil.ReadFile(sf.stateFilePath); os.IsNotExist(err) {
		// Create file
		if _, err = os.Create(sf.stateFilePath); err != nil {
			glog.Errorf("[cpumanager] unable to create state file \"%s\"", sf.stateFilePath)
			return
		}
		glog.Infof("[cpumanager] created empty state file \"%s\"", sf.stateFilePath)
	} else {
		// File exists - try to read
		var readState stateData
		if err = json.Unmarshal(content, &readState); err != nil {
			glog.Errorf("[cpumanager] could not unmarshal, corrupted state file - \"%s\"", sf.stateFilePath)
			return
		}

		if tmpDefaultCPUSet, err = cpuset.Parse(readState.DefaultCpuSet); err != nil {
			glog.Errorf("[cpumanager] could not parse state file - [defaultCpuSet:\"%s\"]", readState.DefaultCpuSet)
			return
		}

		for containerID, cpuString := range readState.Entries {
			if tmpContainerCpuSet, err = cpuset.Parse(cpuString); err != nil {
				glog.Errorf("[cpumanager] could not parse state file - container id: %s, cpuset: \"%s\"", containerID, cpuString)
				return
			}
			tmpAssignments[containerID] = tmpContainerCpuSet
		}

		assignments = tmpAssignments
		defaultCPUSet = tmpDefaultCPUSet
		glog.V(2).Infof("[cpumanager] restored state from state file \"%s\"", sf.stateFilePath)
		glog.V(4).Infof("[cpumanager] defaultCPUSet: %s", defaultCPUSet.String())
	}
	return
}

func (sf *stateFile) updateStateFile(defaultCPUSet cpuset.CPUSet, assignments ContainerCpuAssignment) error {
	sf.RLock()
	defer sf.RUnlock()
	var content []byte
	var err error

	writeState := stateData{
		DefaultCpuSet: defaultCPUSet.String(),
		Entries:       map[string]string{},
	}

	for containerID, cset := range assignments {
		writeState.Entries[containerID] = cset.String()
	}

	if content, err = json.Marshal(writeState); err != nil {
		glog.Errorf("[cpumanager] could not parse state to json")
		return err
	}

	if err = ioutil.WriteFile(sf.stateFilePath, content, 0644); err != nil {
		glog.Errorf("[cpumanager] could not write state to file")
		return err
	}

	return nil

}
