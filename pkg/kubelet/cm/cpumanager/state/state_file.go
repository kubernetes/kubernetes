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

type stateFileData struct {
	DefaultCPUSet string            `json:"defaultCpuSet"`
	Entries       map[string]string `json:"entries,omitempty"`
}

var _ State = &stateFile{}

type stateFile struct {
	sync.RWMutex
	stateFilePath string
	cache         State
}

// NewFileState creates new State for keeping track of cpu/pod assignment with file backend
func NewFileState(filePath string) State {
	stateFile := &stateFile{
		stateFilePath: filePath,
		cache:         NewMemoryState(),
	}

	if err := stateFile.tryRestoreState(); err != nil {
		// could not restore state, init new state file
		glog.Infof("[cpumanager] state file: initializing empty state file")
		stateFile.cache.ClearState()
		stateFile.storeState()
	}

	return stateFile
}

// tryRestoreState tries to read state file, upon any error,
// err message is logged and state is left clean. un-initialized
func (sf *stateFile) tryRestoreState() error {
	sf.Lock()
	defer sf.Unlock()
	var err error

	// used when all parsing is ok
	tmpAssignments := make(ContainerCPUAssignments)
	tmpDefaultCPUSet := cpuset.NewCPUSet()
	tmpContainerCPUSet := cpuset.NewCPUSet()

	var content []byte

	if content, err = ioutil.ReadFile(sf.stateFilePath); os.IsNotExist(err) {
		// Create file
		if _, err = os.Create(sf.stateFilePath); err != nil {
			glog.Errorf("[cpumanager] state file: unable to create state file \"%s\":%s", sf.stateFilePath, err.Error())
			panic("[cpumanager] state file not created")
		}
		glog.Infof("[cpumanager] state file: created empty state file \"%s\"", sf.stateFilePath)
	} else {
		// File exists - try to read
		var readState stateFileData

		if err = json.Unmarshal(content, &readState); err != nil {
			glog.Warningf("[cpumanager] state file: could not unmarshal, corrupted state file - \"%s\"", sf.stateFilePath)
			return err
		}

		if tmpDefaultCPUSet, err = cpuset.Parse(readState.DefaultCPUSet); err != nil {
			glog.Warningf("[cpumanager] state file: could not parse state file - [defaultCpuSet:\"%s\"]", readState.DefaultCPUSet)
			return err
		}

		for containerID, cpuString := range readState.Entries {
			if tmpContainerCPUSet, err = cpuset.Parse(cpuString); err != nil {
				glog.Warningf("[cpumanager] state file: could not parse state file - container id: %s, cpuset: \"%s\"", containerID, cpuString)
				return err
			}
			tmpAssignments[containerID] = tmpContainerCPUSet
		}

		sf.cache.SetDefaultCPUSet(tmpDefaultCPUSet)
		sf.cache.SetCPUAssignments(tmpAssignments)

		glog.V(2).Infof("[cpumanager] state file: restored state from state file \"%s\"", sf.stateFilePath)
		glog.V(2).Infof("[cpumanager] state file: defaultCPUSet: %s", tmpDefaultCPUSet.String())
	}
	return nil
}

// saves state to a file, caller is responsible for locking
func (sf *stateFile) storeState() {
	var content []byte
	var err error

	data := stateFileData{
		DefaultCPUSet: sf.cache.GetDefaultCPUSet().String(),
		Entries:       map[string]string{},
	}

	for containerID, cset := range sf.cache.GetCPUAssignments() {
		data.Entries[containerID] = cset.String()
	}

	if content, err = json.Marshal(data); err != nil {
		panic("[cpumanager] state file: could not serialize state to json")
	}

	if err = ioutil.WriteFile(sf.stateFilePath, content, 0644); err != nil {
		panic("[cpumanager] state file not written")
	}
	return
}

func (sf *stateFile) GetCPUSet(containerID string) (cpuset.CPUSet, bool) {
	sf.RLock()
	defer sf.RUnlock()

	res, ok := sf.cache.GetCPUSet(containerID)
	return res, ok
}

func (sf *stateFile) GetDefaultCPUSet() cpuset.CPUSet {
	sf.RLock()
	defer sf.RUnlock()

	return sf.cache.GetDefaultCPUSet()
}

func (sf *stateFile) GetCPUSetOrDefault(containerID string) cpuset.CPUSet {
	sf.RLock()
	defer sf.RUnlock()

	return sf.cache.GetCPUSetOrDefault(containerID)
}

func (sf *stateFile) GetCPUAssignments() ContainerCPUAssignments {
	sf.RLock()
	defer sf.RUnlock()
	return sf.cache.GetCPUAssignments()
}

func (sf *stateFile) SetCPUSet(containerID string, cset cpuset.CPUSet) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.SetCPUSet(containerID, cset)
	sf.storeState()
}

func (sf *stateFile) SetDefaultCPUSet(cset cpuset.CPUSet) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.SetDefaultCPUSet(cset)
	sf.storeState()
}

func (sf *stateFile) SetCPUAssignments(a ContainerCPUAssignments) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.SetCPUAssignments(a)
	sf.storeState()
}

func (sf *stateFile) Delete(containerID string) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.Delete(containerID)
	sf.storeState()
}

func (sf *stateFile) ClearState() {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.ClearState()
	sf.storeState()
}
