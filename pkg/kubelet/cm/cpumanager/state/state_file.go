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
	"fmt"
	"github.com/golang/glog"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"os"
	"sync"
)

type stateFileData struct {
	PolicyName    string            `json:"policyName"`
	DefaultCPUSet string            `json:"defaultCpuSet"`
	Entries       map[string]string `json:"entries,omitempty"`
}

var _ State = &stateFile{}

type stateFile struct {
	sync.RWMutex
	stateFilePath string
	policyName    string
	cache         State
}

// NewFileState creates new State for keeping track of cpu/pod assignment with file backend
func NewFileState(filePath string, policyName string) State {
	stateFile := &stateFile{
		stateFilePath: filePath,
		cache:         NewMemoryState(),
		policyName:    policyName,
	}

	if err := stateFile.tryRestoreState(); err != nil {
		// could not restore state, init new state file
		msg := fmt.Sprintf("[cpumanager] state file: unable to restore state from disk (%s)\n", err.Error()) +
			"Panicking because we cannot guarantee sane CPU affinity for existing containers.\n" +
			fmt.Sprintf("Please drain this node and delete the CPU manager state file \"%s\" before restarting Kubelet.", stateFile.stateFilePath)
		panic(msg)
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

	content, err = ioutil.ReadFile(sf.stateFilePath)

	// If the state file does not exist or has zero length, write a new file.
	if os.IsNotExist(err) || len(content) == 0 {
		sf.storeState()
		glog.Infof("[cpumanager] state file: created new state file \"%s\"", sf.stateFilePath)
		return nil
	}

	// Fail on any other file read error.
	if err != nil {
		return err
	}

	// File exists; try to read it.
	var readState stateFileData

	if err = json.Unmarshal(content, &readState); err != nil {
		glog.Errorf("[cpumanager] state file: could not unmarshal, corrupted state file - \"%s\"", sf.stateFilePath)
		return err
	}

	if sf.policyName != readState.PolicyName {
		return fmt.Errorf("policy configured \"%s\" != policy from state file \"%s\"", sf.policyName, readState.PolicyName)
	}

	if tmpDefaultCPUSet, err = cpuset.Parse(readState.DefaultCPUSet); err != nil {
		glog.Errorf("[cpumanager] state file: could not parse state file - [defaultCpuSet:\"%s\"]", readState.DefaultCPUSet)
		return err
	}

	for containerID, cpuString := range readState.Entries {
		if tmpContainerCPUSet, err = cpuset.Parse(cpuString); err != nil {
			glog.Errorf("[cpumanager] state file: could not parse state file - container id: %s, cpuset: \"%s\"", containerID, cpuString)
			return err
		}
		tmpAssignments[containerID] = tmpContainerCPUSet
	}

	sf.cache.SetDefaultCPUSet(tmpDefaultCPUSet)
	sf.cache.SetCPUAssignments(tmpAssignments)

	glog.V(2).Infof("[cpumanager] state file: restored state from state file \"%s\"", sf.stateFilePath)
	glog.V(2).Infof("[cpumanager] state file: defaultCPUSet: %s", tmpDefaultCPUSet.String())

	return nil
}

// saves state to a file, caller is responsible for locking
func (sf *stateFile) storeState() {
	var content []byte
	var err error

	data := stateFileData{
		PolicyName:    sf.policyName,
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
