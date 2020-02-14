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
	"io/ioutil"
	"os"
	"sync"

	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

type stateFileDataV1 struct {
	PolicyName    string            `json:"policyName"`
	DefaultCPUSet string            `json:"defaultCpuSet"`
	Entries       map[string]string `json:"entries,omitempty"`
}

type stateFileDataV2 struct {
	PolicyName    string                       `json:"policyName"`
	DefaultCPUSet string                       `json:"defaultCpuSet"`
	Entries       map[string]map[string]string `json:"entries,omitempty"`
}

var _ State = &stateFile{}

type stateFile struct {
	sync.RWMutex
	stateFilePath     string
	policyName        string
	cache             State
	initialContainers containermap.ContainerMap
}

// NewFileState creates new State for keeping track of cpu/pod assignment with file backend
func NewFileState(filePath string, policyName string, initialContainers containermap.ContainerMap) (State, error) {
	stateFile := &stateFile{
		stateFilePath:     filePath,
		cache:             NewMemoryState(),
		policyName:        policyName,
		initialContainers: initialContainers,
	}

	if err := stateFile.tryRestoreState(); err != nil {
		// could not restore state, init new state file
		klog.Errorf("[cpumanager] state file: unable to restore state from disk (%v)"+
			" We cannot guarantee sane CPU affinity for existing containers."+
			" Please drain this node and delete the CPU manager state file \"%s\" before restarting Kubelet.",
			err, stateFile.stateFilePath)
		return nil, err
	}

	return stateFile, nil
}

// migrateV1StateToV2State() converts state from the v1 format to the v2 format
func (sf *stateFile) migrateV1StateToV2State(src *stateFileDataV1, dst *stateFileDataV2) error {
	if src.PolicyName != "" {
		dst.PolicyName = src.PolicyName
	}
	if src.DefaultCPUSet != "" {
		dst.DefaultCPUSet = src.DefaultCPUSet
	}
	for containerID, cset := range src.Entries {
		podUID, containerName, err := sf.initialContainers.GetContainerRef(containerID)
		if err != nil {
			return fmt.Errorf("containerID '%v' not found in initial containers list", containerID)
		}
		if dst.Entries == nil {
			dst.Entries = make(map[string]map[string]string)
		}
		if _, exists := dst.Entries[podUID]; !exists {
			dst.Entries[podUID] = make(map[string]string)
		}
		dst.Entries[podUID][containerName] = cset
	}
	return nil
}

// tryRestoreState tries to read state file, upon any error,
// err message is logged and state is left clean. un-initialized
func (sf *stateFile) tryRestoreState() error {
	sf.Lock()
	defer sf.Unlock()
	var err error
	var content []byte

	content, err = ioutil.ReadFile(sf.stateFilePath)

	// If the state file does not exist or has zero length, write a new file.
	if os.IsNotExist(err) || len(content) == 0 {
		err := sf.storeState()
		if err != nil {
			return err
		}
		klog.Infof("[cpumanager] state file: created new state file \"%s\"", sf.stateFilePath)
		return nil
	}

	// Fail on any other file read error.
	if err != nil {
		return err
	}

	// File exists; try to read it.
	var readStateV1 stateFileDataV1
	var readStateV2 stateFileDataV2

	if err = json.Unmarshal(content, &readStateV1); err != nil {
		readStateV1 = stateFileDataV1{} // reset it back to 0
		if err = json.Unmarshal(content, &readStateV2); err != nil {
			klog.Errorf("[cpumanager] state file: could not unmarshal, corrupted state file - \"%s\"", sf.stateFilePath)
			return err
		}
	}

	if err = sf.migrateV1StateToV2State(&readStateV1, &readStateV2); err != nil {
		klog.Errorf("[cpumanager] state file: could not migrate v1 state to v2 state  - \"%s\"", sf.stateFilePath)
		return err
	}

	if sf.policyName != readStateV2.PolicyName {
		return fmt.Errorf("policy configured \"%s\" != policy from state file \"%s\"", sf.policyName, readStateV2.PolicyName)
	}

	var tmpDefaultCPUSet cpuset.CPUSet
	if tmpDefaultCPUSet, err = cpuset.Parse(readStateV2.DefaultCPUSet); err != nil {
		klog.Errorf("[cpumanager] state file: could not parse state file - [defaultCpuSet:\"%s\"]", readStateV2.DefaultCPUSet)
		return err
	}

	var tmpContainerCPUSet cpuset.CPUSet
	tmpAssignments := ContainerCPUAssignments{}
	for pod := range readStateV2.Entries {
		tmpAssignments[pod] = make(map[string]cpuset.CPUSet)
		for container, cpuString := range readStateV2.Entries[pod] {
			if tmpContainerCPUSet, err = cpuset.Parse(cpuString); err != nil {
				klog.Errorf("[cpumanager] state file: could not parse state file - pod: %s, container: %s, cpuset: \"%s\"", pod, container, cpuString)
				return err
			}
			tmpAssignments[pod][container] = tmpContainerCPUSet
		}
	}

	sf.cache.SetDefaultCPUSet(tmpDefaultCPUSet)
	sf.cache.SetCPUAssignments(tmpAssignments)

	klog.V(2).Infof("[cpumanager] state file: restored state from state file \"%s\"", sf.stateFilePath)
	klog.V(2).Infof("[cpumanager] state file: defaultCPUSet: %s", tmpDefaultCPUSet.String())

	return nil
}

// saves state to a file, caller is responsible for locking
func (sf *stateFile) storeState() error {
	var content []byte
	var err error

	data := stateFileDataV2{
		PolicyName:    sf.policyName,
		DefaultCPUSet: sf.cache.GetDefaultCPUSet().String(),
		Entries:       map[string]map[string]string{},
	}

	assignments := sf.cache.GetCPUAssignments()
	for pod := range assignments {
		data.Entries[pod] = map[string]string{}
		for container, cset := range assignments[pod] {
			data.Entries[pod][container] = cset.String()
		}
	}

	if content, err = json.Marshal(data); err != nil {
		return fmt.Errorf("[cpumanager] state file: could not serialize state to json")
	}

	if err = ioutil.WriteFile(sf.stateFilePath, content, 0644); err != nil {
		return fmt.Errorf("[cpumanager] state file not written")
	}

	return nil
}

func (sf *stateFile) GetCPUSet(podUID string, containerName string) (cpuset.CPUSet, bool) {
	sf.RLock()
	defer sf.RUnlock()

	res, ok := sf.cache.GetCPUSet(podUID, containerName)
	return res, ok
}

func (sf *stateFile) GetDefaultCPUSet() cpuset.CPUSet {
	sf.RLock()
	defer sf.RUnlock()

	return sf.cache.GetDefaultCPUSet()
}

func (sf *stateFile) GetCPUSetOrDefault(podUID string, containerName string) cpuset.CPUSet {
	sf.RLock()
	defer sf.RUnlock()

	return sf.cache.GetCPUSetOrDefault(podUID, containerName)
}

func (sf *stateFile) GetCPUAssignments() ContainerCPUAssignments {
	sf.RLock()
	defer sf.RUnlock()
	return sf.cache.GetCPUAssignments()
}

func (sf *stateFile) SetCPUSet(podUID string, containerName string, cset cpuset.CPUSet) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.SetCPUSet(podUID, containerName, cset)
	err := sf.storeState()
	if err != nil {
		klog.Warningf("store state to checkpoint error: %v", err)
	}
}

func (sf *stateFile) SetDefaultCPUSet(cset cpuset.CPUSet) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.SetDefaultCPUSet(cset)
	err := sf.storeState()
	if err != nil {
		klog.Warningf("store state to checkpoint error: %v", err)
	}
}

func (sf *stateFile) SetCPUAssignments(a ContainerCPUAssignments) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.SetCPUAssignments(a)
	err := sf.storeState()
	if err != nil {
		klog.Warningf("store state to checkpoint error: %v", err)
	}
}

func (sf *stateFile) Delete(podUID string, containerName string) {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.Delete(podUID, containerName)
	err := sf.storeState()
	if err != nil {
		klog.Warningf("store state to checkpoint error: %v", err)
	}
}

func (sf *stateFile) ClearState() {
	sf.Lock()
	defer sf.Unlock()
	sf.cache.ClearState()
	err := sf.storeState()
	if err != nil {
		klog.Warningf("store state to checkpoint error: %v", err)
	}
}
