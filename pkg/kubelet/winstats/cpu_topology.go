//go:build windows
// +build windows

/*
Copyright 2024 The Kubernetes Authors.

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

package winstats

import (
	"fmt"
	"syscall"
	"unsafe"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/klog/v2"
)

var (
	procGetLogicalProcessorInformationEx = modkernel32.NewProc("GetLogicalProcessorInformationEx")
	getNumaAvailableMemoryNodeEx         = modkernel32.NewProc("GetNumaAvailableMemoryNodeEx")
	procGetNumaNodeProcessorMaskEx       = modkernel32.NewProc("GetNumaNodeProcessorMaskEx")
)

type relationType int

const (
	relationProcessorCore relationType = iota
	relationNumaNode
	relationCache
	relationProcessorPackage
	relationGroup
	relationProcessorDie
	relationNumaNodeEx
	relationProcessorModule
	relationAll = 0xffff
)

type systemLogicalProcessorInformationEx struct {
	Relationship uint32
	Size         uint32
	data         interface{}
}

type processorRelationship struct {
	Flags           byte
	EfficiencyClass byte
	Reserved        [20]byte
	GroupCount      uint16
	// groupMasks is an []GroupAffinity. In c++ this is a union of either one or many GroupAffinity based on GroupCount
	GroupMasks interface{}
}

// GroupAffinity represents the processor group affinity of cpus
// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-group_affinity
type GroupAffinity struct {
	Mask     uint64
	Group    uint16
	Reserved [3]uint16
}

// MaskString returns the affinity mask as a string of 0s and 1s
func (a GroupAffinity) MaskString() string {
	return fmt.Sprintf("%064b", a.Mask)
}

// Processors returns a list of processors ids that are part of the affinity mask
// Windows doesn't track processors by ID but kubelet converts them to a number
func (a GroupAffinity) Processors() []int {
	processors := []int{}
	for i := 0; i < 64; i++ {
		if a.Mask&(1<<i) != 0 {
			processors = append(processors, i+(int(a.Group)*64))
		}
	}
	return processors
}

// CpusToGroupAffinity converts a list of CPUs to a map of GroupAffinity split by windows CPU group.
// Windows doesn't track processors by ID but kubelet converts them to a number and this function goes in reverse.
func CpusToGroupAffinity(cpus []int) map[int]*GroupAffinity {
	groupAffinities := make(map[int]*GroupAffinity)
	for _, cpu := range cpus {
		group := uint16(cpu / 64)

		groupAffinity, ok := groupAffinities[int(group)]
		if !ok {
			groupAffinity = &GroupAffinity{
				Group: group,
			}
			groupAffinities[int(group)] = groupAffinity
		}
		mask := uint64(1 << (cpu % 64))
		groupAffinity.Mask |= mask
	}
	return groupAffinities
}

// GetCPUsForNUMANode queries the system for the CPUs that are part of the given NUMA node.
func GetCPUsforNUMANode(nodeNumber uint16) (*GroupAffinity, error) {
	var affinity GroupAffinity

	r1, _, err := procGetNumaNodeProcessorMaskEx.Call(
		uintptr(nodeNumber),
		uintptr(unsafe.Pointer(&affinity)),
	)
	if r1 == 0 {
		return nil, fmt.Errorf("Error getting CPU mask for NUMA node %d: %v", nodeNumber, err)
	}

	return &affinity, nil
}

type numaNodeRelationship struct {
	NodeNumber uint32
	Reserved   [18]byte
	GroupCount uint16
	GroupMasks interface{} //[]GroupAffinity in c++ this is a union of either one or many GroupAffinity based on GroupCount
}

type processor struct {
	CoreID   int
	SocketID int
	NodeID   int
}

func processorInfo(logger klog.Logger, relationShip relationType) (int, int, []cadvisorapi.Node, error) {
	// Call once to get the length of data to return
	var returnLength uint32 = 0
	r1, _, err := procGetLogicalProcessorInformationEx.Call(
		uintptr(relationShip),
		uintptr(0),
		uintptr(unsafe.Pointer(&returnLength)),
	)
	if r1 != 0 && err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
		return 0, 0, nil, fmt.Errorf("call to GetLogicalProcessorInformationEx failed: %v", err)
	}

	// Allocate the buffer with the length it should be
	buffer := make([]byte, returnLength)

	// Call GetLogicalProcessorInformationEx again to get the actual information
	r1, _, err = procGetLogicalProcessorInformationEx.Call(
		uintptr(relationShip),
		uintptr(unsafe.Pointer(&buffer[0])),
		uintptr(unsafe.Pointer(&returnLength)),
	)
	if r1 == 0 {
		return 0, 0, nil, fmt.Errorf("call to GetLogicalProcessorInformationEx failed: %v", err)
	}

	return convertWinApiToCadvisorApi(logger, buffer)
}

func convertWinApiToCadvisorApi(logger klog.Logger, buffer []byte) (int, int, []cadvisorapi.Node, error) {
	logicalProcessors := make(map[int]*processor)
	numofSockets := 0
	numOfcores := 0
	nodes := []cadvisorapi.Node{}
	for offset := 0; offset < len(buffer); {
		// check size in buffer to avoid out of bounds access, we don't know the type or size yet
		if offset+int(unsafe.Sizeof(systemLogicalProcessorInformationEx{})) > len(buffer) {
			return 0, 0, nil, fmt.Errorf("remaining buffer too small while reading windows processor relationship")
		}
		info := (*systemLogicalProcessorInformationEx)(unsafe.Pointer(&buffer[offset]))
		// check one more time now that we know the size of the struct
		if offset+int(info.Size) > len(buffer) {
			return 0, 0, nil, fmt.Errorf("remaining buffer too small while reading windows processor relationship")
		}
		switch (relationType)(info.Relationship) {
		case relationProcessorCore, relationProcessorPackage:
			relationship := (*processorRelationship)(unsafe.Pointer(&info.data))
			groupMasks := make([]GroupAffinity, relationship.GroupCount)
			for i := 0; i < int(relationship.GroupCount); i++ {
				groupMasks[i] = *(*GroupAffinity)(unsafe.Pointer(uintptr(unsafe.Pointer(&relationship.GroupMasks)) + uintptr(i)*unsafe.Sizeof(GroupAffinity{})))
			}

			if relationProcessorCore == (relationType)(info.Relationship) {
				numOfcores++
			}

			if relationProcessorPackage == (relationType)(info.Relationship) {
				numofSockets++
			}

			//iterate over group masks and add each processor to the map
			for _, groupMask := range groupMasks {
				for _, processorId := range groupMask.Processors() {
					p, ok := logicalProcessors[processorId]
					if !ok {
						p = &processor{}
						logicalProcessors[processorId] = p
					}
					if relationProcessorCore == (relationType)(info.Relationship) {
						p.CoreID = numOfcores
					}
					if relationProcessorPackage == (relationType)(info.Relationship) {
						p.SocketID = numofSockets
					}
				}
			}

		case relationNumaNode, relationNumaNodeEx:
			numaNodeRelationship := (*numaNodeRelationship)(unsafe.Pointer(&info.data))
			groupMasks := make([]GroupAffinity, numaNodeRelationship.GroupCount)
			for i := 0; i < int(numaNodeRelationship.GroupCount); i++ {
				groupMasks[i] = *(*GroupAffinity)(unsafe.Pointer(uintptr(unsafe.Pointer(&numaNodeRelationship.GroupMasks)) + uintptr(i)*unsafe.Sizeof(GroupAffinity{})))
			}

			nodes = append(nodes, cadvisorapi.Node{Id: int(numaNodeRelationship.NodeNumber)})

			for _, groupMask := range groupMasks {
				for processorId := range groupMask.Processors() {
					p, ok := logicalProcessors[processorId]
					if !ok {
						p = &processor{}
						logicalProcessors[processorId] = p
					}
					p.NodeID = int(numaNodeRelationship.NodeNumber)
				}
			}

		default:
			logger.V(4).Info("Not using Windows CPU relationship type", "relationship", info.Relationship)
		}

		// Move the offset to the next SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX struct
		offset += int(info.Size)
	}

	for processId, p := range logicalProcessors {
		node := nodes[p.NodeID]
		if node.Id != p.NodeID {
			return 0, 0, nil, fmt.Errorf("node ID mismatch: %d != %d", node.Id, p.NodeID)
		}
		availableBytes := uint64(0)
		r1, _, err := getNumaAvailableMemoryNodeEx.Call(uintptr(p.NodeID), uintptr(unsafe.Pointer(&availableBytes)))
		if r1 == 0 {
			return 0, 0, nil, fmt.Errorf("call to GetNumaAvailableMemoryNodeEx failed: %v", err)
		}
		node.Memory = availableBytes
		node.AddThread(processId, p.CoreID)
		ok, coreIdx := node.FindCore(p.CoreID)
		if !ok {
			return 0, 0, nil, fmt.Errorf("core not found: %d", p.CoreID)
		}
		node.Cores[coreIdx].SocketID = p.SocketID
		nodes[p.NodeID] = node
	}

	return numOfcores, numofSockets, nodes, nil
}
