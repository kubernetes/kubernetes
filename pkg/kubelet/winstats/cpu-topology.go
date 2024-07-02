package winstats

import (
	"fmt"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/klog/v2"
	"syscall"
	"unsafe"
)

var (
	procGetLogicalProcessorInformationEx = modkernel32.NewProc("GetLogicalProcessorInformationEx")
	getNumaAvailableMemoryNodeEx         = modkernel32.NewProc("GetNumaAvailableMemoryNodeEx")
)

type RelationType int

const (
	RelationProcessorCore RelationType = iota
	RelationNumaNode
	RelationCache
	RelationProcessorPackage
	RelationGroup
	RelationProcessorDie
	RelationNumaNodeEx
	RelationProcessorModule
	RelationAll = 0xffff
)

type SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX struct {
	Relationship uint32
	Size         uint32
	data         interface{}
}

type PROCESSOR_RELATIONSHIP struct {
	Flags           byte
	EfficiencyClass byte
	Reserved        [20]byte
	GroupCount      uint16
	GroupMasks      interface{} //[]GROUP_AFFINITY // in c++ this is a union of either one or many GROUP_AFFINITY based on GroupCount
}

type GROUP_AFFINITY struct {
	Mask     uintptr
	Group    uint16
	Reserved [3]uint16
}

type NUMA_NODE_RELATIONSHIP struct {
	NodeNumber uint32
	Reserved   [18]byte
	GroupCount uint16
	GroupMasks interface{} //[]GROUP_AFFINITY // in c++ this is a union of either one or many GROUP_AFFINITY based on GroupCount
}

type CACHE_RELATIONSHIP struct {
	Level         byte
	Associativity byte
	LineSize      uint16
	CacheSize     uint32
	Type          PROCESSOR_CACHE_TYPE
	Reserved      [18]byte
	GroupCount    uint16
	GroupMasks    interface{} //interface{}[]GROUP_AFFINITY // in c++ this is a union of either one or many GROUP_AFFINITY based on GroupCount
}

type PROCESSOR_CACHE_TYPE int

const (
	CacheUnified PROCESSOR_CACHE_TYPE = iota
	CacheInstruction
	CacheData
	CacheTrace
	CacheUnknown
)

type GROUP_RELATIONSHIP struct {
	MaximumGroupCount uint16
	ActiveGroupCount  uint16
	Reserved          [20]byte
	GroupInfo         interface{} //[]PROCESSOR_GROUP_INFO
}

type PROCESSOR_GROUP_INFO struct {
	MaximumProcessorCount byte
	ActiveProcessorCount  byte
	Reserved              [38]byte
	ActiveProcessorMask   uintptr
}

type processor struct {
	CoreID   int
	SocektID int
	NodeID   int
}

func processorInfo(relationShip RelationType) (int, int, []cadvisorapi.Node, error) {
	// Call once to get the length of data to return
	var returnLength uint32 = 0
	r1, _, err := procGetLogicalProcessorInformationEx.Call(
		uintptr(relationShip),
		uintptr(0),
		uintptr(unsafe.Pointer(&returnLength)),
	)
	if r1 != 0 && err.(syscall.Errno) != syscall.ERROR_INSUFFICIENT_BUFFER {
		return 0, 0, nil, fmt.Errorf("Call to GetLogicalProcessorInformationEx failed: %v", err)
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
		return 0, 0, nil, fmt.Errorf("Call to GetLogicalProcessorInformationEx failed: %v", err)
	}

	processors := make(map[int]*processor)
	numofSockets := 0
	numOfcores := 0
	var nodes []cadvisorapi.Node
	//iterate over the buffer casting it to the correct type
	for offset := 0; offset < len(buffer); {
		info := (*SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(unsafe.Pointer(&buffer[offset]))
		switch (RelationType)(info.Relationship) {
		case RelationProcessorCore, RelationProcessorPackage, RelationProcessorModule:
			processorRelationship := (*PROCESSOR_RELATIONSHIP)(unsafe.Pointer(&info.data))
			groupMasks := make([]GROUP_AFFINITY, processorRelationship.GroupCount)
			for i := 0; i < int(processorRelationship.GroupCount); i++ {
				groupMasks[i] = *(*GROUP_AFFINITY)(unsafe.Pointer(uintptr(unsafe.Pointer(&processorRelationship.GroupMasks)) + uintptr(i)*unsafe.Sizeof(GROUP_AFFINITY{})))
			}

			if RelationProcessorCore == (RelationType)(info.Relationship) {
				numOfcores++
			}

			if RelationProcessorPackage == (RelationType)(info.Relationship) {
				numofSockets++
			}

			//iterate over group masks and add each processor to the map
			for groupNum, groupMask := range groupMasks {
				for i := 0; i < 64; i++ {
					if groupMask.Mask&(1<<i) != 0 {
						p, ok := processors[i*(groupNum+1)]
						if !ok {
							p = &processor{}
							processors[i*(groupNum+1)] = p
						}
						if RelationProcessorCore == (RelationType)(info.Relationship) {
							p.CoreID = numOfcores
						}
						if RelationProcessorPackage == (RelationType)(info.Relationship) {
							p.SocektID = numofSockets
						}
					}
				}
			}

		case RelationNumaNode, RelationNumaNodeEx:
			numaNodeRelationship := (*NUMA_NODE_RELATIONSHIP)(unsafe.Pointer(&info.data))
			groupMasks := make([]GROUP_AFFINITY, numaNodeRelationship.GroupCount)
			for i := 0; i < int(numaNodeRelationship.GroupCount); i++ {
				groupMasks[i] = *(*GROUP_AFFINITY)(unsafe.Pointer(uintptr(unsafe.Pointer(&numaNodeRelationship.GroupMasks)) + uintptr(i)*unsafe.Sizeof(GROUP_AFFINITY{})))
			}

			nodes = append(nodes, cadvisorapi.Node{Id: int(numaNodeRelationship.NodeNumber)})

			for groupNum, groupMask := range groupMasks {
				for i := 0; i < 64; i++ {
					if groupMask.Mask&(1<<i) != 0 {
						p, ok := processors[i*(groupNum+1)]
						if !ok {
							p = &processor{}
							processors[i*(groupNum+1)] = p
						}
						p.NodeID = int(numaNodeRelationship.NodeNumber)
					}
				}
			}

		case RelationCache:
			//cacheRelationship := (*CACHE_RELATIONSHIP)(unsafe.Pointer(&info.data))
			// TODO Process cache relationship data

		default:
			klog.V(4).Infof("Not using relationship type: %d", info.Relationship)
		}

		// Move the offset to the next SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX struct
		offset += int(info.Size)
	}

	for processId, p := range processors {
		klog.V(4).Infof("Processor (%s): %v", processId, p)
		node := nodes[p.NodeID]
		if node.Id != p.NodeID {
			return 0, 0, nil, fmt.Errorf("Node ID mismatch: %d != %d", node.Id, p.NodeID)
		}
		availableBytes := uint64(0)
		r1, _, err := getNumaAvailableMemoryNodeEx.Call(uintptr(p.NodeID), uintptr(unsafe.Pointer(&availableBytes)))
		if r1 == 0 {
			return 0, 0, nil, fmt.Errorf("Call to GetNumaAvailableMemoryNodeEx failed: %v", err)
		}
		node.Memory = availableBytes
		node.AddThread(processId, p.CoreID)
		ok, coreIdx := node.FindCore(p.CoreID)
		if !ok {
			return 0, 0, nil, fmt.Errorf("Core not found: %d", p.CoreID)
		}
		node.Cores[coreIdx].SocketID = p.SocektID
		nodes[p.NodeID] = node
	}

	return numOfcores, numofSockets, nodes, nil
}
