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

func (a GROUP_AFFINITY) Processors() []int {
	processors := []int{}
	for i := 0; i < 64; i++ {
		if a.Mask&(1<<i) != 0 {
			processors = append(processors, i+(int(a.Group)*64))
		}
	}
	return processors
}

func CpusToGroupAffinity(cpus []int) map[int]*GROUP_AFFINITY {
	groupAffinities := make(map[int]*GROUP_AFFINITY)
	for _, cpu := range cpus {
		group := uint16(cpu / 64)

		groupaffinity, ok := groupAffinities[int(group)]
		if !ok {
			groupaffinity = &GROUP_AFFINITY{
				Group: group,
			}
			groupAffinities[int(group)] = groupaffinity
		}
		mask := uintptr(1 << (cpu % 64))
		groupaffinity.Mask |= mask
	}
	return groupAffinities
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
	SocketID int
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

	return convertWinApiToCadvisorApi(buffer)
}

func convertWinApiToCadvisorApi(buffer []byte) (int, int, []cadvisorapi.Node, error) {
	logicalProcessors := make(map[int]*processor)
	numofSockets := 0
	numOfcores := 0
	nodes := []cadvisorapi.Node{}
	//iterate over the buffer casting it to the correct type
	for offset := 0; offset < len(buffer); {
		//todo check if there is enough left in buffer to read system_logical_processor_information_ex?
		info := (*SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(unsafe.Pointer(&buffer[offset]))
		switch (RelationType)(info.Relationship) {
		case RelationProcessorCore, RelationProcessorPackage:
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
			for _, groupMask := range groupMasks {
				for _, processorId := range groupMask.Processors() {
					p, ok := logicalProcessors[processorId]
					if !ok {
						p = &processor{}
						logicalProcessors[processorId] = p
					}
					if RelationProcessorCore == (RelationType)(info.Relationship) {
						p.CoreID = numOfcores
					}
					if RelationProcessorPackage == (RelationType)(info.Relationship) {
						p.SocketID = numofSockets
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

		case RelationCache:
			//cacheRelationship := (*CACHE_RELATIONSHIP)(unsafe.Pointer(&info.data))
			// TODO Process cache relationship data

		default:
			klog.V(4).Infof("Not using relationship type: %d", info.Relationship)
		}

		// Move the offset to the next SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX struct
		offset += int(info.Size)
	}

	for processId, p := range logicalProcessors {
		klog.V(4).Infof("Processor (%d): %v", processId, p)
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
		node.Cores[coreIdx].SocketID = p.SocketID
		nodes[p.NodeID] = node
	}

	return numOfcores, numofSockets, nodes, nil
}
