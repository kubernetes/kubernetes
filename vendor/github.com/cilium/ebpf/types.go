package ebpf

import (
	"github.com/cilium/ebpf/internal/unix"
)

//go:generate stringer -output types_string.go -type=MapType,ProgramType,AttachType,PinType

// MapType indicates the type map structure
// that will be initialized in the kernel.
type MapType uint32

// All the various map types that can be created
const (
	UnspecifiedMap MapType = iota
	// Hash is a hash map
	Hash
	// Array is an array map
	Array
	// ProgramArray - A program array map is a special kind of array map whose map
	// values contain only file descriptors referring to other eBPF
	// programs.  Thus, both the key_size and value_size must be
	// exactly four bytes.  This map is used in conjunction with the
	// TailCall helper.
	ProgramArray
	// PerfEventArray - A perf event array is used in conjunction with PerfEventRead
	// and PerfEventOutput calls, to read the raw bpf_perf_data from the registers.
	PerfEventArray
	// PerCPUHash - This data structure is useful for people who have high performance
	// network needs and can reconcile adds at the end of some cycle, so that
	// hashes can be lock free without the use of XAdd, which can be costly.
	PerCPUHash
	// PerCPUArray - This data structure is useful for people who have high performance
	// network needs and can reconcile adds at the end of some cycle, so that
	// hashes can be lock free without the use of XAdd, which can be costly.
	// Each CPU gets a copy of this hash, the contents of all of which can be reconciled
	// later.
	PerCPUArray
	// StackTrace - This holds whole user and kernel stack traces, it can be retrieved with
	// GetStackID
	StackTrace
	// CGroupArray - This is a very niche structure used to help SKBInCGroup determine
	// if an skb is from a socket belonging to a specific cgroup
	CGroupArray
	// LRUHash - This allows you to create a small hash structure that will purge the
	// least recently used items rather than thow an error when you run out of memory
	LRUHash
	// LRUCPUHash - This is NOT like PerCPUHash, this structure is shared among the CPUs,
	// it has more to do with including the CPU id with the LRU calculation so that if a
	// particular CPU is using a value over-and-over again, then it will be saved, but if
	// a value is being retrieved a lot but sparsely across CPUs it is not as important, basically
	// giving weight to CPU locality over overall usage.
	LRUCPUHash
	// LPMTrie - This is an implementation of Longest-Prefix-Match Trie structure. It is useful,
	// for storing things like IP addresses which can be bit masked allowing for keys of differing
	// values to refer to the same reference based on their masks. See wikipedia for more details.
	LPMTrie
	// ArrayOfMaps - Each item in the array is another map. The inner map mustn't be a map of maps
	// itself.
	ArrayOfMaps
	// HashOfMaps - Each item in the hash map is another map. The inner map mustn't be a map of maps
	// itself.
	HashOfMaps
	// DevMap - Specialized map to store references to network devices.
	DevMap
	// SockMap - Specialized map to store references to sockets.
	SockMap
	// CPUMap - Specialized map to store references to CPUs.
	CPUMap
	// XSKMap - Specialized map for XDP programs to store references to open sockets.
	XSKMap
	// SockHash - Specialized hash to store references to sockets.
	SockHash
	// CGroupStorage - Special map for CGroups.
	CGroupStorage
	// ReusePortSockArray - Specialized map to store references to sockets that can be reused.
	ReusePortSockArray
	// PerCPUCGroupStorage - Special per CPU map for CGroups.
	PerCPUCGroupStorage
	// Queue - FIFO storage for BPF programs.
	Queue
	// Stack - LIFO storage for BPF programs.
	Stack
	// SkStorage - Specialized map for local storage at SK for BPF programs.
	SkStorage
	// DevMapHash - Hash-based indexing scheme for references to network devices.
	DevMapHash
	StructOpts
	RingBuf
	InodeStorage
	TaskStorage
)

// hasPerCPUValue returns true if the Map stores a value per CPU.
func (mt MapType) hasPerCPUValue() bool {
	return mt == PerCPUHash || mt == PerCPUArray || mt == LRUCPUHash
}

// canStoreMap returns true if the map type accepts a map fd
// for update and returns a map id for lookup.
func (mt MapType) canStoreMap() bool {
	return mt == ArrayOfMaps || mt == HashOfMaps
}

// canStoreProgram returns true if the map type accepts a program fd
// for update and returns a program id for lookup.
func (mt MapType) canStoreProgram() bool {
	return mt == ProgramArray
}

// ProgramType of the eBPF program
type ProgramType uint32

// eBPF program types
const (
	UnspecifiedProgram ProgramType = iota
	SocketFilter
	Kprobe
	SchedCLS
	SchedACT
	TracePoint
	XDP
	PerfEvent
	CGroupSKB
	CGroupSock
	LWTIn
	LWTOut
	LWTXmit
	SockOps
	SkSKB
	CGroupDevice
	SkMsg
	RawTracepoint
	CGroupSockAddr
	LWTSeg6Local
	LircMode2
	SkReuseport
	FlowDissector
	CGroupSysctl
	RawTracepointWritable
	CGroupSockopt
	Tracing
	StructOps
	Extension
	LSM
	SkLookup
)

// AttachType of the eBPF program, needed to differentiate allowed context accesses in
// some newer program types like CGroupSockAddr. Should be set to AttachNone if not required.
// Will cause invalid argument (EINVAL) at program load time if set incorrectly.
type AttachType uint32

// AttachNone is an alias for AttachCGroupInetIngress for readability reasons.
const AttachNone AttachType = 0

const (
	AttachCGroupInetIngress AttachType = iota
	AttachCGroupInetEgress
	AttachCGroupInetSockCreate
	AttachCGroupSockOps
	AttachSkSKBStreamParser
	AttachSkSKBStreamVerdict
	AttachCGroupDevice
	AttachSkMsgVerdict
	AttachCGroupInet4Bind
	AttachCGroupInet6Bind
	AttachCGroupInet4Connect
	AttachCGroupInet6Connect
	AttachCGroupInet4PostBind
	AttachCGroupInet6PostBind
	AttachCGroupUDP4Sendmsg
	AttachCGroupUDP6Sendmsg
	AttachLircMode2
	AttachFlowDissector
	AttachCGroupSysctl
	AttachCGroupUDP4Recvmsg
	AttachCGroupUDP6Recvmsg
	AttachCGroupGetsockopt
	AttachCGroupSetsockopt
	AttachTraceRawTp
	AttachTraceFEntry
	AttachTraceFExit
	AttachModifyReturn
	AttachLSMMac
	AttachTraceIter
	AttachCgroupInet4GetPeername
	AttachCgroupInet6GetPeername
	AttachCgroupInet4GetSockname
	AttachCgroupInet6GetSockname
	AttachXDPDevMap
	AttachCgroupInetSockRelease
	AttachXDPCPUMap
	AttachSkLookup
	AttachXDP
)

// AttachFlags of the eBPF program used in BPF_PROG_ATTACH command
type AttachFlags uint32

// PinType determines whether a map is pinned into a BPFFS.
type PinType int

// Valid pin types.
//
// Mirrors enum libbpf_pin_type.
const (
	PinNone PinType = iota
	// Pin an object by using its name as the filename.
	PinByName
)

// LoadPinOptions control how a pinned object is loaded.
type LoadPinOptions struct {
	// Request a read-only or write-only object. The default is a read-write
	// object. Only one of the flags may be set.
	ReadOnly  bool
	WriteOnly bool

	// Raw flags for the syscall. Other fields of this struct take precedence.
	Flags uint32
}

// Marshal returns a value suitable for BPF_OBJ_GET syscall file_flags parameter.
func (lpo *LoadPinOptions) Marshal() uint32 {
	if lpo == nil {
		return 0
	}

	flags := lpo.Flags
	if lpo.ReadOnly {
		flags |= unix.BPF_F_RDONLY
	}
	if lpo.WriteOnly {
		flags |= unix.BPF_F_WRONLY
	}
	return flags
}

// BatchOptions batch map operations options
//
// Mirrors libbpf struct bpf_map_batch_opts
// Currently BPF_F_FLAG is the only supported
// flag (for ElemFlags).
type BatchOptions struct {
	ElemFlags uint64
	Flags     uint64
}
