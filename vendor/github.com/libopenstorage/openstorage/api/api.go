package api

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Strings for VolumeSpec
const (
	Name                 = "name"
	SpecEphemeral        = "ephemeral"
	SpecShared           = "shared"
	SpecSize             = "size"
	SpecScale            = "scale"
	SpecFilesystem       = "fs"
	SpecBlockSize        = "block_size"
	SpecHaLevel          = "repl"
	SpecPriority         = "io_priority"
	SpecSnapshotInterval = "snap_interval"
	SpecAggregationLevel = "aggregation_level"
	SpecDedupe           = "dedupe"
	SpecPassphrase       = "passphrase"
)

// OptionKey specifies a set of recognized query params.
const (
	// OptName query parameter used to lookup volume by name.
	OptName = "Name"
	// OptVolumeID query parameter used to lookup volume by ID.
	OptVolumeID = "VolumeID"
	// OptLabel query parameter used to lookup volume by set of labels.
	OptLabel = "Label"
	// OptConfigLabel query parameter used to lookup volume by set of labels.
	OptConfigLabel = "ConfigLabel"
	// OptCumulative query parameter used to request cumulative stats.
	OptCumulative = "Cumulative"
)

// Api client-server Constants
const (
	OsdVolumePath = "osd-volumes"
	OsdSnapshotPath = "osd-snapshot"
)
// Node describes the state of a node.
// It includes the current physical state (CPU, memory, storage, network usage) as
// well as the containers running on the system.
type Node struct {
	Id        string
	Cpu       float64 // percentage.
	MemTotal  uint64
	MemUsed   uint64
	MemFree   uint64
	Avgload   int
	Status    Status
	GenNumber uint64
	Disks     map[string]StorageResource
	MgmtIp    string
	DataIp    string
	Timestamp time.Time
	StartTime time.Time
	Hostname  string
	NodeData  map[string]interface{}
	// User defined labels for node. Key Value pairs
	NodeLabels map[string]string
}

// Cluster represents the state of the cluster.
type Cluster struct {
	Status Status

	// Id is the ID of the cluster.
	Id string

	// NodeId is the ID of the node on which this cluster object
	// is initialized
	NodeId string

	// Nodes is an array of all the nodes in the cluster.
	Nodes []Node
}

// StatPoint represents the basic structure of a single Stat reported
// TODO: This is the first step to introduce stats in openstorage.
//       Follow up task is to introduce an API for logging stats
type StatPoint struct {
	// Name of the Stat
	Name string
	// Tags for the Stat
	Tags map[string]string
	// Fields and values of the stat
	Fields map[string]interface{}
	// Timestamp in Unix format
	Timestamp int64
}

func DriverTypeSimpleValueOf(s string) (DriverType, error) {
	obj, err := simpleValueOf("driver_type", DriverType_value, s)
	return DriverType(obj), err
}

func (x DriverType) SimpleString() string {
	return simpleString("driver_type", DriverType_name, int32(x))
}

func FSTypeSimpleValueOf(s string) (FSType, error) {
	obj, err := simpleValueOf("fs_type", FSType_value, s)
	return FSType(obj), err
}

func (x FSType) SimpleString() string {
	return simpleString("fs_type", FSType_name, int32(x))
}

func CosTypeSimpleValueOf(s string) (CosType, error) {
	obj, err := simpleValueOf("cos_type", CosType_value, s)
	return CosType(obj), err
}

func (x CosType) SimpleString() string {
	return simpleString("cos_type", CosType_name, int32(x))
}

func GraphDriverChangeTypeSimpleValueOf(s string) (GraphDriverChangeType, error) {
	obj, err := simpleValueOf("graph_driver_change_type", GraphDriverChangeType_value, s)
	return GraphDriverChangeType(obj), err
}

func (x GraphDriverChangeType) SimpleString() string {
	return simpleString("graph_driver_change_type", GraphDriverChangeType_name, int32(x))
}

func VolumeActionParamSimpleValueOf(s string) (VolumeActionParam, error) {
	obj, err := simpleValueOf("volume_action_param", VolumeActionParam_value, s)
	return VolumeActionParam(obj), err
}

func (x VolumeActionParam) SimpleString() string {
	return simpleString("volume_action_param", VolumeActionParam_name, int32(x))
}

func VolumeStateSimpleValueOf(s string) (VolumeState, error) {
	obj, err := simpleValueOf("volume_state", VolumeState_value, s)
	return VolumeState(obj), err
}

func (x VolumeState) SimpleString() string {
	return simpleString("volume_state", VolumeState_name, int32(x))
}

func VolumeStatusSimpleValueOf(s string) (VolumeStatus, error) {
	obj, err := simpleValueOf("volume_status", VolumeStatus_value, s)
	return VolumeStatus(obj), err
}

func (x VolumeStatus) SimpleString() string {
	return simpleString("volume_status", VolumeStatus_name, int32(x))
}

func simpleValueOf(typeString string, valueMap map[string]int32, s string) (int32, error) {
	obj, ok := valueMap[strings.ToUpper(fmt.Sprintf("%s_%s", typeString, s))]
	if !ok {
		return 0, fmt.Errorf("no openstorage.%s for %s", strings.ToUpper(typeString), s)
	}
	return obj, nil
}

func simpleString(typeString string, nameMap map[int32]string, v int32) string {
	s, ok := nameMap[v]
	if !ok {
		return strconv.Itoa(int(v))
	}
	return strings.TrimPrefix(strings.ToLower(s), fmt.Sprintf("%s_", strings.ToLower(typeString)))
}

func toSec(ms uint64) uint64 {
	return ms / 1000
}

func (v *Stats) WriteThroughput() uint64 {
	if v.IntervalMs == 0 {
		return 0
	}
	return (v.WriteBytes) / toSec(v.IntervalMs)
}

func (v *Stats) ReadThroughput() uint64 {
	if v.IntervalMs == 0 {
		return 0
	}
	return (v.ReadBytes) / toSec(v.IntervalMs)
}

func (v *Stats) Latency() uint64 {
	ops := v.Writes + v.Reads
	if ops == 0 {
		return 0
	}
	return (uint64)((v.IoMs * 1000) / (v.Writes + v.Reads))
}

func (v *Stats) Iops() uint64 {
	if v.IntervalMs == 0 {
		return 0
	}
	return (v.Writes + v.Reads) / toSec(v.IntervalMs)
}
