package winstats

import (
	"fmt"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"testing"
	"unsafe"
)

func TestGROUP_AFFINITY_Processors(t *testing.T) {
	tests := []struct {
		name  string
		Mask  uintptr
		Group uint16
		want  []int
	}{
		{
			name:  "empty",
			Mask:  0,
			Group: 0,
			want:  []int{},
		},
		{
			name:  "empty group 2",
			Mask:  0,
			Group: 1,
			want:  []int{},
		},
		{
			name:  "cpu 1 Group 0",
			Mask:  1,
			Group: 0,
			want:  []int{0},
		},
		{
			name:  "cpu 64 Group 0",
			Mask:  1 << 63,
			Group: 0,
			want:  []int{63},
		},
		{
			name:  "cpu 128 Group 1",
			Mask:  1 << 63,
			Group: 1,
			want:  []int{127},
		},
		{
			name:  "cpu 128 (Group 1)",
			Mask:  1 << 63,
			Group: 1,
			want:  []int{127},
		},
		{
			name:  "Mask 1 Group 2",
			Mask:  1,
			Group: 2,
			want:  []int{128},
		},
		{
			name:  "64 cpus group 0",
			Mask:  0xffffffffffffffff,
			Group: 0,
			want:  makeRange(0, 63),
		},
		{
			name:  "64 cpus group 1",
			Mask:  0xffffffffffffffff,
			Group: 1,
			want:  makeRange(64, 127),
		},
		{
			name:  "64 cpus group 1",
			Mask:  0xffffffffffffffff,
			Group: 1,
			want:  makeRange(64, 127),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := GROUP_AFFINITY{
				Mask:  tt.Mask,
				Group: tt.Group,
			}
			assert.Equalf(t, tt.want, a.Processors(), "Processors()")
		})
	}
}

// https://stackoverflow.com/a/39868255/697126
func makeRange(min, max int) []int {
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}

func TestCpusToGroupAffinity(t *testing.T) {
	tests := []struct {
		name string
		cpus []int
		want map[int]*GROUP_AFFINITY
	}{
		{
			name: "empty",
			want: map[int]*GROUP_AFFINITY{},
		},
		{
			name: "single cpu group 0",
			cpus: []int{0},
			want: map[int]*GROUP_AFFINITY{
				0: {
					Mask:  1,
					Group: 0,
				},
			},
		},
		{
			name: "single cpu group 0",
			cpus: []int{63},
			want: map[int]*GROUP_AFFINITY{
				0: {
					Mask:  1 << 63,
					Group: 0,
				},
			},
		},
		{
			name: "single cpu group 1",
			cpus: []int{64},
			want: map[int]*GROUP_AFFINITY{
				1: {
					Mask:  1,
					Group: 1,
				},
			},
		},
		{
			name: "multiple cpus same group",
			cpus: []int{0, 1, 2},
			want: map[int]*GROUP_AFFINITY{
				0: {
					Mask:  1 | 2 | 4, // Binary OR to combine the masks
					Group: 0,
				},
			},
		},
		{
			name: "multiple cpus different groups",
			cpus: []int{0, 64},
			want: map[int]*GROUP_AFFINITY{
				0: {
					Mask:  1,
					Group: 0,
				},
				1: {
					Mask:  1,
					Group: 1,
				},
			},
		},
		{
			name: "multiple cpus different groups",
			cpus: []int{0, 1, 2, 64, 65, 66},
			want: map[int]*GROUP_AFFINITY{
				0: {
					Mask:  1 | 2 | 4,
					Group: 0,
				},
				1: {
					Mask:  1 | 2 | 4,
					Group: 1,
				},
			},
		},
		{
			name: "64 cpus group 0",
			cpus: makeRange(0, 63),
			want: map[int]*GROUP_AFFINITY{
				0: {
					Mask:  0xffffffffffffffff, // All 64 bits set
					Group: 0,
				},
			},
		},
		{
			name: "64 cpus group 1",
			cpus: makeRange(64, 127),
			want: map[int]*GROUP_AFFINITY{
				1: {
					Mask:  0xffffffffffffffff, // All 64 bits set
					Group: 1,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equalf(t, tt.want, CpusToGroupAffinity(tt.cpus), "CpusToGroupAffinity(%v)", tt.cpus)
		})
	}
}

func Test_convertWinApiToCadvisorApi(t *testing.T) {
	tests := []struct {
		name                 string
		buffer               []byte
		expectedNumOfCores   int
		expectedNumOfSockets int
		expectedNodes        []cadvisorapi.Node
		wantErr              assert.ErrorAssertionFunc
	}{
		{
			name:                 "empty",
			buffer:               []byte{},
			expectedNumOfCores:   0,
			expectedNumOfSockets: 0,
			expectedNodes:        []cadvisorapi.Node{},
			wantErr:              assert.NoError,
		},
		{
			name:                 "single core",
			buffer:               createProcessorRelationships([]int{0}),
			expectedNumOfCores:   1,
			expectedNumOfSockets: 1,
			expectedNodes: []cadvisorapi.Node{
				{
					Id: 0,
					Cores: []cadvisorapi.Core{
						{
							Id:      1,
							Threads: []int{0},
						},
					},
				},
			},
			wantErr: assert.NoError,
		},
		{
			name:                 "single core, multiple cpus",
			buffer:               createProcessorRelationships([]int{0, 1, 2}),
			expectedNumOfCores:   1,
			expectedNumOfSockets: 1,
			expectedNodes: []cadvisorapi.Node{
				{
					Id: 0,
					Cores: []cadvisorapi.Core{
						{
							Id:      1,
							Threads: []int{0, 1, 2},
						},
					},
				},
			},
			wantErr: assert.NoError,
		},
		{
			name:                 "single core, multiple groups",
			buffer:               createProcessorRelationships([]int{0, 64}),
			expectedNumOfCores:   1,
			expectedNumOfSockets: 1,
			expectedNodes: []cadvisorapi.Node{
				{
					Id: 0,
					Cores: []cadvisorapi.Core{
						{
							Id:      1,
							Threads: []int{0, 64},
						},
					},
				},
			},
			wantErr: assert.NoError,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			numOfCores, numOfSockets, nodes, err := convertWinApiToCadvisorApi(tt.buffer)
			if !tt.wantErr(t, err, fmt.Sprintf("convertWinApiToCadvisorApi(%v)", tt.name)) {
				return
			}
			assert.Equalf(t, tt.expectedNumOfCores, numOfCores, "num of cores")
			assert.Equalf(t, tt.expectedNumOfSockets, numOfSockets, "num of sockets")
			for node := range nodes {
				assert.Equalf(t, tt.expectedNodes[node].Id, nodes[node].Id, "node id")
				for core := range nodes[node].Cores {
					assert.Equalf(t, tt.expectedNodes[node].Cores[core].Id, nodes[node].Cores[core].Id, "core id")
					assert.Equalf(t, tt.expectedNodes[node].Cores[core].Threads, nodes[node].Cores[core].Threads, "threads")
				}
			}
		})
	}
}

func genbuffer(infos ...SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) []byte {
	var buffer []byte
	for _, info := range infos {
		buffer = append(buffer, structToBytes(info)...)
	}
	return buffer
}

func createProcessorRelationships(cpus []int) []byte {
	groups := CpusToGroupAffinity(cpus)
	grouplen := len(groups)
	groupAffinities := make([]GROUP_AFFINITY, 0, grouplen)
	for _, group := range groups {
		groupAffinities = append(groupAffinities, *group)
	}
	return genbuffer(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX{
		Relationship: uint32(RelationProcessorCore),
		Size:         uint32(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_SIZE + PROCESSOR_RELATIONSHIP_SIZE + (GROUP_AFFINITY_SIZE * grouplen)),
		data: PROCESSOR_RELATIONSHIP{
			Flags:           0,
			EfficiencyClass: 0,
			Reserved:        [20]byte{},
			GroupCount:      uint16(grouplen),
			GroupMasks:      groupAffinities,
		},
	}, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX{
		Relationship: uint32(RelationNumaNode),
		Size:         uint32(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_SIZE + NUMA_NODE_RELATIONSHIP_SIZE + (GROUP_AFFINITY_SIZE * grouplen)),
		data: NUMA_NODE_RELATIONSHIP{
			NodeNumber: 0,
			Reserved:   [18]byte{},
			GroupCount: uint16(grouplen),
			GroupMasks: groupAffinities,
		}}, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX{
		Relationship: uint32(RelationProcessorPackage),
		Size:         uint32(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_SIZE + PROCESSOR_RELATIONSHIP_SIZE + (GROUP_AFFINITY_SIZE * grouplen)),
		data: PROCESSOR_RELATIONSHIP{
			Flags:           0,
			EfficiencyClass: 0,
			Reserved:        [20]byte{},
			GroupCount:      uint16(grouplen),
			GroupMasks:      groupAffinities,
		},
	})
}

const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_SIZE = 8
const PROCESSOR_RELATIONSHIP_SIZE = 24
const NUMA_NODE_RELATIONSHIP_SIZE = 24
const GROUP_AFFINITY_SIZE = int(unsafe.Sizeof(GROUP_AFFINITY{})) // this one is known at compile time

func structToBytes(info SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) []byte {
	var pri []byte = (*(*[SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_SIZE]byte)(unsafe.Pointer(&info)))[:SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX_SIZE]

	switch info.data.(type) {
	case PROCESSOR_RELATIONSHIP:
		rel := info.data.(PROCESSOR_RELATIONSHIP)
		var prBytes []byte = (*(*[PROCESSOR_RELATIONSHIP_SIZE]byte)(unsafe.Pointer(&rel)))[:PROCESSOR_RELATIONSHIP_SIZE]
		pri = append(pri, prBytes...)

		groupAffinities := rel.GroupMasks.([]GROUP_AFFINITY)

		for _, groupAffinity := range groupAffinities {
			var groupByte []byte = (*(*[GROUP_AFFINITY_SIZE]byte)(unsafe.Pointer(&groupAffinity)))[:]
			pri = append(pri, groupByte...)
		}
	case NUMA_NODE_RELATIONSHIP:
		numa := info.data.(NUMA_NODE_RELATIONSHIP)
		var nameBytes []byte = (*(*[NUMA_NODE_RELATIONSHIP_SIZE]byte)(unsafe.Pointer(&numa)))[:NUMA_NODE_RELATIONSHIP_SIZE]
		pri = append(pri, nameBytes...)

		groupAffinities := numa.GroupMasks.([]GROUP_AFFINITY)

		for _, groupAffinity := range groupAffinities {
			var groupByte []byte = (*(*[GROUP_AFFINITY_SIZE]byte)(unsafe.Pointer(&groupAffinity)))[:]
			pri = append(pri, groupByte...)
		}
	}

	return pri
}
