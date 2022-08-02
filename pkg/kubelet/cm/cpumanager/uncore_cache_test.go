package cpumanager

import (
	"fmt"
	"math"
	"testing"
	"time"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

func TestTakeByTopologyUncoreCacheEnabledLegacy(t *testing.T) {
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		numCPUs       int
		expErr        string
		expResult     cpuset.CPUSet
	}{
		// None of the topologies in this test should have more than one uncore cache
		// e.g. the old tests should not change
		{
			"take more cpus than are available from single socket with HT",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 2, 4, 6),
			5,
			"not enough cpus available to satisfy request",
			cpuset.NewCPUSet(),
		},
		{
			"take zero cpus from single socket with HT",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			0,
			"",
			cpuset.NewCPUSet(),
		},
		{
			"take one cpu from single socket with HT",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			1,
			"",
			cpuset.NewCPUSet(0),
		},
		{
			"take one cpu from single socket with HT, some cpus are taken",
			topoSingleSocketHT,
			cpuset.NewCPUSet(1, 3, 5, 6, 7),
			1,
			"",
			cpuset.NewCPUSet(6),
		},
		{
			"take two cpus from single socket with HT",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			2,
			"",
			cpuset.NewCPUSet(0, 4),
		},
		{
			"take all cpus from single socket with HT",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			8,
			"",
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			"take two cpus from single socket with HT, only one core totally free",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 6),
			2,
			"",
			cpuset.NewCPUSet(2, 6),
		},
		{
			"take one cpu from dual socket with HT - core from Socket 0",
			topoDualSocketHT,
			cpuset.NewCPUSet(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			1,
			"",
			cpuset.NewCPUSet(2),
		},
		{
			"take a socket of cpus from dual socket with HT",
			topoDualSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			6,
			"",
			cpuset.NewCPUSet(0, 2, 4, 6, 8, 10),
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			result, err := takeByTopologyNUMAPacked(tc.topo, tc.availableCPUs, tc.numCPUs)
			if tc.expErr != "" && err.Error() != tc.expErr {
				t.Errorf("[%s] expected %v to equal %v", tc.description, err, tc.expErr)
			}
			if !result.Equals(tc.expResult) {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

var (
	// Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
	gold_5218_topology = &topology.CPUTopology{
		NumCPUs:    64,
		NumSockets: 2,
		NumCores:   16,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 0, SocketID: 1, UnCoreCacheID: 0},
			2:  {CoreID: 7, SocketID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 7, SocketID: 1, UnCoreCacheID: 0},
			4:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 1, SocketID: 1, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 6, SocketID: 1, UnCoreCacheID: 0},
			8:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 2, SocketID: 1, UnCoreCacheID: 0},
			10: {CoreID: 5, SocketID: 0, UnCoreCacheID: 0},
			11: {CoreID: 5, SocketID: 1, UnCoreCacheID: 0},
			12: {CoreID: 3, SocketID: 0, UnCoreCacheID: 0},
			13: {CoreID: 3, SocketID: 1, UnCoreCacheID: 0},
			14: {CoreID: 4, SocketID: 0, UnCoreCacheID: 0},
			15: {CoreID: 4, SocketID: 1, UnCoreCacheID: 0},
			16: {CoreID: 8, SocketID: 0, UnCoreCacheID: 0},
			17: {CoreID: 8, SocketID: 1, UnCoreCacheID: 0},
			18: {CoreID: 15, SocketID: 0, UnCoreCacheID: 0},
			19: {CoreID: 15, SocketID: 1, UnCoreCacheID: 0},
			20: {CoreID: 9, SocketID: 0, UnCoreCacheID: 0},
			21: {CoreID: 9, SocketID: 1, UnCoreCacheID: 0},
			22: {CoreID: 14, SocketID: 0, UnCoreCacheID: 0},
			23: {CoreID: 14, SocketID: 1, UnCoreCacheID: 0},
			24: {CoreID: 10, SocketID: 0, UnCoreCacheID: 0},
			25: {CoreID: 10, SocketID: 1, UnCoreCacheID: 0},
			26: {CoreID: 13, SocketID: 0, UnCoreCacheID: 0},
			27: {CoreID: 13, SocketID: 1, UnCoreCacheID: 0},
			28: {CoreID: 11, SocketID: 0, UnCoreCacheID: 0},
			29: {CoreID: 11, SocketID: 1, UnCoreCacheID: 0},
			30: {CoreID: 12, SocketID: 0, UnCoreCacheID: 0},
			31: {CoreID: 12, SocketID: 1, UnCoreCacheID: 0},
			32: {CoreID: 0, SocketID: 0, UnCoreCacheID: 1},
			33: {CoreID: 0, SocketID: 1, UnCoreCacheID: 1},
			34: {CoreID: 7, SocketID: 0, UnCoreCacheID: 1},
			35: {CoreID: 7, SocketID: 1, UnCoreCacheID: 1},
			36: {CoreID: 1, SocketID: 0, UnCoreCacheID: 1},
			37: {CoreID: 1, SocketID: 1, UnCoreCacheID: 1},
			38: {CoreID: 6, SocketID: 0, UnCoreCacheID: 1},
			39: {CoreID: 6, SocketID: 1, UnCoreCacheID: 1},
			40: {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
			41: {CoreID: 2, SocketID: 1, UnCoreCacheID: 1},
			42: {CoreID: 5, SocketID: 0, UnCoreCacheID: 1},
			43: {CoreID: 5, SocketID: 1, UnCoreCacheID: 1},
			44: {CoreID: 3, SocketID: 0, UnCoreCacheID: 1},
			45: {CoreID: 3, SocketID: 1, UnCoreCacheID: 1},
			46: {CoreID: 4, SocketID: 0, UnCoreCacheID: 1},
			47: {CoreID: 4, SocketID: 1, UnCoreCacheID: 1},
			48: {CoreID: 8, SocketID: 0, UnCoreCacheID: 1},
			49: {CoreID: 8, SocketID: 1, UnCoreCacheID: 1},
			50: {CoreID: 15, SocketID: 0, UnCoreCacheID: 1},
			51: {CoreID: 15, SocketID: 1, UnCoreCacheID: 1},
			52: {CoreID: 9, SocketID: 0, UnCoreCacheID: 1},
			53: {CoreID: 9, SocketID: 1, UnCoreCacheID: 1},
			54: {CoreID: 14, SocketID: 0, UnCoreCacheID: 1},
			55: {CoreID: 14, SocketID: 1, UnCoreCacheID: 1},
			56: {CoreID: 10, SocketID: 0, UnCoreCacheID: 1},
			57: {CoreID: 10, SocketID: 1, UnCoreCacheID: 1},
			58: {CoreID: 13, SocketID: 0, UnCoreCacheID: 1},
			59: {CoreID: 13, SocketID: 1, UnCoreCacheID: 1},
			60: {CoreID: 11, SocketID: 0, UnCoreCacheID: 1},
			61: {CoreID: 11, SocketID: 1, UnCoreCacheID: 1},
			62: {CoreID: 12, SocketID: 0, UnCoreCacheID: 1},
			63: {CoreID: 12, SocketID: 1, UnCoreCacheID: 1},
		},
	}

	AMD_EPYC_7502P_32_Core_Processor = &topology.CPUTopology{
		NumCPUs:      64,
		NumCores:     32,
		NumSockets:   1,
		NumNUMANodes: 1,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			32: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			33: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			34: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			35: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			36: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			37: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			38: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			39: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			40: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			41: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			10: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			42: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			11: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			43: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			12: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			44: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			13: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			45: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			14: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			46: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			15: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			47: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			16: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			48: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			17: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			49: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			18: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			50: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			19: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			51: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 4},
			20: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			52: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			21: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			53: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			22: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			54: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			23: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			55: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 5},
			24: {CoreID: 24, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			56: {CoreID: 24, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			25: {CoreID: 25, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			57: {CoreID: 25, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			26: {CoreID: 26, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			58: {CoreID: 26, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			27: {CoreID: 27, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			59: {CoreID: 27, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 6},
			28: {CoreID: 28, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			60: {CoreID: 28, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			29: {CoreID: 29, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			61: {CoreID: 29, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			30: {CoreID: 30, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			62: {CoreID: 30, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			31: {CoreID: 31, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
			63: {CoreID: 31, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 7},
		},
	}

	Intel_R__Xeon_R__Gold_5120_CPU___2_20GHz = &topology.CPUTopology{
		NumCPUs:      56,
		NumCores:     28,
		NumSockets:   2,
		NumNUMANodes: 2,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			28: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			30: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			32: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			34: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			36: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			10: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			38: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			12: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			40: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			14: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			42: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			16: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			44: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			18: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			46: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			20: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			48: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			22: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			50: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			24: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			52: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			26: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			54: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 14, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			29: {CoreID: 14, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			3:  {CoreID: 15, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			31: {CoreID: 15, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			5:  {CoreID: 16, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			33: {CoreID: 16, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			7:  {CoreID: 17, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			35: {CoreID: 17, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			9:  {CoreID: 18, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			37: {CoreID: 18, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			11: {CoreID: 19, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			39: {CoreID: 19, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			13: {CoreID: 20, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			41: {CoreID: 20, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			15: {CoreID: 21, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			43: {CoreID: 21, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			17: {CoreID: 22, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			45: {CoreID: 22, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			19: {CoreID: 23, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			47: {CoreID: 23, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			21: {CoreID: 24, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			49: {CoreID: 24, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			23: {CoreID: 25, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			51: {CoreID: 25, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			25: {CoreID: 26, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			53: {CoreID: 26, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			27: {CoreID: 27, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
			55: {CoreID: 27, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 1},
		},
	}
)

func takeIterator(topo *topology.CPUTopology, takeNumCpus []int, featureFlag bool, t *testing.T) string {
	// Isolate the take operations ensuring the feature flag is dis/enabled
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, featureFlag)()
	results := make(map[int]cpuset.CPUSet)
	cpuSet := topo.CPUDetails.CPUs()
	counter := 0
	for _, takeCpus := range takeNumCpus {
		took, err := takeByTopologyNUMAPacked(topo, cpuSet, takeCpus)
		if err != nil {
			return fmt.Sprint(err)
		}
		counter += 1
		results[counter] = took
		//fmt.Println("     TOOK:", took)
		cpuSet = cpuSet.Difference(took)
	}
	result := fmt.Sprint(results)
	return result
} // the above defer statement should be unwound here

func TestTakeOldVsNew(t *testing.T) {
	examples := []struct {
		topo        *topology.CPUTopology
		takeNumCpus []int
		expOld      string
		expNew      string
	}{
		{
			// This topology has eight UCCs on one NUMA (8x8 UCCs)
			AMD_EPYC_7502P_32_Core_Processor,
			// Take almost all the cpus per ucc per iteration (a degenerate, but interesting case)
			[]int{7, 7, 7, 7, 7, 7, 7, 7},
			// The old scheduler splits the cpus across ucc
			"map[1:0-3,32-34 2:4-6,35-38 3:7-10,39-41 4:11-13,42-45 5:14-17,46-48 6:18-20,49-52 7:21-24,53-55 8:25-27,56-59]",
			// The new scheduler prefers aligning - all takes below are from a unique ucc
			"map[1:0-3,32-34 2:4-7,36-38 3:8-11,40-42 4:12-15,44-46 5:16-19,48-50 6:20-23,52-54 7:24-27,56-58 8:28-31,60-62]",
		},
		{
			// This topology has two UCCs with matching NUMA (2x28 UCCs)
			Intel_R__Xeon_R__Gold_5120_CPU___2_20GHz,
			// Take an increasing number of cpus (in each iteration) to see the assignments
			[]int{4, 5, 6, 7, 8}, // this combination exercises the lower level "takers", leaves holes, etc.
			// The map below shows the "takes".  Notice the last take (8 cpus) yields from both UCC0 and UCC1
			"map[1:0,2,28,30 2:4,6,8,32,34 3:10,12,14,38,40,42 4:16,18,20,36,44,46,48 5:1,22,24,26,29,50,52,54]",
			// With the feature flag enabled the enhanced scheduler takes only from UCC1 to satisfy the entire request
			"map[1:0,2,28,30 2:4,6,8,32,34 3:10,12,14,38,40,42 4:16,18,20,36,44,46,48 5:1,3,5,7,29,31,33,35]",
		},
		{
			// Also has two uncore cahces
			gold_5218_topology,
			[]int{1, 2, 3, 4, 5, 6},
			"map[1:0 2:1,32 3:4-5,36 4:8-9,40-41 5:12-14,44-45 6:10-11,15,42-43,46]",
			"map[1:0 2:4-5 3:8-9,12 4:10-11,14-15 5:2-3,6-7,16 6:20-21,24-25,28-29]",
		},
		{
			gold_5218_topology,
			[]int{4, 5, 6, 7},
			"map[1:0-1,32-33 2:4-5,8,36-37 3:9,12-13,40,44-45 4:10-11,14-15,42,46-47]",
			"map[1:0-1,4-5 2:8-9,12-14 3:2-3,6-7,10-11 4:16-17,20-21,24-25,28]",
		},
	}

	for _, tc := range examples {
		oldResult := takeIterator(tc.topo, tc.takeNumCpus, false, t)
		if oldResult != tc.expOld {
			t.Errorf("\nEXP__OLD: %v\nTO EQUAL: %v", tc.expOld, oldResult)
		}
		newResult := takeIterator(tc.topo, tc.takeNumCpus, true, t)
		if newResult != tc.expNew {
			t.Errorf("\nEXP__NEW: %v\nTO EQUAL: %v", tc.expNew, newResult)
		}
	}
}

func makeReverseRange(min, max int) []int {
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = max - i
	}
	return a
}

func TestMakeReverseRange(t *testing.T) {
	number := 5
	result := makeReverseRange(0, number-1)
	for ndx, result := range result {
		fmt.Println(ndx, result)
		if result != number-ndx-1 {
			t.Errorf("EXPECTED: %v to equal %v", result, number-ndx-1)
			break
		}
	}
}

func AllocationPermutations(number int) [][]int {
	nMax := int(math.Pow(2, float64(number-1)))
	returnListList := make([][]int, nMax)
	counter := 0
	takeAll := make([]int, 1)
	takeAll[0] = number
	returnListList[counter] = takeAll
	counter += 1
	for _, n := range makeReverseRange(1, number-1) {
		recurse := AllocationPermutations(number - n)
		for _, rval := range recurse {
			x := make([]int, 1+len(rval))
			x[0] = n
			ndx := 1
			for _, r := range rval {
				x[ndx] = r
				ndx += 1
			}
			returnListList[counter] = x
			counter += 1
		}
	}
	return returnListList
}

func TestAllocationPermutations(t *testing.T) {
	number := 5
	then := time.Now()
	listOfListOfInt := AllocationPermutations(number)
	now := time.Now()
	fmt.Println(len(listOfListOfInt), now.Sub(then))
	for ndx, value := range listOfListOfInt {
		fmt.Println(ndx, value)
	}
	firstAllocation := listOfListOfInt[0]
	fmt.Println(firstAllocation)
	if len(firstAllocation) != 1 {
		t.Errorf("Expected %v to equal %v", len(firstAllocation), 1)
		return
	}
	if firstAllocation[0] != number {
		t.Errorf("Expected %v to equal %v", firstAllocation[0], number)
		return
	}
	lastAllocation := listOfListOfInt[len(listOfListOfInt)-1]
	fmt.Println(lastAllocation)
	if len(lastAllocation) != number {
		t.Errorf("Expected %v to equal %v", len(lastAllocation), number)
		return
	}
	for ndx, result := range lastAllocation {
		if result != 1 {
			t.Errorf("Expected result[%v], %v to equal %v", ndx, result, 1)
			return
		}
	}
}

type TakeTally struct {
	Take cpuset.CPUSet
	Topo *topology.CPUTopology
}

func (tt *TakeTally) IsSplitAcrossUncoreCaches() bool {
	lastUcc := -1
	isSplit := false
	for _, cpuid := range tt.Take.ToSliceNoSort() {
		ucc := tt.Topo.CPUDetails[cpuid].UnCoreCacheID
		if lastUcc == -1 {
			lastUcc = ucc
		} else {
			if ucc != lastUcc {
				isSplit = true
				break
			}
		}
	}
	return isSplit
}

func TakePattern(topo *topology.CPUTopology, pattern []int, featureFlag bool, t *testing.T) string {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, featureFlag)()
	retStr := ""
	cpuSet := topo.CPUDetails.CPUs()
	retTally := make(map[int]TakeTally)
	patternNdx := 0
	for { // cycle through the pattern until we run out of possible allocations
		takeCpus := pattern[patternNdx%len(pattern)]
		patternNdx += 1
		if takeCpus > cpuSet.Size() {
			break
		}
		// invoke the scheduler
		take, err := takeByTopologyNUMAPacked(topo, cpuSet, takeCpus)
		if err != nil {
			t.Errorf("\nUNEXPECTED TAKE ERROR: %v", err)
			return fmt.Sprint(err)
		}
		cpuSet = cpuSet.Difference(take)
		tt := TakeTally{Take: take, Topo: topo}
		retTally[len(retTally)] = tt
		if tt.IsSplitAcrossUncoreCaches() {
			retStr = retStr + "X"
		} else {
			retStr = retStr + "."
		}
	}
	return retStr
} // the above defer statement should be unwound here

func TestTakeAllocationPermutations(t *testing.T) {
	examples := []struct {
		topo     *topology.CPUTopology
		apNumber int
		expOld   []string
		expNew   []string
	}{
		{
			// This topology has eight UCCs on one NUMA (8x8 UCCs)
			AMD_EPYC_7502P_32_Core_Processor,
			// This should be one more than half the number of cpus per uncore cache
			5,
			// The old scheduler splits the cpus across ucc
			[]string{".X.XX.X..X.X",
				"..X.........X.....X......",
				"......X...X.X.........X..",
				".........X........X..............X....",
				"...X...X.X.........X...X.",
				"......................................",
				"......................................",
				"...................................................",
				"...X.....X.........X.....",
				"....X..............X........X.........",
				"......................................",
				"...................................................",
				".....X........X..............X........",
				"...................................................",
				"...................................................",
				"................................................................",
				""},
			// The new scheduler prefers aligning, unless it is impossible
			[]string{"........XXXX",
				".........................",
				"........................X",
				"......................................",
				"........................X",
				"......................................",
				"......................................",
				"...................................................",
				".........................",
				"......................................",
				"......................................",
				"...................................................",
				"......................................",
				"...................................................",
				"...................................................",
				"................................................................",
				""},
		},
	}
	for _, tc := range examples {
		aps := AllocationPermutations(tc.apNumber)
		counter := 0
		for _, perm := range aps {
			oldResult := TakePattern(tc.topo, perm, false, t)
			if oldResult != tc.expOld[counter] {
				t.Errorf("\nexpOld[%v]: %v\n   result: %v", counter, tc.expOld[counter], oldResult)
				return
			}
			newResult := TakePattern(tc.topo, perm, true, t)
			if newResult != tc.expNew[counter] {
				t.Errorf("\nexpNew[%v]: %v\n   result: %v", counter, tc.expNew[counter], newResult)
				return
			}
			counter += 1
		}
	}
}
