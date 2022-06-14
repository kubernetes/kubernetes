package cpumanager

import (
	"fmt"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

func cpusetForCPUTopology(topo *topology.CPUTopology) cpuset.CPUSet {
	// Many of the test cases use cpuset.CPUSet based on the topology
	elems := makeRange(0, topo.NumCPUs)
	result := cpuset.NewCPUSet(elems...)
	// TODO filters and masks?
	return result
}

func makeRange(min, max int) []int {
	// from Stack Overflow:
	// There is no equivalent to PHP's range in the Go standard library.
	// You have to create one yourself.                               <-- NEAT!
	// The simplest is to use a for loop:
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}

func TestCPUAccumulatorFreeCPUsUncoreCacheEnabledLegacy(t *testing.T) {
	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		availableCPUs cpuset.CPUSet
		expect        []int
	}{
		{
			"single socket HT, 8 cpus free",
			topoSingleSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			[]int{0, 4, 1, 5, 2, 6, 3, 7},
		},
		{
			"single socket HT, 5 cpus free",
			topoSingleSocketHT,
			cpuset.NewCPUSet(3, 4, 5, 6, 7),
			[]int{4, 5, 6, 3, 7},
		},
		{
			"dual socket HT, 12 cpus free",
			topoDualSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11},
		},
		{
			"dual socket HT, 11 cpus free",
			topoDualSocketHT,
			cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			[]int{6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11},
		},
		{
			"dual socket HT, 10 cpus free",
			topoDualSocketHT,
			cpuset.NewCPUSet(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			[]int{2, 8, 4, 10, 1, 7, 3, 9, 5, 11},
		},
		{
			"dual socket HT, 10 cpus free",
			topoDualSocketHT,
			cpuset.NewCPUSet(0, 1, 2, 3, 4, 6, 7, 8, 9, 10),
			[]int{1, 7, 3, 9, 0, 6, 2, 8, 4, 10},
		},
	}
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			acc := newCPUAccumulator(tc.topo, tc.availableCPUs, 0)
			result := acc.freeCPUs()
			if !reflect.DeepEqual(result, tc.expect) {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expect)
			}
		})
	}
}

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
		// Apply t.Run() test pattern
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
	topoDualUncoreCacheSingleSocketHT = &topology.CPUTopology{
		NumCPUs:    16,
		NumSockets: 1,
		NumCores:   8,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 3, SocketID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 3, SocketID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 4, SocketID: 0, UnCoreCacheID: 1},
			9:  {CoreID: 4, SocketID: 0, UnCoreCacheID: 1},
			10: {CoreID: 5, SocketID: 0, UnCoreCacheID: 1},
			11: {CoreID: 5, SocketID: 0, UnCoreCacheID: 1},
			12: {CoreID: 6, SocketID: 0, UnCoreCacheID: 1},
			13: {CoreID: 6, SocketID: 0, UnCoreCacheID: 1},
			14: {CoreID: 7, SocketID: 0, UnCoreCacheID: 1},
			15: {CoreID: 7, SocketID: 0, UnCoreCacheID: 1},
		},
	}

	// FIXME comment from jfbai: topoDualUncoreCacheSingleSocketHT = &topology.CPUTopology{
	// NumCPUs: 12,
	// NumSockets: 2,
	// NumCores: 6,
	// NumUnCoreCaches: 4,
	// CPUDetails: map[int]topology.CPUInfo{
	//  0: {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
	//  1: {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
	//  2: {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
	//  3: {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
	//  4: {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
	//  5: {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
	//  6: {CoreID: 3, SocketID: 1, UnCoreCacheID: 8},
	//  7: {CoreID: 3, SocketID: 1, UnCoreCacheID: 8},
	//  8: {CoreID: 4, SocketID: 1, UnCoreCacheID: 8},
	//  9: {CoreID: 4, SocketID: 1, UnCoreCacheID: 8},
	// 10: {CoreID: 5, SocketID: 1, UnCoreCacheID: 9},
	// 11: {CoreID: 5, SocketID: 1, UnCoreCacheID: 9}, }, }

	topoFROMjfbai = &topology.CPUTopology{
		NumCPUs:    12,
		NumSockets: 2,
		NumCores:   6,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
			5:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
			6:  {CoreID: 3, SocketID: 1, UnCoreCacheID: 8},
			7:  {CoreID: 3, SocketID: 1, UnCoreCacheID: 8},
			8:  {CoreID: 4, SocketID: 1, UnCoreCacheID: 8},
			9:  {CoreID: 4, SocketID: 1, UnCoreCacheID: 8},
			10: {CoreID: 5, SocketID: 1, UnCoreCacheID: 9},
			11: {CoreID: 5, SocketID: 1, UnCoreCacheID: 9},
		},
	}
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
	// AMD EPYC 7402P 24-Core Processor
	epyc_7402p_topology = &topology.CPUTopology{
		NumCPUs:    48,
		NumSockets: 1,
		NumCores:   24,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 4, SocketID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 5, SocketID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 6, SocketID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 8, SocketID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 9, SocketID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 10, SocketID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 12, SocketID: 0, UnCoreCacheID: 0},
			10: {CoreID: 13, SocketID: 0, UnCoreCacheID: 0},
			11: {CoreID: 14, SocketID: 0, UnCoreCacheID: 0},
			12: {CoreID: 16, SocketID: 0, UnCoreCacheID: 0},
			13: {CoreID: 17, SocketID: 0, UnCoreCacheID: 0},
			14: {CoreID: 18, SocketID: 0, UnCoreCacheID: 0},
			15: {CoreID: 20, SocketID: 0, UnCoreCacheID: 0},
			16: {CoreID: 21, SocketID: 0, UnCoreCacheID: 0},
			17: {CoreID: 22, SocketID: 0, UnCoreCacheID: 0},
			18: {CoreID: 24, SocketID: 0, UnCoreCacheID: 0},
			19: {CoreID: 25, SocketID: 0, UnCoreCacheID: 0},
			20: {CoreID: 26, SocketID: 0, UnCoreCacheID: 0},
			21: {CoreID: 28, SocketID: 0, UnCoreCacheID: 0},
			22: {CoreID: 29, SocketID: 0, UnCoreCacheID: 0},
			23: {CoreID: 30, SocketID: 0, UnCoreCacheID: 0},
			24: {CoreID: 0, SocketID: 0, UnCoreCacheID: 1},
			25: {CoreID: 1, SocketID: 0, UnCoreCacheID: 1},
			26: {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
			27: {CoreID: 4, SocketID: 0, UnCoreCacheID: 1},
			28: {CoreID: 5, SocketID: 0, UnCoreCacheID: 1},
			29: {CoreID: 6, SocketID: 0, UnCoreCacheID: 1},
			30: {CoreID: 8, SocketID: 0, UnCoreCacheID: 1},
			31: {CoreID: 9, SocketID: 0, UnCoreCacheID: 1},
			32: {CoreID: 10, SocketID: 0, UnCoreCacheID: 1},
			33: {CoreID: 12, SocketID: 0, UnCoreCacheID: 1},
			34: {CoreID: 13, SocketID: 0, UnCoreCacheID: 1},
			35: {CoreID: 14, SocketID: 0, UnCoreCacheID: 1},
			36: {CoreID: 16, SocketID: 0, UnCoreCacheID: 1},
			37: {CoreID: 17, SocketID: 0, UnCoreCacheID: 1},
			38: {CoreID: 18, SocketID: 0, UnCoreCacheID: 1},
			39: {CoreID: 20, SocketID: 0, UnCoreCacheID: 1},
			40: {CoreID: 21, SocketID: 0, UnCoreCacheID: 1},
			41: {CoreID: 22, SocketID: 0, UnCoreCacheID: 1},
			42: {CoreID: 24, SocketID: 0, UnCoreCacheID: 1},
			43: {CoreID: 25, SocketID: 0, UnCoreCacheID: 1},
			44: {CoreID: 26, SocketID: 0, UnCoreCacheID: 1},
			45: {CoreID: 28, SocketID: 0, UnCoreCacheID: 1},
			46: {CoreID: 29, SocketID: 0, UnCoreCacheID: 1},
			47: {CoreID: 30, SocketID: 0, UnCoreCacheID: 1},
		},
	}
	// EPYC 7502P 32
	epyc_7502p_32_topology = &topology.CPUTopology{
		NumCPUs:    64,
		NumSockets: 1,
		NumCores:   32,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, UnCoreCacheID: 1},
			5:  {CoreID: 5, SocketID: 0, UnCoreCacheID: 1},
			6:  {CoreID: 6, SocketID: 0, UnCoreCacheID: 1},
			7:  {CoreID: 7, SocketID: 0, UnCoreCacheID: 1},
			8:  {CoreID: 8, SocketID: 0, UnCoreCacheID: 2},
			9:  {CoreID: 9, SocketID: 0, UnCoreCacheID: 2},
			10: {CoreID: 10, SocketID: 0, UnCoreCacheID: 2},
			11: {CoreID: 11, SocketID: 0, UnCoreCacheID: 2},
			12: {CoreID: 12, SocketID: 0, UnCoreCacheID: 3},
			13: {CoreID: 13, SocketID: 0, UnCoreCacheID: 3},
			14: {CoreID: 14, SocketID: 0, UnCoreCacheID: 3},
			15: {CoreID: 15, SocketID: 0, UnCoreCacheID: 3},
			16: {CoreID: 16, SocketID: 0, UnCoreCacheID: 4},
			17: {CoreID: 17, SocketID: 0, UnCoreCacheID: 4},
			18: {CoreID: 18, SocketID: 0, UnCoreCacheID: 4},
			19: {CoreID: 19, SocketID: 0, UnCoreCacheID: 4},
			20: {CoreID: 20, SocketID: 0, UnCoreCacheID: 5},
			21: {CoreID: 21, SocketID: 0, UnCoreCacheID: 5},
			22: {CoreID: 22, SocketID: 0, UnCoreCacheID: 5},
			23: {CoreID: 23, SocketID: 0, UnCoreCacheID: 5},
			24: {CoreID: 24, SocketID: 0, UnCoreCacheID: 6},
			25: {CoreID: 25, SocketID: 0, UnCoreCacheID: 6},
			26: {CoreID: 26, SocketID: 0, UnCoreCacheID: 6},
			27: {CoreID: 27, SocketID: 0, UnCoreCacheID: 6},
			28: {CoreID: 28, SocketID: 0, UnCoreCacheID: 7},
			29: {CoreID: 29, SocketID: 0, UnCoreCacheID: 7},
			30: {CoreID: 30, SocketID: 0, UnCoreCacheID: 7},
			31: {CoreID: 31, SocketID: 0, UnCoreCacheID: 7},
			32: {CoreID: 0, SocketID: 0, UnCoreCacheID: 1},
			33: {CoreID: 1, SocketID: 0, UnCoreCacheID: 1},
			34: {CoreID: 2, SocketID: 0, UnCoreCacheID: 1},
			35: {CoreID: 3, SocketID: 0, UnCoreCacheID: 1},
			36: {CoreID: 4, SocketID: 0, UnCoreCacheID: 1},
			37: {CoreID: 5, SocketID: 0, UnCoreCacheID: 1},
			38: {CoreID: 6, SocketID: 0, UnCoreCacheID: 1},
			39: {CoreID: 7, SocketID: 0, UnCoreCacheID: 1},
			40: {CoreID: 8, SocketID: 0, UnCoreCacheID: 1},
			41: {CoreID: 9, SocketID: 0, UnCoreCacheID: 1},
			42: {CoreID: 10, SocketID: 0, UnCoreCacheID: 1},
			43: {CoreID: 11, SocketID: 0, UnCoreCacheID: 1},
			44: {CoreID: 12, SocketID: 0, UnCoreCacheID: 1},
			45: {CoreID: 13, SocketID: 0, UnCoreCacheID: 1},
			46: {CoreID: 14, SocketID: 0, UnCoreCacheID: 1},
			47: {CoreID: 15, SocketID: 0, UnCoreCacheID: 1},
			48: {CoreID: 16, SocketID: 0, UnCoreCacheID: 1},
			49: {CoreID: 17, SocketID: 0, UnCoreCacheID: 1},
			50: {CoreID: 18, SocketID: 0, UnCoreCacheID: 1},
			51: {CoreID: 19, SocketID: 0, UnCoreCacheID: 1},
			52: {CoreID: 20, SocketID: 0, UnCoreCacheID: 1},
			53: {CoreID: 21, SocketID: 0, UnCoreCacheID: 1},
			54: {CoreID: 22, SocketID: 0, UnCoreCacheID: 1},
			55: {CoreID: 23, SocketID: 0, UnCoreCacheID: 1},
			56: {CoreID: 24, SocketID: 0, UnCoreCacheID: 1},
			57: {CoreID: 25, SocketID: 0, UnCoreCacheID: 1},
			58: {CoreID: 26, SocketID: 0, UnCoreCacheID: 1},
			59: {CoreID: 27, SocketID: 0, UnCoreCacheID: 1},
			60: {CoreID: 28, SocketID: 0, UnCoreCacheID: 1},
			61: {CoreID: 29, SocketID: 0, UnCoreCacheID: 1},
			62: {CoreID: 30, SocketID: 0, UnCoreCacheID: 1},
			63: {CoreID: 31, SocketID: 0, UnCoreCacheID: 1},
		},
	}
	epyc_7513 = &topology.CPUTopology{
		NumCPUs:      64,
		NumSockets:   1,
		NumNUMANodes: 1,
		NumCores:     32,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			10: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			11: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			12: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			13: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			14: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			15: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			16: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			17: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			18: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			19: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			20: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			21: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			22: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			23: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			24: {CoreID: 24, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			25: {CoreID: 25, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			26: {CoreID: 26, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			27: {CoreID: 27, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			28: {CoreID: 28, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			29: {CoreID: 29, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			30: {CoreID: 30, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			31: {CoreID: 31, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			32: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			33: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			34: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			35: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			36: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			37: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			38: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			39: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			40: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			41: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			42: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			43: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			44: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			45: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			46: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			47: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 1},
			48: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			49: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			50: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			51: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			52: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			53: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			54: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			55: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 2},
			56: {CoreID: 24, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			57: {CoreID: 25, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			58: {CoreID: 26, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			59: {CoreID: 27, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			60: {CoreID: 28, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			61: {CoreID: 29, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			62: {CoreID: 30, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
			63: {CoreID: 31, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 3},
		},
	}

	AMD_EPYC_7402P_24_Core_Processor = &topology.CPUTopology{
		NumCPUs:      48,
		NumCores:     24,
		NumSockets:   1,
		NumNUMANodes: 1,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			24: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			25: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			26: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			27: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			28: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			29: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			30: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			31: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			32: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			33: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			10: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			34: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			11: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			35: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			12: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			36: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			13: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			37: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			14: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			38: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			15: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			39: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			16: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			40: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			17: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			41: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			18: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			42: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			19: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			43: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			20: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			44: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			21: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			45: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			22: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			46: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			23: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			47: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
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

	Intel_R__Xeon_R__E_2278G_CPU___3_40GHz = &topology.CPUTopology{
		NumCPUs:      8,
		NumCores:     8,
		NumSockets:   1,
		NumNUMANodes: 1,
		CPUDetails: map[int]topology.CPUInfo{
			0: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
		},
	}

	Intel_R__Xeon_R__Silver_4214_CPU___2_20GHz = &topology.CPUTopology{
		NumCPUs:      48,
		NumCores:     24,
		NumSockets:   2,
		NumNUMANodes: 2,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			24: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			25: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			26: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			27: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			28: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			29: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			30: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			31: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			32: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			33: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			10: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			34: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			11: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			35: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			12: {CoreID: 12, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			36: {CoreID: 12, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			13: {CoreID: 13, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			37: {CoreID: 13, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			14: {CoreID: 14, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			38: {CoreID: 14, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			15: {CoreID: 15, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			39: {CoreID: 15, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			16: {CoreID: 16, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			40: {CoreID: 16, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			17: {CoreID: 17, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			41: {CoreID: 17, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			18: {CoreID: 18, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			42: {CoreID: 18, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			19: {CoreID: 19, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			43: {CoreID: 19, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			20: {CoreID: 20, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			44: {CoreID: 20, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			21: {CoreID: 21, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			45: {CoreID: 21, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			22: {CoreID: 22, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			46: {CoreID: 22, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			23: {CoreID: 23, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
			47: {CoreID: 23, SocketID: 1, NUMANodeID: 1, UnCoreCacheID: 0},
		},
	}

	AMD_EPYC_7513_32_Core_Processor = &topology.CPUTopology{
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
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			36: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			37: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			38: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			39: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			40: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			41: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			10: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			42: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			11: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			43: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			12: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			44: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			13: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			45: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			14: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			46: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			15: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			47: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			16: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			48: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			17: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			49: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			18: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			50: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			19: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			51: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			20: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			52: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			21: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			53: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			22: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			54: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			23: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			55: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			24: {CoreID: 24, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			56: {CoreID: 24, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			25: {CoreID: 25, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			57: {CoreID: 25, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			26: {CoreID: 26, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			58: {CoreID: 26, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			27: {CoreID: 27, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			59: {CoreID: 27, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			28: {CoreID: 28, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			60: {CoreID: 28, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			29: {CoreID: 29, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			61: {CoreID: 29, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			30: {CoreID: 30, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			62: {CoreID: 30, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			31: {CoreID: 31, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			63: {CoreID: 31, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
		},
	}

	AMD_EPYC_7443P_24_Core_Processor = &topology.CPUTopology{
		NumCPUs:      48,
		NumCores:     24,
		NumSockets:   1,
		NumNUMANodes: 1,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			24: {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			25: {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			26: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			27: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			28: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			29: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			30: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			31: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			32: {CoreID: 8, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			33: {CoreID: 9, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			10: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			34: {CoreID: 10, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			11: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			35: {CoreID: 11, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			12: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			36: {CoreID: 12, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			13: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			37: {CoreID: 13, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			14: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			38: {CoreID: 14, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			15: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			39: {CoreID: 15, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			16: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			40: {CoreID: 16, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			17: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			41: {CoreID: 17, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			18: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			42: {CoreID: 18, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			19: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			43: {CoreID: 19, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			20: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			44: {CoreID: 20, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			21: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			45: {CoreID: 21, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			22: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			46: {CoreID: 22, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			23: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			47: {CoreID: 23, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
		},
	}

	Intel_R__Xeon_R__E_2378G_CPU___2_80GHz = &topology.CPUTopology{
		NumCPUs:      16,
		NumCores:     8,
		NumSockets:   1,
		NumNUMANodes: 1,
		CPUDetails: map[int]topology.CPUInfo{
			0:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			8:  {CoreID: 0, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			1:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			9:  {CoreID: 1, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			2:  {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			10: {CoreID: 2, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			3:  {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			11: {CoreID: 3, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			4:  {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			12: {CoreID: 4, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			5:  {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			13: {CoreID: 5, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			6:  {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			14: {CoreID: 6, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			7:  {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
			15: {CoreID: 7, SocketID: 0, NUMANodeID: 0, UnCoreCacheID: 0},
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

func TestTakeTopology(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		numCpus     int
		expResult   string
	}{
		{
			"topoDualUncoreCacheSingleSocketHT",
			topoDualUncoreCacheSingleSocketHT,
			3, // 2 is non-deterministic
			"map[1:0-2 2:3-5 3:8-10 4:11-13 5:6-7,14]",
		},
		{
			"topoFROMjfbai",
			topoFROMjfbai,
			3, // 2 is non-deterministic
			"map[1:0-2 2:6-8 3:3-5 4:9-11]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description+fmt.Sprintf("_BY_%d_numCpus", tc.numCpus), func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for len(cpuSet.ToSliceNoSort()) >= tc.numCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, tc.numCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTakeTopologyIterate(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"topoDualUncoreCacheSingleSocketHT",
			topoDualUncoreCacheSingleSocketHT,
			[]int{1, 1, 1, 1, 5, 4},
			"map[1:0 2:1 3:2 4:3 5:8-12 6:4-7]",
		},
		{
			"topoFROMjfbai",
			topoFROMjfbai,
			[]int{1, 4, 3},
			"map[1:0 2:6-9 3:1-3]",
		},
		{
			"topoQuadSocketFourWayHT",
			topoQuadSocketFourWayHT,
			[]int{1, 2, 3},
			"map[1:0 2:109,169 3:50,110,229]",
		},
		{
			"gold_5218_topology",
			gold_5218_topology,
			[]int{1, 2, 3, 4},
			"map[1:0 2:1,4 3:5,8-9 4:12-15]",
		},
		{
			"epyc_7402p_topology",
			epyc_7402p_topology,
			[]int{1, 2, 3, 4},
			"map[1:0 2:1-2 3:3-5 4:6-9]",
		},
		{
			"epyc_7502p_32_topology",
			epyc_7502p_32_topology,
			[]int{2, 2, 2, 2},
			"map[1:0-1 2:2-3 3:8-9 4:10-11]",
		},
		{
			"epyc_7502p_32_topology_TAKE_BY_3",
			epyc_7502p_32_topology,
			[]int{3, 3, 3, 3},
			"map[1:0-2 2:8-10 3:12-14 4:16-18]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTakeEpyc7513(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"epyc_7513",
			epyc_7513,
			[]int{2, 4, 8},
			"map[1:0,32 2:1-2,33-34 3:3-6,35-38]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_AMD_EPYC_7402P_24_Core_Processor(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"AMD_EPYC_7402P_24_Core_Processor",
			AMD_EPYC_7402P_24_Core_Processor,
			[]int{2, 4, 8},
			"map[1:0,24 2:1-2,25-26 3:3-6,27-30]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_AMD_EPYC_7502P_32_Core_Processor(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"AMD_EPYC_7502P_32_Core_Processor",
			AMD_EPYC_7502P_32_Core_Processor,
			[]int{2, 4, 8},
			"map[1:0,32 2:1-2,33-34 3:4-7,36-39]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_Intel_R__Xeon_R__E_2278G_CPU___3_40GHz(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"Intel_R__Xeon_R__E_2278G_CPU___3_40GHz",
			Intel_R__Xeon_R__E_2278G_CPU___3_40GHz,
			[]int{2, 4},
			"map[1:0-1 2:2-5]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_Intel_R__Xeon_R__Silver_4214_CPU___2_20GHz(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"Intel_R__Xeon_R__Silver_4214_CPU___2_20GHz",
			Intel_R__Xeon_R__Silver_4214_CPU___2_20GHz,
			[]int{2, 4, 8},
			"map[1:0,24 2:1-2,25-26 3:3-6,27-30]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_AMD_EPYC_7513_32_Core_Processor(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"AMD_EPYC_7513_32_Core_Processor",
			AMD_EPYC_7513_32_Core_Processor,
			[]int{2, 4, 8},
			"map[1:0,32 2:1-2,33-34 3:3-6,35-38]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_AMD_EPYC_7443P_24_Core_Processor(t *testing.T) {
	testCases := []struct {
		description string
		topo        *topology.CPUTopology
		//numCpus     int
		takeNumCpus []int
		expResult   string
	}{
		{
			"AMD_EPYC_7443P_24_Core_Processor",
			AMD_EPYC_7443P_24_Core_Processor,
			[]int{2, 4, 8},
			"map[1:0,24 2:1-2,25-26 3:3-6,27-30]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			results := make(map[int]cpuset.CPUSet)
			cpuSet := cpusetForCPUTopology(tc.topo)
			counter := 0
			for _, takeCpus := range tc.takeNumCpus {
				took, err := takeByTopologyNUMAPacked(tc.topo, cpuSet, takeCpus)
				if err != nil {
					t.Errorf("[%s] ERROR: %v", tc.description, err.Error())
					break
				}
				counter += 1
				results[counter] = took
				// fmt.Println(counter, took)
				fmt.Println("     TOOK:", took) // FIXME
				cpuSet = cpuSet.Difference(took)
			}
			result := fmt.Sprint(results)
			if result != tc.expResult {
				t.Errorf("[%s] expected %v to equal %v", tc.description, result, tc.expResult)
			}
		})
	}
}

func TestTake_Intel_R__Xeon_R__E_2378G_CPU___2_80GHz(t *testing.T) {
	examples := []struct {
		takeNumCpus []int
		expResult   string
	}{
		{
			[]int{1, 2, 3},
			"map[1:0 2:1,9 3:2,8,10]",
		},
		{
			[]int{2, 4, 8},
			"map[1:0,8 2:1-2,9-10 3:3-6,11-14]",
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, true)()
	for _, tc := range examples {
		result := take_iterator(Intel_R__Xeon_R__E_2378G_CPU___2_80GHz, tc.takeNumCpus, true, t)
		if result != tc.expResult {
			t.Errorf("\nEXPECTED: %v\nTO EQUAL: %v", result, tc.expResult)
			return
		}
	}
}

func take_iterator(topo *topology.CPUTopology, takeNumCpus []int, featureFlag bool, t *testing.T) string {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CPUManagerUncoreCacheAlign, featureFlag)()
	results := make(map[int]cpuset.CPUSet)
	cpuSet := cpusetForCPUTopology(topo)
	counter := 0
	for _, takeCpus := range takeNumCpus {
		took, err := takeByTopologyNUMAPacked(topo, cpuSet, takeCpus)
		if err != nil {
			return fmt.Sprint(err)
		}
		counter += 1
		results[counter] = took
		fmt.Println("     TOOK:", took) // FIXME
		cpuSet = cpuSet.Difference(took)
	}
	result := fmt.Sprint(results)
	return result
}

func TestTake_Intel_R__Xeon_R__Gold_5120_CPU___2_20GHz(t *testing.T) {
	examples := []struct {
		takeNumCpus []int
		expOld      string
		expNew      string
	}{
		{
			[]int{1, 2, 3},
			"map[1:0 2:2,30 3:4,28,32]",
			"map[1:0 2:2,30 3:4,28,32]",
		},
		{
			[]int{2, 4, 8, 15},
			"map[1:0,28 2:2,4,30,32 3:6,8,10,12,34,36,38,40 4:1,14,16,18,20,22,24,26,42,44,46,48,50,52,54]",
			"map[1:0,28 2:2,4,30,32 3:6,8,10,12,34,36,38,40 4:1,3,5,7,9,11,13,15,29,31,33,35,37,39,41]",
		},
	}

	for _, tc := range examples {
		oldResult, newResult := take_iterator_old_new(Intel_R__Xeon_R__Gold_5120_CPU___2_20GHz, tc.takeNumCpus, t)
		if oldResult != tc.expOld {
			t.Errorf("\nEXP__OLD: %v\nTO EQUAL: %v", tc.expOld, oldResult)
			return
		}
		if newResult != tc.expNew {
			t.Errorf("\nEXP__NEW: %v\nTO EQUAL: %v", tc.expNew, newResult)
			return
		}
	}
}

func take_iterator_old_new(topo *topology.CPUTopology, takeNumCpus []int, t *testing.T) (string, string) {
	oldResult := take_iterator(topo, takeNumCpus, false, t)
	newResult := take_iterator(topo, takeNumCpus, true, t)
	return oldResult, newResult
}
