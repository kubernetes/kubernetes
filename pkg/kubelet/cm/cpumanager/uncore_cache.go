package cpumanager

import (
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

func (a *cpuAccumulator) getUnCoreCacheID(cpuid int) int {
	uccID := a.topo.CPUDetails[cpuid].UnCoreCacheID
	return uccID
}

func (a *cpuAccumulator) tryMaskUncoreCacheCPUs() {
	// Examine the available cpus in accumulator and choose best fitting L3 cache (if possible)
	if !isUncoreCacheAlignEnabled() {
		return // TODO remove this when feature gate is default true... or not
	}
	if a.numCPUsNeeded <= 1 {
		// zero, or one cpu is already uncore cache aligned let it be
		return
	}
	ua := newUncoreAccumulator(a)
	if ua.numberAvailableCaches() <= 1 {
		// all remaining free cpus have same uncore cache already
		return
	}
	uccPick := ua.pickCache()
	if uccPick == -1 {
		// no available ucc big enough for numCpus
		return
	}
	ua.maskUncoreCacheCpus(uccPick)
}

func isUncoreCacheAlignEnabled() bool {
	return feature.DefaultFeatureGate.Enabled(features.CPUManagerUncoreCacheAlign)
}

type UncoreAccumulator struct {
	CpuAccumulator      *cpuAccumulator
	AvailableCpuIdArray []int
	Ucc2Count           map[int]int
}

func newUncoreAccumulator(a *cpuAccumulator) *UncoreAccumulator {
	// Count how many available cpus are in each uncore cache
	cpus := a.sortAvailableCPUs()
	ucc2count := make(map[int]int)
	for _, cpu := range cpus {
		ucc := a.getUnCoreCacheID(cpu)
		// use this to get the len()
		ucc2count[ucc] += 1
	}
	return &UncoreAccumulator{
		CpuAccumulator:      a,
		AvailableCpuIdArray: cpus,
		Ucc2Count:           ucc2count,
	}
}

func (ua *UncoreAccumulator) numberAvailableCaches() int {
	return len(ua.Ucc2Count)
}

func (ua *UncoreAccumulator) pickCache() int {
	// Deterministically choose ucc to pick from, if possible
	uccPick := -1
	countMin := -1
	numCpus := ua.CpuAccumulator.numCPUsNeeded
	for _, cpu := range ua.AvailableCpuIdArray {
		ucc := ua.CpuAccumulator.getUnCoreCacheID(cpu)
		count := ua.Ucc2Count[ucc]
		if count < numCpus {
			continue // not enough cpus in this ucc for request
		}
		if count == numCpus {
			uccPick = ucc
			break // found perfect fit, this is our pick
		}
		// count > numCpus
		if countMin == -1 || count < countMin {
			// if no perfect fit found, we will pick this ucc
			uccPick = ucc
			countMin = count
		}
	}
	return uccPick
}

func (ua *UncoreAccumulator) maskUncoreCacheCpus(uccPick int) {
	builder := cpuset.NewBuilder()
	for _, cpu := range ua.AvailableCpuIdArray {
		if ua.CpuAccumulator.getUnCoreCacheID(cpu) != uccPick {
			continue // only taking cpus in uccPick
		}
		builder.Add(cpu)
	}
	// change the original accumulator details to only include cpus from ucc pick
	ua.CpuAccumulator.details = ua.CpuAccumulator.topo.CPUDetails.KeepOnly(builder.Result())
	// Now, the next cpu managers can only take from this uncore cache
}
