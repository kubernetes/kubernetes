package cpumanager

import (
	"fmt"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

func (a *cpuAccumulator) getUnCoreCacheID(cpuid int) int {
	uccID := a.topo.CPUDetails[cpuid].UnCoreCacheID
	return uccID
}

func (a *cpuAccumulator) tryTakeAlignedUncoreCacheCPUs(numCpus int) {
	if numCpus <= 1 {
		fmt.Println("UCC IGNORE SINGLE NUM CPUS") // FIXME
		return                                    // zero?? or one cpu is already aligned, let the next cpu manager allocate it better
	}
	if !isUncoreCacheAlignEnabled() {
		return // TODO remove this when feature gate is default true... or not
	}
	ua := newUncoreAccumulator(a)
	fmt.Println("")                                   // FIXME
	fmt.Println("numCpus: ", numCpus)                 // FIXME
	fmt.Println("availcpus:", ua.AvailableCpuIdArray) // FIXME
	fmt.Println("ucc2count:", ua.Ucc2Count)           // FIXME
	if ua.numberAvailableCaches() <= 1 {
		// all remaining free cpus have same uncore cache, let the next cpu manager allocate it better
		fmt.Println("UCC IGNORE SINGLE CACHES") // FIXME
		return
	}
	uccPick := ua.pickCache()
	if uccPick == -1 {
		// no available ucc big enough for numCpus
		fmt.Println("UCC NO CACHE BIG ENOUGH") // FIXME
		return
	}
	ua.takeCpusFromCache(numCpus, uccPick)
}

func (a *cpuAccumulator) tryMaskUncoreCacheCPUs() {
	if !isUncoreCacheAlignEnabled() {
		return // TODO remove this when feature gate is default true... or not
	}
	if a.numCPUsNeeded <= 1 {
		fmt.Println("UCC IGNORE SINGLE NUM CPUS") // FIXME
		return                                    // zero?? or one cpu is already aligned, let the next cpu manager allocate it better
	}
	ua := newUncoreAccumulator(a)
	fmt.Println("")                                   // FIXME
	fmt.Println("numCpus: ", a.numCPUsNeeded)         // FIXME
	fmt.Println("availcpus:", ua.AvailableCpuIdArray) // FIXME
	fmt.Println("ucc2count:", ua.Ucc2Count)           // FIXME
	if ua.numberAvailableCaches() <= 1 {
		// all remaining free cpus have same uncore cache already
		fmt.Println("UCC SINGLE CACHE ALREADY") // FIXME
		return
	}
	uccPick := ua.pickCache()
	if uccPick == -1 {
		// no available ucc big enough for numCpus
		fmt.Println("UCC NO CACHE BIG ENOUGH") // FIXME
		return
	}
	ua.maskUncoreCacheCpus(uccPick)
}

func isUncoreCacheAlignEnabled() bool {
	if feature.DefaultFeatureGate.Enabled(features.CPUManagerUncoreCacheAlign) {
		return true
	}
	return false
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

func (ua *UncoreAccumulator) takeCpusFromCache(numCpus int, uccPick int) {
	for _, cpu := range ua.AvailableCpuIdArray {
		if ua.CpuAccumulator.getUnCoreCacheID(cpu) != uccPick {
			continue // only taking cpus in uccPick
		}
		ua.CpuAccumulator.take(cpuset.NewCPUSet(cpu))
		numCpus -= 1
		if numCpus == 0 {
			return // SUCCESS
		}
	}
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

func (a *cpuAccumulator) OLDtryTakeAlignedUncoreCacheCPUs(numCpus int) {
	// must find a single uncore cache for all numCpus, or nothing!
	if !isUncoreCacheAlignEnabled() {
		return // TODO remove this when feature gate is default true... or not
	}
	if numCpus <= 1 {
		return // zero?? or one cpu is already aligned, let the next cpu manager allocate it better
	}
	// The other cpu managers use this to make decisions about allocations
	// If we need to change the sorting algorithm, we should do it locally here (not everywhere)
	cpus := a.sortAvailableCPUs()
	fmt.Println("cpus:", cpus) // FIXME

	// Count how many available cpus are in each uncore cache
	ucc2count := make(map[int]int)
	for _, cpu := range cpus {
		ucc := a.getUnCoreCacheID(cpu)
		// use this to get the len()
		ucc2count[ucc] += 1
	}
	fmt.Println("ucc2count:", ucc2count) // FIXME
	if len(ucc2count) <= 1 {
		return // all remaining free cpus have same uncore cache, let the next cpu manager allocate it better
	}

	// Deterministically choose ucc to pick from, if possible
	uccPick := -1
	countMax := -1
	for _, cpu := range cpus {
		ucc := a.getUnCoreCacheID(cpu)
		count := ucc2count[ucc]
		if count < numCpus {
			continue // not enough cpus in this ucc for request
		}
		if count == numCpus {
			uccPick = ucc
			break // found perfect fit, this is our pick
		}
		// count > numCpus
		if countMax == -1 || count > countMax {
			// if no perfect fit found, we will pick this ucc
			uccPick = ucc
			countMax = count
		}
	}

	if uccPick == -1 {
		return // there is no available ucc big enough for numCpus
	}

	// take cpus from this ucc
	for _, cpu := range cpus {
		if a.getUnCoreCacheID(cpu) != uccPick {
			continue // only taking cpus in uccPick
		}
		a.take(cpuset.NewCPUSet(cpu))
		numCpus -= 1
		if numCpus == 0 {
			if !a.isSatisfied() {
				klog.Errorf("NOT SATISFIED!!!")
			}
			return // SUCCESS
		}
	}
	klog.Errorf("THIS SHOULD NEVER HAPPEN!!!")
}

// EVERYTHING BELOW THIS LINE IS POTENTIALLY FOSSIL CODE

func (a *cpuAccumulator) PROPOSED_sortAvailableCores() []int {
	if isUncoreCacheAlignEnabled() {
		var result []int
		for _, cache := range a.sortAvailableUncoreCaches() {
			cores := a.details.CoresInUncoreCaches(cache).ToSliceNoSort()
			a.sort(cores, a.details.CPUsInCores)
			result = append(result, cores...)
		}
		return result
	}
	return a.numaOrSocketsFirst.sortAvailableCores()
}

func (a *cpuAccumulator) tryTakeFulldUncoreCacheCPUs() {
	// must find a single uncore cache for all numCpus, or nothing!
	if !isUncoreCacheAlignEnabled() {
		return // TODO remove this when feature gate is default true... or not
	}
	for _, uncorecache := range a.freeUncoreCaches() {
		cpusInUncoreCache := a.topo.CPUDetails.CPUsInUncoreCaches(uncorecache)
		if !a.needs(cpusInUncoreCache.Size()) {
			continue
		}
		klog.V(3).InfoS("takeFullUncoreCaches: claiming uncore-cache", "uncore-cache", uncorecache)
		a.take(cpusInUncoreCache)
	}
}

// Returns free uncore cache IDs as a slice sorted by sortAvailableUncoreCaches().
// Only support when CpuManagerUncoreCacheAlign is enabled.
func (a *cpuAccumulator) freeUncoreCaches() []int {
	free := []int{}
	for _, cache := range a.sortAvailableUncoreCaches() {
		if a.isUncoreCacheFree(cache) {
			free = append(free, cache)
		}
	}
	return free
}

// Returns true if the supplied core is fully available in `topoDetails`.
func (a *cpuAccumulator) isUncoreCacheFree(uncoreCacheID int) bool {
	return a.details.CPUsInUncoreCaches(uncoreCacheID).Size() == a.CPUsPerUncoreCache()
}

// CPUsPerUncoreCache returns the average number of logical CPUs are associated with
// each uncore cache id. Even CPUs share the same llc id may not the same.
func (a *cpuAccumulator) CPUsPerUncoreCache() int {
	ucc2count := make(map[int]int)
	for _, cpu := range a.details {
		ucc := cpu.UnCoreCacheID
		// use this to get the len()
		ucc2count[ucc] += 1
	}
	if len(ucc2count) <= 1 {
		return 0
	}
	return len(a.details) / len(ucc2count)

	// WUZ
	//if a.NumUnCoreCaches == 0 {
	//	return 0
	//}
	//return topo.NumCPUs / topo.NumUnCoreCaches
}

// FIXME COMMENT BY jfbai: In some special cases, it may give wrong answer. In VM for example, host machine may reserve cores to for virtualization and these cores are reserved evenly on multiple chips, in these case, kubelet in VM gets all the caches same as the host machine, but the threads are less than host machine, then CPUsPerUncoreCache will be wrong.
// FIXME REPLY BY ranchothu: thanks, good question, the key is not evenly allocate of physical cpu to vcpu. this depend too much on the virtualization implementation of cloud vendors, and in some caces(like yours) the problem occurs.
// FIXME     Although in our cloud the problem is not exists, but, i will take it to consideration and make a change, to make the algorithm more common.

// Sort all sockets with free CPUs using the sort() algorithm defined above.
func (a *cpuAccumulator) sortAvailableUncoreCaches() []int {
	var result []int
	for _, socket := range a.sortAvailableSockets() {
		caches := a.details.UncoreCachesInSocket(socket).ToSliceNoSort()
		a.sort(caches, a.details.CPUsInUncoreCaches)
		result = append(result, caches...)
	}
	return result
}
