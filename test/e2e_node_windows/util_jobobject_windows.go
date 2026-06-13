//go:build windows

/*
Copyright The Kubernetes Authors.

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

package e2enodewindows

import (
	"fmt"
	"math/bits"
	"unsafe"

	"golang.org/x/sys/windows"

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

// Host-side job-object validation reads the Win32 kernel state directly,
// complementing the CRI ContainerStatus read-back used elsewhere in this
// suite. The CRI check verifies "containerd remembers what the kubelet
// asked for"; this check verifies "the kernel actually applied it".
//
// Scope: WCOW process-isolated containers only. Naming convention
// "\Container_<full-id>" is containerd/hcsshim internal and used here
// only as a diagnostic source of truth, mirroring what hcsshim's own
// test suite does.

const jobObjectQueryAccess = 0x0004 // JOB_OBJECT_QUERY

var (
	modntdll            = windows.NewLazySystemDLL("ntdll.dll")
	procNtOpenJobObject = modntdll.NewProc("NtOpenJobObject")
)

// groupAffinity mirrors Win32 GROUP_AFFINITY (winnt.h). KAFFINITY is
// ULONG_PTR which is uintptr on 64-bit Go.
type groupAffinity struct {
	Mask     uintptr
	Group    uint16
	Reserved [3]uint16
}

// openJobObject opens the container silo job object by its object-manager
// name (e.g. "\Container_<id>"). It uses the native NtOpenJobObject rather
// than the Win32 OpenJobObjectW: the silo lives at the root of the NT object
// namespace, and OpenJobObjectW only resolves names under BaseNamedObjects
// and rejects names containing a backslash (it would return
// ERROR_BAD_PATHNAME "the specified path is invalid"). Returns an error whose
// NTStatus is windows.STATUS_OBJECT_NAME_NOT_FOUND when no such job object
// exists.
func openJobObject(name string) (windows.Handle, error) {
	uname, err := windows.NewNTUnicodeString(name)
	if err != nil {
		return 0, err
	}
	oa := windows.OBJECT_ATTRIBUTES{
		ObjectName: uname,
		Attributes: windows.OBJ_CASE_INSENSITIVE,
	}
	oa.Length = uint32(unsafe.Sizeof(oa))

	var h windows.Handle
	// NTSTATUS NtOpenJobObject(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES)
	r, _, _ := procNtOpenJobObject.Call(
		uintptr(unsafe.Pointer(&h)),
		uintptr(jobObjectQueryAccess),
		uintptr(unsafe.Pointer(&oa)),
	)
	if status := windows.NTStatus(r); status != windows.STATUS_SUCCESS {
		return 0, fmt.Errorf("NtOpenJobObject(%s): %w", name, status)
	}
	return h, nil
}

// getHostJobAffinity opens the containerd-created job object for a
// process-isolated container and reads its JobObjectGroupInformationEx
// (the GROUP_AFFINITY array currently applied by the Windows kernel).
func getHostJobAffinity(pod *v1.Pod, ctnName string) ([]groupAffinity, error) {
	cntID, err := containerIDForContainer(pod, ctnName)
	if err != nil {
		return nil, err
	}

	// The silo job object is only readable by SYSTEM (see util_system_windows.go),
	// so open and query it while impersonating SYSTEM. The handle's access is
	// checked at open time, so reading under the same impersonated context is
	// what matters here.
	var out []groupAffinity
	err = runAsSystem(func() error {
		h, err := openJobObject(`\Container_` + cntID)
		if err != nil {
			return err
		}
		defer windows.CloseHandle(h)

		// JobObjectGroupInformationEx returns a flat array of GROUP_AFFINITY.
		// 64 entries covers Windows' architectural limit on processor groups,
		// far more than any real system has.
		const maxGroups = 64
		entrySize := int(unsafe.Sizeof(groupAffinity{}))
		buf := make([]byte, maxGroups*entrySize)

		var returned uint32
		if err := windows.QueryInformationJobObject(
			h, windows.JobObjectGroupInformationEx,
			uintptr(unsafe.Pointer(&buf[0])), uint32(len(buf)), &returned,
		); err != nil {
			return fmt.Errorf("QueryInformationJobObject(JobObjectGroupInformationEx): %w", err)
		}

		count := int(returned) / entrySize
		out = make([]groupAffinity, count)
		for i := 0; i < count; i++ {
			out[i] = *(*groupAffinity)(unsafe.Pointer(&buf[i*entrySize]))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// countCPUsInHostJobAffinity sums set bits across all GROUP_AFFINITY
// entries returned by getHostJobAffinity.
func countCPUsInHostJobAffinity(affs []groupAffinity) int {
	total := 0
	for _, a := range affs {
		total += bits.OnesCount64(uint64(a.Mask))
	}
	return total
}

// hostJobAffinitiesOverlap reports whether two host job-object affinities share
// at least one CPU (same bit set in the same processor group). It is the
// host-side counterpart to windowsAffinitiesOverlap (which compares CRI masks).
func hostJobAffinitiesOverlap(a, b []groupAffinity) bool {
	maskA := map[uint16]uint64{}
	for _, x := range a {
		maskA[x.Group] |= uint64(x.Mask)
	}
	for _, x := range b {
		if maskA[x.Group]&uint64(x.Mask) != 0 {
			return true
		}
	}
	return false
}

// hostJobAffinityMatchesCRI returns nil iff the host job-object affinity
// matches the CRI-reported affinity bit-for-bit in every processor group.
// Bit-for-bit agreement proves both that the runtime applied the
// kubelet's request and that CRI's read-back is faithful.
func hostJobAffinityMatchesCRI(host []groupAffinity, cri []*runtimeapi.WindowsCpuGroupAffinity) error {
	hostByGroup := map[uint16]uint64{}
	for _, a := range host {
		hostByGroup[a.Group] |= uint64(a.Mask)
	}
	criByGroup := map[uint16]uint64{}
	for _, a := range cri {
		criByGroup[uint16(a.CpuGroup)] |= a.CpuMask
	}
	for g, hMask := range hostByGroup {
		if criByGroup[g] != hMask {
			return fmt.Errorf("group %d: host mask=0x%x, CRI mask=0x%x", g, hMask, criByGroup[g])
		}
	}
	for g, cMask := range criByGroup {
		if _, ok := hostByGroup[g]; !ok && cMask != 0 {
			return fmt.Errorf("group %d: CRI mask=0x%x, host has no affinity in this group", g, cMask)
		}
	}
	return nil
}

// validateHostJobAffinityProcessIsolated cross-checks the host job-object
// affinity against the CRI-reported affinity for a process-isolated
// container.
func validateHostJobAffinityProcessIsolated(pod *v1.Pod, ctnName string, cri []*runtimeapi.WindowsCpuGroupAffinity) error {
	host, err := getHostJobAffinity(pod, ctnName)
	if err != nil {
		return err
	}
	if got, want := countCPUsInHostJobAffinity(host), countCPUsInAffinities(cri); got != want {
		return fmt.Errorf("CPU count disagreement: host job=%d, CRI=%d", got, want)
	}
	return hostJobAffinityMatchesCRI(host, cri)
}
