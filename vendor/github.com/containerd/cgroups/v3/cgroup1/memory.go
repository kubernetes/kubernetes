/*
   Copyright The containerd Authors.

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

package cgroup1

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	v1 "github.com/containerd/cgroups/v3/cgroup1/stats"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"golang.org/x/sys/unix"
)

// MemoryEvent is an interface that V1 memory Cgroup notifications implement. Arg returns the
// file name whose fd should be written to "cgroups.event_control". EventFile returns the name of
// the file that supports the notification api e.g. "memory.usage_in_bytes".
type MemoryEvent interface {
	Arg() string
	EventFile() string
}

type memoryThresholdEvent struct {
	threshold uint64
	swap      bool
}

// MemoryThresholdEvent returns a new [MemoryEvent] representing the memory threshold set.
// If swap is true, the event will be registered using memory.memsw.usage_in_bytes
func MemoryThresholdEvent(threshold uint64, swap bool) MemoryEvent {
	return &memoryThresholdEvent{
		threshold,
		swap,
	}
}

func (m *memoryThresholdEvent) Arg() string {
	return strconv.FormatUint(m.threshold, 10)
}

func (m *memoryThresholdEvent) EventFile() string {
	if m.swap {
		return "memory.memsw.usage_in_bytes"
	}
	return "memory.usage_in_bytes"
}

type oomEvent struct{}

// OOMEvent returns a new oom event to be used with RegisterMemoryEvent.
func OOMEvent() MemoryEvent {
	return &oomEvent{}
}

func (oom *oomEvent) Arg() string {
	return ""
}

func (oom *oomEvent) EventFile() string {
	return "memory.oom_control"
}

type memoryPressureEvent struct {
	pressureLevel MemoryPressureLevel
	hierarchy     EventNotificationMode
}

// MemoryPressureEvent returns a new [MemoryEvent] representing the memory pressure set.
func MemoryPressureEvent(pressureLevel MemoryPressureLevel, hierarchy EventNotificationMode) MemoryEvent {
	return &memoryPressureEvent{
		pressureLevel,
		hierarchy,
	}
}

func (m *memoryPressureEvent) Arg() string {
	return string(m.pressureLevel) + "," + string(m.hierarchy)
}

func (m *memoryPressureEvent) EventFile() string {
	return "memory.pressure_level"
}

// MemoryPressureLevel corresponds to the memory pressure levels defined
// for memory cgroups.
type MemoryPressureLevel string

// The three memory pressure levels are as follows.
//   - The "low" level means that the system is reclaiming memory for new
//     allocations. Monitoring this reclaiming activity might be useful for
//     maintaining cache level. Upon notification, the program (typically
//     "Activity Manager") might analyze vmstat and act in advance (i.e.
//     prematurely shutdown unimportant services).
//   - The "medium" level means that the system is experiencing medium memory
//     pressure, the system might be making swap, paging out active file caches,
//     etc. Upon this event applications may decide to further analyze
//     vmstat/zoneinfo/memcg or internal memory usage statistics and free any
//     resources that can be easily reconstructed or re-read from a disk.
//   - The "critical" level means that the system is actively thrashing, it is
//     about to out of memory (OOM) or even the in-kernel OOM killer is on its
//     way to trigger. Applications should do whatever they can to help the
//     system. It might be too late to consult with vmstat or any other
//     statistics, so it is advisable to take an immediate action.
//     "https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt" Section 11
const (
	LowPressure      MemoryPressureLevel = "low"
	MediumPressure   MemoryPressureLevel = "medium"
	CriticalPressure MemoryPressureLevel = "critical"
)

// EventNotificationMode corresponds to the notification modes
// for the memory cgroups pressure level notifications.
type EventNotificationMode string

// There are three optional modes that specify different propagation behavior:
//   - "default": this is the default behavior specified above. This mode is the
//     same as omitting the optional mode parameter, preserved by backwards
//     compatibility.
//   - "hierarchy": events always propagate up to the root, similar to the default
//     behavior, except that propagation continues regardless of whether there are
//     event listeners at each level, with the "hierarchy" mode. In the above
//     example, groups A, B, and C will receive notification of memory pressure.
//   - "local": events are pass-through, i.e. they only receive notifications when
//     memory pressure is experienced in the memcg for which the notification is
//     registered. In the above example, group C will receive notification if
//     registered for "local" notification and the group experiences memory
//     pressure. However, group B will never receive notification, regardless if
//     there is an event listener for group C or not, if group B is registered for
//     local notification.
//     "https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt" Section 11
const (
	DefaultMode   EventNotificationMode = "default"
	LocalMode     EventNotificationMode = "local"
	HierarchyMode EventNotificationMode = "hierarchy"
)

// NewMemory returns a Memory controller given the root folder of cgroups.
// It may optionally accept other configuration options, such as IgnoreModules(...)
func NewMemory(root string, options ...func(*memoryController)) *memoryController {
	mc := &memoryController{
		root:    filepath.Join(root, string(Memory)),
		ignored: map[string]struct{}{},
	}
	for _, opt := range options {
		opt(mc)
	}
	return mc
}

// IgnoreModules configure the memory controller to not read memory metrics for some
// module names (e.g. passing "memsw" would avoid all the memory.memsw.* entries)
func IgnoreModules(names ...string) func(*memoryController) {
	return func(mc *memoryController) {
		for _, name := range names {
			mc.ignored[name] = struct{}{}
		}
	}
}

// OptionalSwap allows the memory controller to not fail if cgroups is not accounting
// Swap memory (there are no memory.memsw.* entries)
func OptionalSwap() func(*memoryController) {
	return func(mc *memoryController) {
		_, err := os.Stat(filepath.Join(mc.root, "memory.memsw.usage_in_bytes"))
		if os.IsNotExist(err) {
			mc.ignored["memsw"] = struct{}{}
		}
	}
}

type memoryController struct {
	root    string
	ignored map[string]struct{}
}

func (m *memoryController) Name() Name {
	return Memory
}

func (m *memoryController) Path(path string) string {
	return filepath.Join(m.root, path)
}

func (m *memoryController) Create(path string, resources *specs.LinuxResources) error {
	if err := os.MkdirAll(m.Path(path), defaultDirPerm); err != nil {
		return err
	}
	if resources.Memory == nil {
		return nil
	}
	return m.set(path, getMemorySettings(resources))
}

func (m *memoryController) Update(path string, resources *specs.LinuxResources) error {
	if resources.Memory == nil {
		return nil
	}
	g := func(v *int64) bool {
		return v != nil && *v > 0
	}
	settings := getMemorySettings(resources)
	if g(resources.Memory.Limit) && g(resources.Memory.Swap) {
		// if the updated swap value is larger than the current memory limit set the swap changes first
		// then set the memory limit as swap must always be larger than the current limit
		current, err := readUint(filepath.Join(m.Path(path), "memory.limit_in_bytes"))
		if err != nil {
			return err
		}
		if current < uint64(*resources.Memory.Swap) {
			settings[0], settings[1] = settings[1], settings[0]
		}
	}
	return m.set(path, settings)
}

func (m *memoryController) Stat(path string, stats *v1.Metrics) error {
	fMemStat, err := os.Open(filepath.Join(m.Path(path), "memory.stat"))
	if err != nil {
		return err
	}
	defer fMemStat.Close()
	stats.Memory = &v1.MemoryStat{
		Usage:     &v1.MemoryEntry{},
		Swap:      &v1.MemoryEntry{},
		Kernel:    &v1.MemoryEntry{},
		KernelTCP: &v1.MemoryEntry{},
	}
	if err := m.parseStats(fMemStat, stats.Memory); err != nil {
		return err
	}

	fMemOomControl, err := os.Open(filepath.Join(m.Path(path), "memory.oom_control"))
	if err != nil {
		return err
	}
	defer fMemOomControl.Close()
	stats.MemoryOomControl = &v1.MemoryOomControl{}
	if err := m.parseOomControlStats(fMemOomControl, stats.MemoryOomControl); err != nil {
		return err
	}
	for _, t := range []struct {
		module string
		entry  *v1.MemoryEntry
	}{
		{
			module: "",
			entry:  stats.Memory.Usage,
		},
		{
			module: "memsw",
			entry:  stats.Memory.Swap,
		},
		{
			module: "kmem",
			entry:  stats.Memory.Kernel,
		},
		{
			module: "kmem.tcp",
			entry:  stats.Memory.KernelTCP,
		},
	} {
		if _, ok := m.ignored[t.module]; ok {
			continue
		}
		for _, tt := range []struct {
			name  string
			value *uint64
		}{
			{
				name:  "usage_in_bytes",
				value: &t.entry.Usage,
			},
			{
				name:  "max_usage_in_bytes",
				value: &t.entry.Max,
			},
			{
				name:  "failcnt",
				value: &t.entry.Failcnt,
			},
			{
				name:  "limit_in_bytes",
				value: &t.entry.Limit,
			},
		} {
			parts := []string{"memory"}
			if t.module != "" {
				parts = append(parts, t.module)
			}
			parts = append(parts, tt.name)
			v, err := readUint(filepath.Join(m.Path(path), strings.Join(parts, ".")))
			if err != nil {
				return err
			}
			*tt.value = v
		}
	}
	return nil
}

func (m *memoryController) parseStats(r io.Reader, stat *v1.MemoryStat) error {
	var (
		raw  = make(map[string]uint64)
		sc   = bufio.NewScanner(r)
		line int
	)
	for sc.Scan() {
		key, v, err := parseKV(sc.Text())
		if err != nil {
			return fmt.Errorf("%d: %v", line, err)
		}
		raw[key] = v
		line++
	}
	if err := sc.Err(); err != nil {
		return err
	}
	stat.Cache = raw["cache"]
	stat.RSS = raw["rss"]
	stat.RSSHuge = raw["rss_huge"]
	stat.MappedFile = raw["mapped_file"]
	stat.Dirty = raw["dirty"]
	stat.Writeback = raw["writeback"]
	stat.PgPgIn = raw["pgpgin"]
	stat.PgPgOut = raw["pgpgout"]
	stat.PgFault = raw["pgfault"]
	stat.PgMajFault = raw["pgmajfault"]
	stat.InactiveAnon = raw["inactive_anon"]
	stat.ActiveAnon = raw["active_anon"]
	stat.InactiveFile = raw["inactive_file"]
	stat.ActiveFile = raw["active_file"]
	stat.Unevictable = raw["unevictable"]
	stat.HierarchicalMemoryLimit = raw["hierarchical_memory_limit"]
	stat.HierarchicalSwapLimit = raw["hierarchical_memsw_limit"]
	stat.TotalCache = raw["total_cache"]
	stat.TotalRSS = raw["total_rss"]
	stat.TotalRSSHuge = raw["total_rss_huge"]
	stat.TotalMappedFile = raw["total_mapped_file"]
	stat.TotalDirty = raw["total_dirty"]
	stat.TotalWriteback = raw["total_writeback"]
	stat.TotalPgPgIn = raw["total_pgpgin"]
	stat.TotalPgPgOut = raw["total_pgpgout"]
	stat.TotalPgFault = raw["total_pgfault"]
	stat.TotalPgMajFault = raw["total_pgmajfault"]
	stat.TotalInactiveAnon = raw["total_inactive_anon"]
	stat.TotalActiveAnon = raw["total_active_anon"]
	stat.TotalInactiveFile = raw["total_inactive_file"]
	stat.TotalActiveFile = raw["total_active_file"]
	stat.TotalUnevictable = raw["total_unevictable"]
	return nil
}

func (m *memoryController) parseOomControlStats(r io.Reader, stat *v1.MemoryOomControl) error {
	var (
		raw  = make(map[string]uint64)
		sc   = bufio.NewScanner(r)
		line int
	)
	for sc.Scan() {
		key, v, err := parseKV(sc.Text())
		if err != nil {
			return fmt.Errorf("%d: %v", line, err)
		}
		raw[key] = v
		line++
	}
	if err := sc.Err(); err != nil {
		return err
	}
	stat.OomKillDisable = raw["oom_kill_disable"]
	stat.UnderOom = raw["under_oom"]
	stat.OomKill = raw["oom_kill"]
	return nil
}

func (m *memoryController) set(path string, settings []memorySettings) error {
	for _, t := range settings {
		if t.value != nil {
			if err := os.WriteFile(
				filepath.Join(m.Path(path), "memory."+t.name),
				[]byte(strconv.FormatInt(*t.value, 10)),
				defaultFilePerm,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

type memorySettings struct {
	name  string
	value *int64
}

func getMemorySettings(resources *specs.LinuxResources) []memorySettings {
	mem := resources.Memory
	var swappiness *int64
	if mem.Swappiness != nil {
		v := int64(*mem.Swappiness)
		swappiness = &v
	}
	return []memorySettings{
		{
			name:  "limit_in_bytes",
			value: mem.Limit,
		},
		{
			name:  "soft_limit_in_bytes",
			value: mem.Reservation,
		},
		{
			name:  "memsw.limit_in_bytes",
			value: mem.Swap,
		},
		{
			name:  "kmem.limit_in_bytes",
			value: mem.Kernel,
		},
		{
			name:  "kmem.tcp.limit_in_bytes",
			value: mem.KernelTCP,
		},
		{
			name:  "oom_control",
			value: getOomControlValue(mem),
		},
		{
			name:  "swappiness",
			value: swappiness,
		},
	}
}

func getOomControlValue(mem *specs.LinuxMemory) *int64 {
	if mem.DisableOOMKiller != nil && *mem.DisableOOMKiller {
		i := int64(1)
		return &i
	}
	return nil
}

func (m *memoryController) memoryEvent(path string, event MemoryEvent) (uintptr, error) {
	root := m.Path(path)
	efd, err := unix.Eventfd(0, unix.EFD_CLOEXEC)
	if err != nil {
		return 0, err
	}
	evtFile, err := os.Open(filepath.Join(root, event.EventFile()))
	if err != nil {
		unix.Close(efd)
		return 0, err
	}
	defer evtFile.Close()
	data := fmt.Sprintf("%d %d %s", efd, evtFile.Fd(), event.Arg())
	evctlPath := filepath.Join(root, "cgroup.event_control")
	if err := os.WriteFile(evctlPath, []byte(data), 0o700); err != nil {
		unix.Close(efd)
		return 0, err
	}
	return uintptr(efd), nil
}
