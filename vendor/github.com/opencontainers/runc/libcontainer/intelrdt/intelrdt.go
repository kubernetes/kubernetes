// +build linux

package intelrdt

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/moby/sys/mountinfo"
	"github.com/opencontainers/runc/libcontainer/configs"
)

/*
 * About Intel RDT features:
 * Intel platforms with new Xeon CPU support Resource Director Technology (RDT).
 * Cache Allocation Technology (CAT) and Memory Bandwidth Allocation (MBA) are
 * two sub-features of RDT.
 *
 * Cache Allocation Technology (CAT) provides a way for the software to restrict
 * cache allocation to a defined 'subset' of L3 cache which may be overlapping
 * with other 'subsets'. The different subsets are identified by class of
 * service (CLOS) and each CLOS has a capacity bitmask (CBM).
 *
 * Memory Bandwidth Allocation (MBA) provides indirect and approximate throttle
 * over memory bandwidth for the software. A user controls the resource by
 * indicating the percentage of maximum memory bandwidth or memory bandwidth
 * limit in MBps unit if MBA Software Controller is enabled.
 *
 * More details about Intel RDT CAT and MBA can be found in the section 17.18
 * of Intel Software Developer Manual:
 * https://software.intel.com/en-us/articles/intel-sdm
 *
 * About Intel RDT kernel interface:
 * In Linux 4.10 kernel or newer, the interface is defined and exposed via
 * "resource control" filesystem, which is a "cgroup-like" interface.
 *
 * Comparing with cgroups, it has similar process management lifecycle and
 * interfaces in a container. But unlike cgroups' hierarchy, it has single level
 * filesystem layout.
 *
 * CAT and MBA features are introduced in Linux 4.10 and 4.12 kernel via
 * "resource control" filesystem.
 *
 * Intel RDT "resource control" filesystem hierarchy:
 * mount -t resctrl resctrl /sys/fs/resctrl
 * tree /sys/fs/resctrl
 * /sys/fs/resctrl/
 * |-- info
 * |   |-- L3
 * |   |   |-- cbm_mask
 * |   |   |-- min_cbm_bits
 * |   |   |-- num_closids
 * |   |-- L3_MON
 * |   |   |-- max_threshold_occupancy
 * |   |   |-- mon_features
 * |   |   |-- num_rmids
 * |   |-- MB
 * |       |-- bandwidth_gran
 * |       |-- delay_linear
 * |       |-- min_bandwidth
 * |       |-- num_closids
 * |-- ...
 * |-- schemata
 * |-- tasks
 * |-- <container_id>
 *     |-- ...
 *     |-- schemata
 *     |-- tasks
 *
 * For runc, we can make use of `tasks` and `schemata` configuration for L3
 * cache and memory bandwidth resources constraints.
 *
 * The file `tasks` has a list of tasks that belongs to this group (e.g.,
 * <container_id>" group). Tasks can be added to a group by writing the task ID
 * to the "tasks" file (which will automatically remove them from the previous
 * group to which they belonged). New tasks created by fork(2) and clone(2) are
 * added to the same group as their parent.
 *
 * The file `schemata` has a list of all the resources available to this group.
 * Each resource (L3 cache, memory bandwidth) has its own line and format.
 *
 * L3 cache schema:
 * It has allocation bitmasks/values for L3 cache on each socket, which
 * contains L3 cache id and capacity bitmask (CBM).
 * 	Format: "L3:<cache_id0>=<cbm0>;<cache_id1>=<cbm1>;..."
 * For example, on a two-socket machine, the schema line could be "L3:0=ff;1=c0"
 * which means L3 cache id 0's CBM is 0xff, and L3 cache id 1's CBM is 0xc0.
 *
 * The valid L3 cache CBM is a *contiguous bits set* and number of bits that can
 * be set is less than the max bit. The max bits in the CBM is varied among
 * supported Intel CPU models. Kernel will check if it is valid when writing.
 * e.g., default value 0xfffff in root indicates the max bits of CBM is 20
 * bits, which mapping to entire L3 cache capacity. Some valid CBM values to
 * set in a group: 0xf, 0xf0, 0x3ff, 0x1f00 and etc.
 *
 * Memory bandwidth schema:
 * It has allocation values for memory bandwidth on each socket, which contains
 * L3 cache id and memory bandwidth.
 * 	Format: "MB:<cache_id0>=bandwidth0;<cache_id1>=bandwidth1;..."
 * For example, on a two-socket machine, the schema line could be "MB:0=20;1=70"
 *
 * The minimum bandwidth percentage value for each CPU model is predefined and
 * can be looked up through "info/MB/min_bandwidth". The bandwidth granularity
 * that is allocated is also dependent on the CPU model and can be looked up at
 * "info/MB/bandwidth_gran". The available bandwidth control steps are:
 * min_bw + N * bw_gran. Intermediate values are rounded to the next control
 * step available on the hardware.
 *
 * If MBA Software Controller is enabled through mount option "-o mba_MBps":
 * mount -t resctrl resctrl -o mba_MBps /sys/fs/resctrl
 * We could specify memory bandwidth in "MBps" (Mega Bytes per second) unit
 * instead of "percentages". The kernel underneath would use a software feedback
 * mechanism or a "Software Controller" which reads the actual bandwidth using
 * MBM counters and adjust the memory bandwidth percentages to ensure:
 * "actual memory bandwidth < user specified memory bandwidth".
 *
 * For example, on a two-socket machine, the schema line could be
 * "MB:0=5000;1=7000" which means 5000 MBps memory bandwidth limit on socket 0
 * and 7000 MBps memory bandwidth limit on socket 1.
 *
 * For more information about Intel RDT kernel interface:
 * https://www.kernel.org/doc/Documentation/x86/intel_rdt_ui.txt
 *
 * An example for runc:
 * Consider a two-socket machine with two L3 caches where the default CBM is
 * 0x7ff and the max CBM length is 11 bits, and minimum memory bandwidth of 10%
 * with a memory bandwidth granularity of 10%.
 *
 * Tasks inside the container only have access to the "upper" 7/11 of L3 cache
 * on socket 0 and the "lower" 5/11 L3 cache on socket 1, and may use a
 * maximum memory bandwidth of 20% on socket 0 and 70% on socket 1.
 *
 * "linux": {
 *     "intelRdt": {
 *         "l3CacheSchema": "L3:0=7f0;1=1f",
 *         "memBwSchema": "MB:0=20;1=70"
 * 	}
 * }
 */

type Manager interface {
	// Applies Intel RDT configuration to the process with the specified pid
	Apply(pid int) error

	// Returns statistics for Intel RDT
	GetStats() (*Stats, error)

	// Destroys the Intel RDT 'container_id' group
	Destroy() error

	// Returns Intel RDT path to save in a state file and to be able to
	// restore the object later
	GetPath() string

	// Set Intel RDT "resource control" filesystem as configured.
	Set(container *configs.Config) error
}

// This implements interface Manager
type intelRdtManager struct {
	mu     sync.Mutex
	config *configs.Config
	id     string
	path   string
}

func NewManager(config *configs.Config, id string, path string) Manager {
	return &intelRdtManager{
		config: config,
		id:     id,
		path:   path,
	}
}

const (
	IntelRdtTasks = "tasks"
)

var (
	// The absolute root path of the Intel RDT "resource control" filesystem
	intelRdtRoot     string
	intelRdtRootLock sync.Mutex

	// The flag to indicate if Intel RDT/CAT is enabled
	catEnabled bool
	// The flag to indicate if Intel RDT/MBA is enabled
	mbaEnabled bool
	// The flag to indicate if Intel RDT/MBA Software Controller is enabled
	mbaScEnabled bool

	// For Intel RDT initialization
	initOnce sync.Once
)

type intelRdtData struct {
	root   string
	config *configs.Config
	pid    int
}

// Check if Intel RDT sub-features are enabled in featuresInit()
func featuresInit() {
	initOnce.Do(func() {
		// 1. Check if hardware and kernel support Intel RDT sub-features
		flagsSet, err := parseCpuInfoFile("/proc/cpuinfo")
		if err != nil {
			return
		}

		// 2. Check if Intel RDT "resource control" filesystem is mounted
		// The user guarantees to mount the filesystem
		if !isIntelRdtMounted() {
			return
		}

		// 3. Double check if Intel RDT sub-features are available in
		// "resource control" filesystem. Intel RDT sub-features can be
		// selectively disabled or enabled by kernel command line
		// (e.g., rdt=!l3cat,mba) in 4.14 and newer kernel
		if flagsSet.CAT {
			if _, err := os.Stat(filepath.Join(intelRdtRoot, "info", "L3")); err == nil {
				catEnabled = true
			}
		}
		if mbaScEnabled {
			// We confirm MBA Software Controller is enabled in step 2,
			// MBA should be enabled because MBA Software Controller
			// depends on MBA
			mbaEnabled = true
		} else if flagsSet.MBA {
			if _, err := os.Stat(filepath.Join(intelRdtRoot, "info", "MB")); err == nil {
				mbaEnabled = true
			}
		}
		if flagsSet.MBMTotal || flagsSet.MBMLocal || flagsSet.CMT {
			if _, err := os.Stat(filepath.Join(intelRdtRoot, "info", "L3_MON")); err != nil {
				return
			}
			enabledMonFeatures, err = getMonFeatures(intelRdtRoot)
			if err != nil {
				return
			}
			if enabledMonFeatures.mbmTotalBytes || enabledMonFeatures.mbmLocalBytes {
				mbmEnabled = true
			}
			if enabledMonFeatures.llcOccupancy {
				cmtEnabled = true
			}
		}
	})
}

// Return the mount point path of Intel RDT "resource control" filesysem
func findIntelRdtMountpointDir(f io.Reader) (string, error) {
	mi, err := mountinfo.GetMountsFromReader(f, func(m *mountinfo.Info) (bool, bool) {
		// similar to mountinfo.FSTypeFilter but stops after the first match
		if m.FSType == "resctrl" {
			return false, true // don't skip, stop
		}
		return true, false // skip, keep going
	})
	if err != nil {
		return "", err
	}
	if len(mi) < 1 {
		return "", NewNotFoundError("Intel RDT")
	}

	// Check if MBA Software Controller is enabled through mount option "-o mba_MBps"
	if strings.Contains(","+mi[0].VFSOptions+",", ",mba_MBps,") {
		mbaScEnabled = true
	}

	return mi[0].Mountpoint, nil
}

// Gets the root path of Intel RDT "resource control" filesystem
func getIntelRdtRoot() (string, error) {
	intelRdtRootLock.Lock()
	defer intelRdtRootLock.Unlock()

	if intelRdtRoot != "" {
		return intelRdtRoot, nil
	}

	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return "", err
	}
	root, err := findIntelRdtMountpointDir(f)
	f.Close()
	if err != nil {
		return "", err
	}

	if _, err := os.Stat(root); err != nil {
		return "", err
	}

	intelRdtRoot = root
	return intelRdtRoot, nil
}

func isIntelRdtMounted() bool {
	_, err := getIntelRdtRoot()
	return err == nil
}

type cpuInfoFlags struct {
	CAT bool // Cache Allocation Technology
	MBA bool // Memory Bandwidth Allocation

	// Memory Bandwidth Monitoring related.
	MBMTotal bool
	MBMLocal bool

	CMT bool // Cache Monitoring Technology
}

func parseCpuInfoFile(path string) (cpuInfoFlags, error) {
	infoFlags := cpuInfoFlags{}

	f, err := os.Open(path)
	if err != nil {
		return infoFlags, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()

		// Search "cat_l3" and "mba" flags in first "flags" line
		if strings.HasPrefix(line, "flags") {
			flags := strings.Split(line, " ")
			// "cat_l3" flag for CAT and "mba" flag for MBA
			for _, flag := range flags {
				switch flag {
				case "cat_l3":
					infoFlags.CAT = true
				case "mba":
					infoFlags.MBA = true
				case "cqm_mbm_total":
					infoFlags.MBMTotal = true
				case "cqm_mbm_local":
					infoFlags.MBMLocal = true
				case "cqm_occup_llc":
					infoFlags.CMT = true
				}
			}
			return infoFlags, nil
		}
	}
	if err := s.Err(); err != nil {
		return infoFlags, err
	}

	return infoFlags, nil
}

func parseUint(s string, base, bitSize int) (uint64, error) {
	value, err := strconv.ParseUint(s, base, bitSize)
	if err != nil {
		intValue, intErr := strconv.ParseInt(s, base, bitSize)
		// 1. Handle negative values greater than MinInt64 (and)
		// 2. Handle negative values lesser than MinInt64
		if intErr == nil && intValue < 0 {
			return 0, nil
		} else if intErr != nil && intErr.(*strconv.NumError).Err == strconv.ErrRange && intValue < 0 {
			return 0, nil
		}

		return value, err
	}

	return value, nil
}

// Gets a single uint64 value from the specified file.
func getIntelRdtParamUint(path, file string) (uint64, error) {
	fileName := filepath.Join(path, file)
	contents, err := ioutil.ReadFile(fileName)
	if err != nil {
		return 0, err
	}

	res, err := parseUint(string(bytes.TrimSpace(contents)), 10, 64)
	if err != nil {
		return res, fmt.Errorf("unable to parse %q as a uint from file %q", string(contents), fileName)
	}
	return res, nil
}

// Gets a string value from the specified file
func getIntelRdtParamString(path, file string) (string, error) {
	contents, err := ioutil.ReadFile(filepath.Join(path, file))
	if err != nil {
		return "", err
	}

	return string(bytes.TrimSpace(contents)), nil
}

func writeFile(dir, file, data string) error {
	if dir == "" {
		return fmt.Errorf("no such directory for %s", file)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, file), []byte(data+"\n"), 0o600); err != nil {
		return fmt.Errorf("failed to write %v to %v: %v", data, file, err)
	}
	return nil
}

func getIntelRdtData(c *configs.Config, pid int) (*intelRdtData, error) {
	rootPath, err := getIntelRdtRoot()
	if err != nil {
		return nil, err
	}
	return &intelRdtData{
		root:   rootPath,
		config: c,
		pid:    pid,
	}, nil
}

// Get the read-only L3 cache information
func getL3CacheInfo() (*L3CacheInfo, error) {
	l3CacheInfo := &L3CacheInfo{}

	rootPath, err := getIntelRdtRoot()
	if err != nil {
		return l3CacheInfo, err
	}

	path := filepath.Join(rootPath, "info", "L3")
	cbmMask, err := getIntelRdtParamString(path, "cbm_mask")
	if err != nil {
		return l3CacheInfo, err
	}
	minCbmBits, err := getIntelRdtParamUint(path, "min_cbm_bits")
	if err != nil {
		return l3CacheInfo, err
	}
	numClosids, err := getIntelRdtParamUint(path, "num_closids")
	if err != nil {
		return l3CacheInfo, err
	}

	l3CacheInfo.CbmMask = cbmMask
	l3CacheInfo.MinCbmBits = minCbmBits
	l3CacheInfo.NumClosids = numClosids

	return l3CacheInfo, nil
}

// Get the read-only memory bandwidth information
func getMemBwInfo() (*MemBwInfo, error) {
	memBwInfo := &MemBwInfo{}

	rootPath, err := getIntelRdtRoot()
	if err != nil {
		return memBwInfo, err
	}

	path := filepath.Join(rootPath, "info", "MB")
	bandwidthGran, err := getIntelRdtParamUint(path, "bandwidth_gran")
	if err != nil {
		return memBwInfo, err
	}
	delayLinear, err := getIntelRdtParamUint(path, "delay_linear")
	if err != nil {
		return memBwInfo, err
	}
	minBandwidth, err := getIntelRdtParamUint(path, "min_bandwidth")
	if err != nil {
		return memBwInfo, err
	}
	numClosids, err := getIntelRdtParamUint(path, "num_closids")
	if err != nil {
		return memBwInfo, err
	}

	memBwInfo.BandwidthGran = bandwidthGran
	memBwInfo.DelayLinear = delayLinear
	memBwInfo.MinBandwidth = minBandwidth
	memBwInfo.NumClosids = numClosids

	return memBwInfo, nil
}

// Get diagnostics for last filesystem operation error from file info/last_cmd_status
func getLastCmdStatus() (string, error) {
	rootPath, err := getIntelRdtRoot()
	if err != nil {
		return "", err
	}

	path := filepath.Join(rootPath, "info")
	lastCmdStatus, err := getIntelRdtParamString(path, "last_cmd_status")
	if err != nil {
		return "", err
	}

	return lastCmdStatus, nil
}

// WriteIntelRdtTasks writes the specified pid into the "tasks" file
func WriteIntelRdtTasks(dir string, pid int) error {
	if dir == "" {
		return fmt.Errorf("no such directory for %s", IntelRdtTasks)
	}

	// Don't attach any pid if -1 is specified as a pid
	if pid != -1 {
		if err := ioutil.WriteFile(filepath.Join(dir, IntelRdtTasks), []byte(strconv.Itoa(pid)), 0o600); err != nil {
			return fmt.Errorf("failed to write %v to %v: %v", pid, IntelRdtTasks, err)
		}
	}
	return nil
}

// Check if Intel RDT/CAT is enabled
func IsCATEnabled() bool {
	featuresInit()
	return catEnabled
}

// Check if Intel RDT/MBA is enabled
func IsMBAEnabled() bool {
	featuresInit()
	return mbaEnabled
}

// Check if Intel RDT/MBA Software Controller is enabled
func IsMBAScEnabled() bool {
	featuresInit()
	return mbaScEnabled
}

// Get the 'container_id' path in Intel RDT "resource control" filesystem
func GetIntelRdtPath(id string) (string, error) {
	rootPath, err := getIntelRdtRoot()
	if err != nil {
		return "", err
	}

	path := filepath.Join(rootPath, id)
	return path, nil
}

// Applies Intel RDT configuration to the process with the specified pid
func (m *intelRdtManager) Apply(pid int) (err error) {
	// If intelRdt is not specified in config, we do nothing
	if m.config.IntelRdt == nil {
		return nil
	}
	d, err := getIntelRdtData(m.config, pid)
	if err != nil && !IsNotFound(err) {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	path, err := d.join(m.id)
	if err != nil {
		return err
	}

	m.path = path
	return nil
}

// Destroys the Intel RDT 'container_id' group
func (m *intelRdtManager) Destroy() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err := os.RemoveAll(m.GetPath()); err != nil {
		return err
	}
	m.path = ""
	return nil
}

// Returns Intel RDT path to save in a state file and to be able to
// restore the object later
func (m *intelRdtManager) GetPath() string {
	if m.path == "" {
		m.path, _ = GetIntelRdtPath(m.id)
	}
	return m.path
}

// Returns statistics for Intel RDT
func (m *intelRdtManager) GetStats() (*Stats, error) {
	// If intelRdt is not specified in config
	if m.config.IntelRdt == nil {
		return nil, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	stats := NewStats()

	rootPath, err := getIntelRdtRoot()
	if err != nil {
		return nil, err
	}
	// The read-only L3 cache and memory bandwidth schemata in root
	tmpRootStrings, err := getIntelRdtParamString(rootPath, "schemata")
	if err != nil {
		return nil, err
	}
	schemaRootStrings := strings.Split(tmpRootStrings, "\n")

	// The L3 cache and memory bandwidth schemata in 'container_id' group
	containerPath := m.GetPath()
	tmpStrings, err := getIntelRdtParamString(containerPath, "schemata")
	if err != nil {
		return nil, err
	}
	schemaStrings := strings.Split(tmpStrings, "\n")

	if IsCATEnabled() {
		// The read-only L3 cache information
		l3CacheInfo, err := getL3CacheInfo()
		if err != nil {
			return nil, err
		}
		stats.L3CacheInfo = l3CacheInfo

		// The read-only L3 cache schema in root
		for _, schemaRoot := range schemaRootStrings {
			if strings.Contains(schemaRoot, "L3") {
				stats.L3CacheSchemaRoot = strings.TrimSpace(schemaRoot)
			}
		}

		// The L3 cache schema in 'container_id' group
		for _, schema := range schemaStrings {
			if strings.Contains(schema, "L3") {
				stats.L3CacheSchema = strings.TrimSpace(schema)
			}
		}
	}

	if IsMBAEnabled() {
		// The read-only memory bandwidth information
		memBwInfo, err := getMemBwInfo()
		if err != nil {
			return nil, err
		}
		stats.MemBwInfo = memBwInfo

		// The read-only memory bandwidth information
		for _, schemaRoot := range schemaRootStrings {
			if strings.Contains(schemaRoot, "MB") {
				stats.MemBwSchemaRoot = strings.TrimSpace(schemaRoot)
			}
		}

		// The memory bandwidth schema in 'container_id' group
		for _, schema := range schemaStrings {
			if strings.Contains(schema, "MB") {
				stats.MemBwSchema = strings.TrimSpace(schema)
			}
		}
	}

	if IsMBMEnabled() || IsCMTEnabled() {
		err = getMonitoringStats(containerPath, stats)
		if err != nil {
			return nil, err
		}
	}

	return stats, nil
}

// Set Intel RDT "resource control" filesystem as configured.
func (m *intelRdtManager) Set(container *configs.Config) error {
	// About L3 cache schema:
	// It has allocation bitmasks/values for L3 cache on each socket,
	// which contains L3 cache id and capacity bitmask (CBM).
	// 	Format: "L3:<cache_id0>=<cbm0>;<cache_id1>=<cbm1>;..."
	// For example, on a two-socket machine, the schema line could be:
	// 	L3:0=ff;1=c0
	// which means L3 cache id 0's CBM is 0xff, and L3 cache id 1's CBM
	// is 0xc0.
	//
	// The valid L3 cache CBM is a *contiguous bits set* and number of
	// bits that can be set is less than the max bit. The max bits in the
	// CBM is varied among supported Intel CPU models. Kernel will check
	// if it is valid when writing. e.g., default value 0xfffff in root
	// indicates the max bits of CBM is 20 bits, which mapping to entire
	// L3 cache capacity. Some valid CBM values to set in a group:
	// 0xf, 0xf0, 0x3ff, 0x1f00 and etc.
	//
	//
	// About memory bandwidth schema:
	// It has allocation values for memory bandwidth on each socket, which
	// contains L3 cache id and memory bandwidth.
	// 	Format: "MB:<cache_id0>=bandwidth0;<cache_id1>=bandwidth1;..."
	// For example, on a two-socket machine, the schema line could be:
	// 	"MB:0=20;1=70"
	//
	// The minimum bandwidth percentage value for each CPU model is
	// predefined and can be looked up through "info/MB/min_bandwidth".
	// The bandwidth granularity that is allocated is also dependent on
	// the CPU model and can be looked up at "info/MB/bandwidth_gran".
	// The available bandwidth control steps are: min_bw + N * bw_gran.
	// Intermediate values are rounded to the next control step available
	// on the hardware.
	//
	// If MBA Software Controller is enabled through mount option
	// "-o mba_MBps": mount -t resctrl resctrl -o mba_MBps /sys/fs/resctrl
	// We could specify memory bandwidth in "MBps" (Mega Bytes per second)
	// unit instead of "percentages". The kernel underneath would use a
	// software feedback mechanism or a "Software Controller" which reads
	// the actual bandwidth using MBM counters and adjust the memory
	// bandwidth percentages to ensure:
	// "actual memory bandwidth < user specified memory bandwidth".
	//
	// For example, on a two-socket machine, the schema line could be
	// "MB:0=5000;1=7000" which means 5000 MBps memory bandwidth limit on
	// socket 0 and 7000 MBps memory bandwidth limit on socket 1.
	if container.IntelRdt != nil {
		path := m.GetPath()
		l3CacheSchema := container.IntelRdt.L3CacheSchema
		memBwSchema := container.IntelRdt.MemBwSchema

		// Write a single joint schema string to schemata file
		if l3CacheSchema != "" && memBwSchema != "" {
			if err := writeFile(path, "schemata", l3CacheSchema+"\n"+memBwSchema); err != nil {
				return NewLastCmdError(err)
			}
		}

		// Write only L3 cache schema string to schemata file
		if l3CacheSchema != "" && memBwSchema == "" {
			if err := writeFile(path, "schemata", l3CacheSchema); err != nil {
				return NewLastCmdError(err)
			}
		}

		// Write only memory bandwidth schema string to schemata file
		if l3CacheSchema == "" && memBwSchema != "" {
			if err := writeFile(path, "schemata", memBwSchema); err != nil {
				return NewLastCmdError(err)
			}
		}
	}

	return nil
}

func (raw *intelRdtData) join(id string) (string, error) {
	path := filepath.Join(raw.root, id)
	if err := os.MkdirAll(path, 0o755); err != nil {
		return "", NewLastCmdError(err)
	}

	if err := WriteIntelRdtTasks(path, raw.pid); err != nil {
		return "", NewLastCmdError(err)
	}
	return path, nil
}

type NotFoundError struct {
	ResourceControl string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("mountpoint for %s not found", e.ResourceControl)
}

func NewNotFoundError(res string) error {
	return &NotFoundError{
		ResourceControl: res,
	}
}

func IsNotFound(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*NotFoundError)
	return ok
}

type LastCmdError struct {
	LastCmdStatus string
	Err           error
}

func (e *LastCmdError) Error() string {
	return e.Err.Error() + ", last_cmd_status: " + e.LastCmdStatus
}

func NewLastCmdError(err error) error {
	lastCmdStatus, err1 := getLastCmdStatus()
	if err1 == nil {
		return &LastCmdError{
			LastCmdStatus: lastCmdStatus,
			Err:           err,
		}
	}
	return err
}
