package intelrdt

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/moby/sys/mountinfo"
	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"
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
 * |-- <clos>
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

type Manager struct {
	mu     sync.Mutex
	config *configs.Config
	id     string
	path   string
}

// NewManager returns a new instance of Manager, or nil if the Intel RDT
// functionality is not specified in the config, available from hardware or
// enabled in the kernel.
func NewManager(config *configs.Config, id string, path string) *Manager {
	if config.IntelRdt == nil {
		return nil
	}
	if _, err := Root(); err != nil {
		// Intel RDT is not available.
		return nil
	}
	return newManager(config, id, path)
}

// newManager is the same as NewManager, except it does not check if the feature
// is actually available. Used by unit tests that mock intelrdt paths.
func newManager(config *configs.Config, id string, path string) *Manager {
	return &Manager{
		config: config,
		id:     id,
		path:   path,
	}
}

const (
	intelRdtTasks = "tasks"
)

var (
	// The flag to indicate if Intel RDT/CAT is enabled
	catEnabled bool
	// The flag to indicate if Intel RDT/MBA is enabled
	mbaEnabled bool

	// For Intel RDT initialization
	initOnce sync.Once

	errNotFound = errors.New("Intel RDT not available")
)

// Check if Intel RDT sub-features are enabled in featuresInit()
func featuresInit() {
	initOnce.Do(func() {
		// 1. Check if Intel RDT "resource control" filesystem is available.
		// The user guarantees to mount the filesystem.
		root, err := Root()
		if err != nil {
			return
		}

		// 2. Check if Intel RDT sub-features are available in "resource
		// control" filesystem. Intel RDT sub-features can be
		// selectively disabled or enabled by kernel command line
		// (e.g., rdt=!l3cat,mba) in 4.14 and newer kernel
		if _, err := os.Stat(filepath.Join(root, "info", "L3")); err == nil {
			catEnabled = true
		}
		if _, err := os.Stat(filepath.Join(root, "info", "MB")); err == nil {
			mbaEnabled = true
		}
		if _, err := os.Stat(filepath.Join(root, "info", "L3_MON")); err != nil {
			return
		}
		enabledMonFeatures, err = getMonFeatures(root)
		if err != nil {
			return
		}
		if enabledMonFeatures.mbmTotalBytes || enabledMonFeatures.mbmLocalBytes {
			mbmEnabled = true
		}
		if enabledMonFeatures.llcOccupancy {
			cmtEnabled = true
		}
	})
}

// findIntelRdtMountpointDir returns the mount point of the Intel RDT "resource control" filesystem.
func findIntelRdtMountpointDir() (string, error) {
	mi, err := mountinfo.GetMounts(func(m *mountinfo.Info) (bool, bool) {
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
		return "", errNotFound
	}

	return mi[0].Mountpoint, nil
}

// For Root() use only.
var (
	intelRdtRoot    string
	intelRdtRootErr error
	rootOnce        sync.Once
)

// The kernel creates this (empty) directory if resctrl is supported by the
// hardware and kernel. The user is responsible for mounting the resctrl
// filesystem, and they could mount it somewhere else if they wanted to.
const defaultResctrlMountpoint = "/sys/fs/resctrl"

// Root returns the Intel RDT "resource control" filesystem mount point.
func Root() (string, error) {
	rootOnce.Do(func() {
		// Does this system support resctrl?
		var statfs unix.Statfs_t
		if err := unix.Statfs(defaultResctrlMountpoint, &statfs); err != nil {
			if errors.Is(err, unix.ENOENT) {
				err = errNotFound
			}
			intelRdtRootErr = err
			return
		}

		// Has the resctrl fs been mounted to the default mount point?
		if statfs.Type == unix.RDTGROUP_SUPER_MAGIC {
			intelRdtRoot = defaultResctrlMountpoint
			return
		}

		// The resctrl fs could have been mounted somewhere nonstandard.
		intelRdtRoot, intelRdtRootErr = findIntelRdtMountpointDir()
	})

	return intelRdtRoot, intelRdtRootErr
}

// Gets a single uint64 value from the specified file.
func getIntelRdtParamUint(path, file string) (uint64, error) {
	fileName := filepath.Join(path, file)
	contents, err := os.ReadFile(fileName)
	if err != nil {
		return 0, err
	}

	res, err := fscommon.ParseUint(string(bytes.TrimSpace(contents)), 10, 64)
	if err != nil {
		return res, fmt.Errorf("unable to parse %q as a uint from file %q", string(contents), fileName)
	}
	return res, nil
}

// Gets a string value from the specified file
func getIntelRdtParamString(path, file string) (string, error) {
	contents, err := os.ReadFile(filepath.Join(path, file))
	if err != nil {
		return "", err
	}

	return string(bytes.TrimSpace(contents)), nil
}

func writeFile(dir, file, data string) error {
	if dir == "" {
		return fmt.Errorf("no such directory for %s", file)
	}
	if err := os.WriteFile(filepath.Join(dir, file), []byte(data+"\n"), 0o600); err != nil {
		return newLastCmdError(fmt.Errorf("intelrdt: unable to write %v: %w", data, err))
	}
	return nil
}

// Get the read-only L3 cache information
func getL3CacheInfo() (*L3CacheInfo, error) {
	l3CacheInfo := &L3CacheInfo{}

	rootPath, err := Root()
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

	rootPath, err := Root()
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
	rootPath, err := Root()
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
		return fmt.Errorf("no such directory for %s", intelRdtTasks)
	}

	// Don't attach any pid if -1 is specified as a pid
	if pid != -1 {
		if err := os.WriteFile(filepath.Join(dir, intelRdtTasks), []byte(strconv.Itoa(pid)), 0o600); err != nil {
			return newLastCmdError(fmt.Errorf("intelrdt: unable to add pid %d: %w", pid, err))
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

// Get the path of the clos group in "resource control" filesystem that the container belongs to
func (m *Manager) getIntelRdtPath() (string, error) {
	rootPath, err := Root()
	if err != nil {
		return "", err
	}

	clos := m.id
	if m.config.IntelRdt != nil && m.config.IntelRdt.ClosID != "" {
		clos = m.config.IntelRdt.ClosID
	}

	return filepath.Join(rootPath, clos), nil
}

// Applies Intel RDT configuration to the process with the specified pid
func (m *Manager) Apply(pid int) (err error) {
	// If intelRdt is not specified in config, we do nothing
	if m.config.IntelRdt == nil {
		return nil
	}

	path, err := m.getIntelRdtPath()
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if m.config.IntelRdt.ClosID != "" && m.config.IntelRdt.L3CacheSchema == "" && m.config.IntelRdt.MemBwSchema == "" {
		// Check that the CLOS exists, i.e. it has been pre-configured to
		// conform with the runtime spec
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("clos dir not accessible (must be pre-created when l3CacheSchema and memBwSchema are empty): %w", err)
		}
	}

	if err := os.MkdirAll(path, 0o755); err != nil {
		return newLastCmdError(err)
	}

	if err := WriteIntelRdtTasks(path, pid); err != nil {
		return newLastCmdError(err)
	}

	m.path = path
	return nil
}

// Destroys the Intel RDT container-specific 'container_id' group
func (m *Manager) Destroy() error {
	// Don't remove resctrl group if closid has been explicitly specified. The
	// group is likely externally managed, i.e. by some other entity than us.
	// There are probably other containers/tasks sharing the same group.
	if m.config.IntelRdt != nil && m.config.IntelRdt.ClosID == "" {
		m.mu.Lock()
		defer m.mu.Unlock()
		if err := os.RemoveAll(m.GetPath()); err != nil {
			return err
		}
		m.path = ""
	}
	return nil
}

// Returns Intel RDT path to save in a state file and to be able to
// restore the object later
func (m *Manager) GetPath() string {
	if m.path == "" {
		m.path, _ = m.getIntelRdtPath()
	}
	return m.path
}

// Returns statistics for Intel RDT
func (m *Manager) GetStats() (*Stats, error) {
	// If intelRdt is not specified in config
	if m.config.IntelRdt == nil {
		return nil, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	stats := newStats()

	rootPath, err := Root()
	if err != nil {
		return nil, err
	}
	// The read-only L3 cache and memory bandwidth schemata in root
	tmpRootStrings, err := getIntelRdtParamString(rootPath, "schemata")
	if err != nil {
		return nil, err
	}
	schemaRootStrings := strings.Split(tmpRootStrings, "\n")

	// The L3 cache and memory bandwidth schemata in container's clos group
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

		// The L3 cache schema in container's clos group
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

		// The memory bandwidth schema in container's clos group
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
func (m *Manager) Set(container *configs.Config) error {
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

		// TODO: verify that l3CacheSchema and/or memBwSchema match the
		// existing schemata if ClosID has been specified. This is a more
		// involved than reading the file and doing plain string comparison as
		// the value written in does not necessarily match what gets read out
		// (leading zeros, cache id ordering etc).

		// Write a single joint schema string to schemata file
		if l3CacheSchema != "" && memBwSchema != "" {
			if err := writeFile(path, "schemata", l3CacheSchema+"\n"+memBwSchema); err != nil {
				return err
			}
		}

		// Write only L3 cache schema string to schemata file
		if l3CacheSchema != "" && memBwSchema == "" {
			if err := writeFile(path, "schemata", l3CacheSchema); err != nil {
				return err
			}
		}

		// Write only memory bandwidth schema string to schemata file
		if l3CacheSchema == "" && memBwSchema != "" {
			if err := writeFile(path, "schemata", memBwSchema); err != nil {
				return err
			}
		}
	}

	return nil
}

func newLastCmdError(err error) error {
	status, err1 := getLastCmdStatus()
	if err1 == nil {
		return fmt.Errorf("%w, last_cmd_status: %s", err, status)
	}
	return err
}
