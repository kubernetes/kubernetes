package execdriver

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/daemon/execdriver/native/template"
	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/cgroups/fs"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func InitContainer(c *Command) *configs.Config {
	container := template.New()

	container.Hostname = getEnv("HOSTNAME", c.ProcessConfig.Env)
	container.Cgroups.Name = c.ID
	container.Cgroups.AllowedDevices = c.AllowedDevices
	container.Devices = c.AutoCreatedDevices
	container.Rootfs = c.Rootfs
	container.Readonlyfs = c.ReadonlyRootfs
	container.Privatefs = true

	// check to see if we are running in ramdisk to disable pivot root
	container.NoPivotRoot = os.Getenv("DOCKER_RAMDISK") != ""

	// Default parent cgroup is "docker". Override if required.
	if c.CgroupParent != "" {
		container.Cgroups.Parent = c.CgroupParent
	}
	return container
}

func getEnv(key string, env []string) string {
	for _, pair := range env {
		parts := strings.SplitN(pair, "=", 2)
		if parts[0] == key {
			return parts[1]
		}
	}
	return ""
}

func SetupCgroups(container *configs.Config, c *Command) error {
	if c.Resources != nil {
		container.Cgroups.CpuShares = c.Resources.CpuShares
		container.Cgroups.Memory = c.Resources.Memory
		container.Cgroups.MemoryReservation = c.Resources.Memory
		container.Cgroups.MemorySwap = c.Resources.MemorySwap
		container.Cgroups.CpusetCpus = c.Resources.CpusetCpus
		container.Cgroups.CpusetMems = c.Resources.CpusetMems
		container.Cgroups.CpuPeriod = c.Resources.CpuPeriod
		container.Cgroups.CpuQuota = c.Resources.CpuQuota
		container.Cgroups.BlkioWeight = c.Resources.BlkioWeight
		container.Cgroups.OomKillDisable = c.Resources.OomKillDisable
		container.Cgroups.MemorySwappiness = c.Resources.MemorySwappiness
	}

	return nil
}

// Returns the network statistics for the network interfaces represented by the NetworkRuntimeInfo.
func getNetworkInterfaceStats(interfaceName string) (*libcontainer.NetworkInterface, error) {
	out := &libcontainer.NetworkInterface{Name: interfaceName}
	// This can happen if the network runtime information is missing - possible if the
	// container was created by an old version of libcontainer.
	if interfaceName == "" {
		return out, nil
	}
	type netStatsPair struct {
		// Where to write the output.
		Out *uint64
		// The network stats file to read.
		File string
	}
	// Ingress for host veth is from the container. Hence tx_bytes stat on the host veth is actually number of bytes received by the container.
	netStats := []netStatsPair{
		{Out: &out.RxBytes, File: "tx_bytes"},
		{Out: &out.RxPackets, File: "tx_packets"},
		{Out: &out.RxErrors, File: "tx_errors"},
		{Out: &out.RxDropped, File: "tx_dropped"},

		{Out: &out.TxBytes, File: "rx_bytes"},
		{Out: &out.TxPackets, File: "rx_packets"},
		{Out: &out.TxErrors, File: "rx_errors"},
		{Out: &out.TxDropped, File: "rx_dropped"},
	}
	for _, netStat := range netStats {
		data, err := readSysfsNetworkStats(interfaceName, netStat.File)
		if err != nil {
			return nil, err
		}
		*(netStat.Out) = data
	}
	return out, nil
}

// Reads the specified statistics available under /sys/class/net/<EthInterface>/statistics
func readSysfsNetworkStats(ethInterface, statsFile string) (uint64, error) {
	data, err := ioutil.ReadFile(filepath.Join("/sys/class/net", ethInterface, "statistics", statsFile))
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}

func Stats(containerDir string, containerMemoryLimit int64, machineMemory int64) (*ResourceStats, error) {
	f, err := os.Open(filepath.Join(containerDir, "state.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	type network struct {
		Type              string
		HostInterfaceName string
	}

	state := struct {
		CgroupPaths map[string]string `json:"cgroup_paths"`
		Networks    []network
	}{}

	if err := json.NewDecoder(f).Decode(&state); err != nil {
		return nil, err
	}
	now := time.Now()

	mgr := fs.Manager{Paths: state.CgroupPaths}
	cstats, err := mgr.GetStats()
	if err != nil {
		return nil, err
	}
	stats := &libcontainer.Stats{CgroupStats: cstats}
	// if the container does not have any memory limit specified set the
	// limit to the machines memory
	memoryLimit := containerMemoryLimit
	if memoryLimit == 0 {
		memoryLimit = machineMemory
	}
	for _, iface := range state.Networks {
		switch iface.Type {
		case "veth":
			istats, err := getNetworkInterfaceStats(iface.HostInterfaceName)
			if err != nil {
				return nil, err
			}
			stats.Interfaces = append(stats.Interfaces, istats)
		}
	}
	return &ResourceStats{
		Stats:       stats,
		Read:        now,
		MemoryLimit: memoryLimit,
	}, nil
}
