// +build solaris,cgo

package daemon

import (
	"fmt"
	"net"
	"strconv"

	"github.com/docker/docker/api/types"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/sysinfo"
	refstore "github.com/docker/docker/reference"
	"github.com/docker/libnetwork"
	nwconfig "github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/drivers/solaris/bridge"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	lntypes "github.com/docker/libnetwork/types"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/opencontainers/selinux/go-selinux/label"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

//#include <zone.h>
import "C"

const (
	defaultVirtualSwitch = "Virtual Switch"
	platformSupported    = true
	solarisMinCPUShares  = 1
	solarisMaxCPUShares  = 65535
)

func getMemoryResources(config containertypes.Resources) specs.CappedMemory {
	memory := specs.CappedMemory{}

	if config.Memory > 0 {
		memory.Physical = strconv.FormatInt(config.Memory, 10)
	}

	if config.MemorySwap != 0 {
		memory.Swap = strconv.FormatInt(config.MemorySwap, 10)
	}

	return memory
}

func getCPUResources(config containertypes.Resources) specs.CappedCPU {
	cpu := specs.CappedCPU{}

	if config.CpusetCpus != "" {
		cpu.Ncpus = config.CpusetCpus
	}

	return cpu
}

func (daemon *Daemon) cleanupMountsByID(id string) error {
	return nil
}

func (daemon *Daemon) parseSecurityOpt(container *container.Container, hostConfig *containertypes.HostConfig) error {
	return parseSecurityOpt(container, hostConfig)
}

func parseSecurityOpt(container *container.Container, config *containertypes.HostConfig) error {
	//Since config.SecurityOpt is specifically defined as a "List of string values to
	//customize labels for MLs systems, such as SELinux"
	//until we figure out how to map to Trusted Extensions
	//this is being disabled for now on Solaris
	var (
		labelOpts []string
		err       error
	)

	if len(config.SecurityOpt) > 0 {
		return errors.New("Security options are not supported on Solaris")
	}

	container.ProcessLabel, container.MountLabel, err = label.InitLabels(labelOpts)
	return err
}

func setupRemappedRoot(config *Config) ([]idtools.IDMap, []idtools.IDMap, error) {
	return nil, nil, nil
}

func setupDaemonRoot(config *Config, rootDir string, rootUID, rootGID int) error {
	return nil
}

func (daemon *Daemon) getLayerInit() func(string) error {
	return nil
}

func checkKernel() error {
	// solaris can rely upon checkSystem() below, we don't skew kernel versions
	return nil
}

func (daemon *Daemon) getCgroupDriver() string {
	return ""
}

func (daemon *Daemon) adaptContainerSettings(hostConfig *containertypes.HostConfig, adjustCPUShares bool) error {
	if hostConfig.CPUShares < 0 {
		logrus.Warnf("Changing requested CPUShares of %d to minimum allowed of %d", hostConfig.CPUShares, solarisMinCPUShares)
		hostConfig.CPUShares = solarisMinCPUShares
	} else if hostConfig.CPUShares > solarisMaxCPUShares {
		logrus.Warnf("Changing requested CPUShares of %d to maximum allowed of %d", hostConfig.CPUShares, solarisMaxCPUShares)
		hostConfig.CPUShares = solarisMaxCPUShares
	}

	if hostConfig.Memory > 0 && hostConfig.MemorySwap == 0 {
		// By default, MemorySwap is set to twice the size of Memory.
		hostConfig.MemorySwap = hostConfig.Memory * 2
	}

	if hostConfig.ShmSize != 0 {
		hostConfig.ShmSize = container.DefaultSHMSize
	}
	if hostConfig.OomKillDisable == nil {
		defaultOomKillDisable := false
		hostConfig.OomKillDisable = &defaultOomKillDisable
	}

	return nil
}

// UsingSystemd returns true if cli option includes native.cgroupdriver=systemd
func UsingSystemd(config *Config) bool {
	return false
}

// verifyPlatformContainerSettings performs platform-specific validation of the
// hostconfig and config structures.
func verifyPlatformContainerSettings(daemon *Daemon, hostConfig *containertypes.HostConfig, config *containertypes.Config, update bool) ([]string, error) {
	fixMemorySwappiness(resources)
	warnings := []string{}
	sysInfo := sysinfo.New(true)
	// NOTE: We do not enforce a minimum value for swap limits for zones on Solaris and
	// therefore we will not do that for Docker container either.
	if hostConfig.Memory > 0 && !sysInfo.MemoryLimit {
		warnings = append(warnings, "Your kernel does not support memory limit capabilities. Limitation discarded.")
		logrus.Warnf("Your kernel does not support memory limit capabilities. Limitation discarded.")
		hostConfig.Memory = 0
		hostConfig.MemorySwap = -1
	}
	if hostConfig.Memory > 0 && hostConfig.MemorySwap != -1 && !sysInfo.SwapLimit {
		warnings = append(warnings, "Your kernel does not support swap limit capabilities, memory limited without swap.")
		logrus.Warnf("Your kernel does not support swap limit capabilities, memory limited without swap.")
		hostConfig.MemorySwap = -1
	}
	if hostConfig.Memory > 0 && hostConfig.MemorySwap > 0 && hostConfig.MemorySwap < hostConfig.Memory {
		return warnings, fmt.Errorf("Minimum memoryswap limit should be larger than memory limit, see usage.")
	}
	// Solaris NOTE: We allow and encourage setting the swap without setting the memory limit.

	if hostConfig.MemorySwappiness != nil && !sysInfo.MemorySwappiness {
		warnings = append(warnings, "Your kernel does not support memory swappiness capabilities, memory swappiness discarded.")
		logrus.Warnf("Your kernel does not support memory swappiness capabilities, memory swappiness discarded.")
		hostConfig.MemorySwappiness = nil
	}
	if hostConfig.MemoryReservation > 0 && !sysInfo.MemoryReservation {
		warnings = append(warnings, "Your kernel does not support memory soft limit capabilities. Limitation discarded.")
		logrus.Warnf("Your kernel does not support memory soft limit capabilities. Limitation discarded.")
		hostConfig.MemoryReservation = 0
	}
	if hostConfig.Memory > 0 && hostConfig.MemoryReservation > 0 && hostConfig.Memory < hostConfig.MemoryReservation {
		return warnings, fmt.Errorf("Minimum memory limit should be larger than memory reservation limit, see usage.")
	}
	if hostConfig.KernelMemory > 0 && !sysInfo.KernelMemory {
		warnings = append(warnings, "Your kernel does not support kernel memory limit capabilities. Limitation discarded.")
		logrus.Warnf("Your kernel does not support kernel memory limit capabilities. Limitation discarded.")
		hostConfig.KernelMemory = 0
	}
	if hostConfig.CPUShares != 0 && !sysInfo.CPUShares {
		warnings = append(warnings, "Your kernel does not support CPU shares. Shares discarded.")
		logrus.Warnf("Your kernel does not support CPU shares. Shares discarded.")
		hostConfig.CPUShares = 0
	}
	if hostConfig.CPUShares < 0 {
		warnings = append(warnings, "Invalid CPUShares value. Must be positive. Discarding.")
		logrus.Warnf("Invalid CPUShares value. Must be positive. Discarding.")
		hostConfig.CPUQuota = 0
	}
	if hostConfig.CPUShares > 0 && !sysinfo.IsCPUSharesAvailable() {
		warnings = append(warnings, "Global zone default scheduling class not FSS. Discarding shares.")
		logrus.Warnf("Global zone default scheduling class not FSS. Discarding shares.")
		hostConfig.CPUShares = 0
	}

	// Solaris NOTE: Linux does not do negative checking for CPUShares and Quota here. But it makes sense to.
	if hostConfig.CPUPeriod > 0 && !sysInfo.CPUCfsPeriod {
		warnings = append(warnings, "Your kernel does not support CPU cfs period. Period discarded.")
		logrus.Warnf("Your kernel does not support CPU cfs period. Period discarded.")
		if hostConfig.CPUQuota > 0 {
			warnings = append(warnings, "Quota will be applied on default period, not period specified.")
			logrus.Warnf("Quota will be applied on default period, not period specified.")
		}
		hostConfig.CPUPeriod = 0
	}
	if hostConfig.CPUQuota != 0 && !sysInfo.CPUCfsQuota {
		warnings = append(warnings, "Your kernel does not support CPU cfs quota. Quota discarded.")
		logrus.Warnf("Your kernel does not support CPU cfs quota. Quota discarded.")
		hostConfig.CPUQuota = 0
	}
	if hostConfig.CPUQuota < 0 {
		warnings = append(warnings, "Invalid CPUQuota value. Must be positive. Discarding.")
		logrus.Warnf("Invalid CPUQuota value. Must be positive. Discarding.")
		hostConfig.CPUQuota = 0
	}
	if (hostConfig.CpusetCpus != "" || hostConfig.CpusetMems != "") && !sysInfo.Cpuset {
		warnings = append(warnings, "Your kernel does not support cpuset. Cpuset discarded.")
		logrus.Warnf("Your kernel does not support cpuset. Cpuset discarded.")
		hostConfig.CpusetCpus = ""
		hostConfig.CpusetMems = ""
	}
	cpusAvailable, err := sysInfo.IsCpusetCpusAvailable(hostConfig.CpusetCpus)
	if err != nil {
		return warnings, fmt.Errorf("Invalid value %s for cpuset cpus.", hostConfig.CpusetCpus)
	}
	if !cpusAvailable {
		return warnings, fmt.Errorf("Requested CPUs are not available - requested %s, available: %s.", hostConfig.CpusetCpus, sysInfo.Cpus)
	}
	memsAvailable, err := sysInfo.IsCpusetMemsAvailable(hostConfig.CpusetMems)
	if err != nil {
		return warnings, fmt.Errorf("Invalid value %s for cpuset mems.", hostConfig.CpusetMems)
	}
	if !memsAvailable {
		return warnings, fmt.Errorf("Requested memory nodes are not available - requested %s, available: %s.", hostConfig.CpusetMems, sysInfo.Mems)
	}
	if hostConfig.BlkioWeight > 0 && !sysInfo.BlkioWeight {
		warnings = append(warnings, "Your kernel does not support Block I/O weight. Weight discarded.")
		logrus.Warnf("Your kernel does not support Block I/O weight. Weight discarded.")
		hostConfig.BlkioWeight = 0
	}
	if hostConfig.OomKillDisable != nil && !sysInfo.OomKillDisable {
		*hostConfig.OomKillDisable = false
		// Don't warn; this is the default setting but only applicable to Linux
	}

	if sysInfo.IPv4ForwardingDisabled {
		warnings = append(warnings, "IPv4 forwarding is disabled. Networking will not work.")
		logrus.Warnf("IPv4 forwarding is disabled. Networking will not work")
	}

	// Solaris NOTE: We do not allow setting Linux specific options, so check and warn for all of them.

	if hostConfig.CapAdd != nil || hostConfig.CapDrop != nil {
		warnings = append(warnings, "Adding or dropping kernel capabilities unsupported on Solaris.Discarding capabilities lists.")
		logrus.Warnf("Adding or dropping kernel capabilities unsupported on Solaris.Discarding capabilities lists.")
		hostConfig.CapAdd = nil
		hostConfig.CapDrop = nil
	}

	if hostConfig.GroupAdd != nil {
		warnings = append(warnings, "Additional groups unsupported on Solaris.Discarding groups lists.")
		logrus.Warnf("Additional groups unsupported on Solaris.Discarding groups lists.")
		hostConfig.GroupAdd = nil
	}

	if hostConfig.IpcMode != "" {
		warnings = append(warnings, "IPC namespace assignment unsupported on Solaris.Discarding IPC setting.")
		logrus.Warnf("IPC namespace assignment unsupported on Solaris.Discarding IPC setting.")
		hostConfig.IpcMode = ""
	}

	if hostConfig.PidMode != "" {
		warnings = append(warnings, "PID namespace setting  unsupported on Solaris. Running container in host PID namespace.")
		logrus.Warnf("PID namespace setting  unsupported on Solaris. Running container in host PID namespace.")
		hostConfig.PidMode = ""
	}

	if hostConfig.Privileged {
		warnings = append(warnings, "Privileged mode unsupported on Solaris. Discarding privileged mode setting.")
		logrus.Warnf("Privileged mode unsupported on Solaris. Discarding privileged mode setting.")
		hostConfig.Privileged = false
	}

	if hostConfig.UTSMode != "" {
		warnings = append(warnings, "UTS namespace assignment unsupported on Solaris.Discarding UTS setting.")
		logrus.Warnf("UTS namespace assignment unsupported on Solaris.Discarding UTS setting.")
		hostConfig.UTSMode = ""
	}

	if hostConfig.CgroupParent != "" {
		warnings = append(warnings, "Specifying Cgroup parent unsupported on Solaris. Discarding cgroup parent setting.")
		logrus.Warnf("Specifying Cgroup parent unsupported on Solaris. Discarding cgroup parent setting.")
		hostConfig.CgroupParent = ""
	}

	if hostConfig.Ulimits != nil {
		warnings = append(warnings, "Specifying ulimits unsupported on Solaris. Discarding ulimits setting.")
		logrus.Warnf("Specifying ulimits unsupported on Solaris. Discarding ulimits setting.")
		hostConfig.Ulimits = nil
	}

	return warnings, nil
}

// reloadPlatform updates configuration with platform specific options
// and updates the passed attributes
func (daemon *Daemon) reloadPlatform(config *Config, attributes map[string]string) {
}

// verifyDaemonSettings performs validation of daemon config struct
func verifyDaemonSettings(config *Config) error {

	if config.DefaultRuntime == "" {
		config.DefaultRuntime = stockRuntimeName
	}
	if config.Runtimes == nil {
		config.Runtimes = make(map[string]types.Runtime)
	}
	stockRuntimeOpts := []string{}
	config.Runtimes[stockRuntimeName] = types.Runtime{Path: DefaultRuntimeBinary, Args: stockRuntimeOpts}

	// checkSystem validates platform-specific requirements
	return nil
}

func checkSystem() error {
	// check OS version for compatibility, ensure running in global zone
	var err error
	var id C.zoneid_t

	if id, err = C.getzoneid(); err != nil {
		return fmt.Errorf("Exiting. Error getting zone id: %+v", err)
	}
	if int(id) != 0 {
		return fmt.Errorf("Exiting because the Docker daemon is not running in the global zone")
	}

	v, err := kernel.GetKernelVersion()
	if kernel.CompareKernelVersion(*v, kernel.VersionInfo{Kernel: 5, Major: 12, Minor: 0}) < 0 {
		return fmt.Errorf("Your Solaris kernel version: %s doesn't support Docker. Please upgrade to 5.12.0", v.String())
	}
	return err
}

// configureMaxThreads sets the Go runtime max threads threshold
// which is 90% of the kernel setting from /proc/sys/kernel/threads-max
func configureMaxThreads(config *Config) error {
	return nil
}

// configureKernelSecuritySupport configures and validate security support for the kernel
func configureKernelSecuritySupport(config *config.Config, driverNames []string) error {
	return nil
}

func (daemon *Daemon) initNetworkController(config *Config, activeSandboxes map[string]interface{}) (libnetwork.NetworkController, error) {
	netOptions, err := daemon.networkOptions(config, daemon.PluginStore, activeSandboxes)
	if err != nil {
		return nil, err
	}

	controller, err := libnetwork.New(netOptions...)
	if err != nil {
		return nil, fmt.Errorf("error obtaining controller instance: %v", err)
	}

	// Initialize default network on "null"
	if _, err := controller.NewNetwork("null", "none", "", libnetwork.NetworkOptionPersist(false)); err != nil {
		return nil, fmt.Errorf("Error creating default 'null' network: %v", err)
	}

	if !config.DisableBridge {
		// Initialize default driver "bridge"
		if err := initBridgeDriver(controller, config); err != nil {
			return nil, err
		}
	}

	return controller, nil
}

func initBridgeDriver(controller libnetwork.NetworkController, config *Config) error {
	if n, err := controller.NetworkByName("bridge"); err == nil {
		if err = n.Delete(); err != nil {
			return fmt.Errorf("could not delete the default bridge network: %v", err)
		}
	}

	bridgeName := bridge.DefaultBridgeName
	if config.bridgeConfig.Iface != "" {
		bridgeName = config.bridgeConfig.Iface
	}
	netOption := map[string]string{
		bridge.BridgeName:    bridgeName,
		bridge.DefaultBridge: strconv.FormatBool(true),
		netlabel.DriverMTU:   strconv.Itoa(config.Mtu),
		bridge.EnableICC:     strconv.FormatBool(config.bridgeConfig.InterContainerCommunication),
	}

	// --ip processing
	if config.bridgeConfig.DefaultIP != nil {
		netOption[bridge.DefaultBindingIP] = config.bridgeConfig.DefaultIP.String()
	}

	var ipamV4Conf *libnetwork.IpamConf

	ipamV4Conf = &libnetwork.IpamConf{AuxAddresses: make(map[string]string)}

	nwList, _, err := netutils.ElectInterfaceAddresses(bridgeName)
	if err != nil {
		return errors.Wrap(err, "list bridge addresses failed")
	}

	nw := nwList[0]
	if len(nwList) > 1 && config.bridgeConfig.FixedCIDR != "" {
		_, fCIDR, err := net.ParseCIDR(config.bridgeConfig.FixedCIDR)
		if err != nil {
			return errors.Wrap(err, "parse CIDR failed")
		}
		// Iterate through in case there are multiple addresses for the bridge
		for _, entry := range nwList {
			if fCIDR.Contains(entry.IP) {
				nw = entry
				break
			}
		}
	}

	ipamV4Conf.PreferredPool = lntypes.GetIPNetCanonical(nw).String()
	hip, _ := lntypes.GetHostPartIP(nw.IP, nw.Mask)
	if hip.IsGlobalUnicast() {
		ipamV4Conf.Gateway = nw.IP.String()
	}

	if config.bridgeConfig.IP != "" {
		ipamV4Conf.PreferredPool = config.bridgeConfig.IP
		ip, _, err := net.ParseCIDR(config.bridgeConfig.IP)
		if err != nil {
			return err
		}
		ipamV4Conf.Gateway = ip.String()
	} else if bridgeName == bridge.DefaultBridgeName && ipamV4Conf.PreferredPool != "" {
		logrus.Infof("Default bridge (%s) is assigned with an IP address %s. Daemon option --bip can be used to set a preferred IP address", bridgeName, ipamV4Conf.PreferredPool)
	}

	if config.bridgeConfig.FixedCIDR != "" {
		_, fCIDR, err := net.ParseCIDR(config.bridgeConfig.FixedCIDR)
		if err != nil {
			return err
		}

		ipamV4Conf.SubPool = fCIDR.String()
	}

	if config.bridgeConfig.DefaultGatewayIPv4 != nil {
		ipamV4Conf.AuxAddresses["DefaultGatewayIPv4"] = config.bridgeConfig.DefaultGatewayIPv4.String()
	}

	v4Conf := []*libnetwork.IpamConf{ipamV4Conf}
	v6Conf := []*libnetwork.IpamConf{}

	// Initialize default network on "bridge" with the same name
	_, err = controller.NewNetwork("bridge", "bridge", "",
		libnetwork.NetworkOptionDriverOpts(netOption),
		libnetwork.NetworkOptionIpam("default", "", v4Conf, v6Conf, nil),
		libnetwork.NetworkOptionDeferIPv6Alloc(false))
	if err != nil {
		return fmt.Errorf("Error creating default 'bridge' network: %v", err)
	}
	return nil
}

// registerLinks sets up links between containers and writes the
// configuration out for persistence.
func (daemon *Daemon) registerLinks(container *container.Container, hostConfig *containertypes.HostConfig) error {
	return nil
}

func (daemon *Daemon) cleanupMounts() error {
	return nil
}

// conditionalMountOnStart is a platform specific helper function during the
// container start to call mount.
func (daemon *Daemon) conditionalMountOnStart(container *container.Container) error {
	return daemon.Mount(container)
}

// conditionalUnmountOnCleanup is a platform specific helper function called
// during the cleanup of a container to unmount.
func (daemon *Daemon) conditionalUnmountOnCleanup(container *container.Container) error {
	return daemon.Unmount(container)
}

func restoreCustomImage(is image.Store, ls layer.Store, rs refstore.Store) error {
	// Solaris has no custom images to register
	return nil
}

func driverOptions(config *Config) []nwconfig.Option {
	return []nwconfig.Option{}
}

func (daemon *Daemon) stats(c *container.Container) (*types.StatsJSON, error) {
	return nil, nil
}

// setDefaultIsolation determine the default isolation mode for the
// daemon to run in. This is only applicable on Windows
func (daemon *Daemon) setDefaultIsolation() error {
	return nil
}

func rootFSToAPIType(rootfs *image.RootFS) types.RootFS {
	return types.RootFS{}
}

func setupDaemonProcess(config *Config) error {
	return nil
}

func (daemon *Daemon) setupSeccompProfile() error {
	return nil
}

func getRealPath(path string) (string, error) {
	return fileutils.ReadSymlinkedDirectory(path)
}
