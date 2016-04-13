// +build !windows

package daemon

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/nat"
	"github.com/docker/docker/pkg/parsers"
	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/runconfig"
	"github.com/docker/docker/utils"
	volumedrivers "github.com/docker/docker/volume/drivers"
	"github.com/docker/docker/volume/local"
	"github.com/docker/libnetwork"
	nwapi "github.com/docker/libnetwork/api"
	nwconfig "github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/options"
	"github.com/opencontainers/runc/libcontainer/label"
)

func (daemon *Daemon) Changes(container *Container) ([]archive.Change, error) {
	initID := fmt.Sprintf("%s-init", container.ID)
	return daemon.driver.Changes(container.ID, initID)
}

func (daemon *Daemon) Diff(container *Container) (archive.Archive, error) {
	initID := fmt.Sprintf("%s-init", container.ID)
	return daemon.driver.Diff(container.ID, initID)
}

func parseSecurityOpt(container *Container, config *runconfig.HostConfig) error {
	var (
		labelOpts []string
		err       error
	)

	for _, opt := range config.SecurityOpt {
		con := strings.SplitN(opt, ":", 2)
		if len(con) == 1 {
			return fmt.Errorf("Invalid --security-opt: %q", opt)
		}
		switch con[0] {
		case "label":
			labelOpts = append(labelOpts, con[1])
		case "apparmor":
			container.AppArmorProfile = con[1]
		default:
			return fmt.Errorf("Invalid --security-opt: %q", opt)
		}
	}

	container.ProcessLabel, container.MountLabel, err = label.InitLabels(labelOpts)
	return err
}

func (daemon *Daemon) createRootfs(container *Container) error {
	// Step 1: create the container directory.
	// This doubles as a barrier to avoid race conditions.
	if err := os.Mkdir(container.root, 0700); err != nil {
		return err
	}
	initID := fmt.Sprintf("%s-init", container.ID)
	if err := daemon.driver.Create(initID, container.ImageID); err != nil {
		return err
	}
	initPath, err := daemon.driver.Get(initID, "")
	if err != nil {
		return err
	}
	defer daemon.driver.Put(initID)

	if err := setupInitLayer(initPath); err != nil {
		return err
	}

	if err := daemon.driver.Create(container.ID, initID); err != nil {
		return err
	}
	return nil
}

func checkKernel() error {
	// Check for unsupported kernel versions
	// FIXME: it would be cleaner to not test for specific versions, but rather
	// test for specific functionalities.
	// Unfortunately we can't test for the feature "does not cause a kernel panic"
	// without actually causing a kernel panic, so we need this workaround until
	// the circumstances of pre-3.10 crashes are clearer.
	// For details see https://github.com/docker/docker/issues/407
	if k, err := kernel.GetKernelVersion(); err != nil {
		logrus.Warnf("%s", err)
	} else {
		if kernel.CompareKernelVersion(k, &kernel.KernelVersionInfo{Kernel: 3, Major: 10, Minor: 0}) < 0 {
			if os.Getenv("DOCKER_NOWARN_KERNEL_VERSION") == "" {
				logrus.Warnf("You are running linux kernel version %s, which might be unstable running docker. Please upgrade your kernel to 3.10.0.", k.String())
			}
		}
	}
	return nil
}

func (daemon *Daemon) verifyContainerSettings(hostConfig *runconfig.HostConfig, config *runconfig.Config) ([]string, error) {
	var warnings []string

	if config != nil {
		// The check for a valid workdir path is made on the server rather than in the
		// client. This is because we don't know the type of path (Linux or Windows)
		// to validate on the client.
		if config.WorkingDir != "" && !filepath.IsAbs(config.WorkingDir) {
			return warnings, fmt.Errorf("The working directory '%s' is invalid. It needs to be an absolute path.", config.WorkingDir)
		}
	}

	if hostConfig == nil {
		return warnings, nil
	}

	for port := range hostConfig.PortBindings {
		_, portStr := nat.SplitProtoPort(string(port))
		if _, err := nat.ParsePort(portStr); err != nil {
			return warnings, fmt.Errorf("Invalid port specification: %q", portStr)
		}
		for _, pb := range hostConfig.PortBindings[port] {
			_, err := nat.NewPort(nat.SplitProtoPort(pb.HostPort))
			if err != nil {
				return warnings, fmt.Errorf("Invalid port specification: %q", pb.HostPort)
			}
		}
	}
	if hostConfig.LxcConf.Len() > 0 && !strings.Contains(daemon.ExecutionDriver().Name(), "lxc") {
		return warnings, fmt.Errorf("Cannot use --lxc-conf with execdriver: %s", daemon.ExecutionDriver().Name())
	}
	if hostConfig.Memory != 0 && hostConfig.Memory < 4194304 {
		return warnings, fmt.Errorf("Minimum memory limit allowed is 4MB")
	}
	if hostConfig.Memory > 0 && !daemon.SystemConfig().MemoryLimit {
		warnings = append(warnings, "Your kernel does not support memory limit capabilities. Limitation discarded.")
		logrus.Warnf("Your kernel does not support memory limit capabilities. Limitation discarded.")
		hostConfig.Memory = 0
	}
	if hostConfig.Memory > 0 && hostConfig.MemorySwap != -1 && !daemon.SystemConfig().SwapLimit {
		warnings = append(warnings, "Your kernel does not support swap limit capabilities, memory limited without swap.")
		logrus.Warnf("Your kernel does not support swap limit capabilities, memory limited without swap.")
		hostConfig.MemorySwap = -1
	}
	if hostConfig.Memory > 0 && hostConfig.MemorySwap > 0 && hostConfig.MemorySwap < hostConfig.Memory {
		return warnings, fmt.Errorf("Minimum memoryswap limit should be larger than memory limit, see usage.")
	}
	if hostConfig.Memory == 0 && hostConfig.MemorySwap > 0 {
		return warnings, fmt.Errorf("You should always set the Memory limit when using Memoryswap limit, see usage.")
	}
	if hostConfig.MemorySwappiness != -1 && !daemon.SystemConfig().MemorySwappiness {
		warnings = append(warnings, "Your kernel does not support memory swappiness capabilities, memory swappiness discarded.")
		logrus.Warnf("Your kernel does not support memory swappiness capabilities, memory swappiness discarded.")
		hostConfig.MemorySwappiness = -1
	}
	if hostConfig.MemorySwappiness != -1 && (hostConfig.MemorySwappiness < 0 || hostConfig.MemorySwappiness > 100) {
		return warnings, fmt.Errorf("Invalid value: %d, valid memory swappiness range is 0-100.", hostConfig.MemorySwappiness)
	}
	if hostConfig.CpuPeriod > 0 && !daemon.SystemConfig().CpuCfsPeriod {
		warnings = append(warnings, "Your kernel does not support CPU cfs period. Period discarded.")
		logrus.Warnf("Your kernel does not support CPU cfs period. Period discarded.")
		hostConfig.CpuPeriod = 0
	}
	if hostConfig.CpuQuota > 0 && !daemon.SystemConfig().CpuCfsQuota {
		warnings = append(warnings, "Your kernel does not support CPU cfs quota. Quota discarded.")
		logrus.Warnf("Your kernel does not support CPU cfs quota. Quota discarded.")
		hostConfig.CpuQuota = 0
	}
	if hostConfig.BlkioWeight > 0 && (hostConfig.BlkioWeight < 10 || hostConfig.BlkioWeight > 1000) {
		return warnings, fmt.Errorf("Range of blkio weight is from 10 to 1000.")
	}
	if hostConfig.OomKillDisable && !daemon.SystemConfig().OomKillDisable {
		hostConfig.OomKillDisable = false
		return warnings, fmt.Errorf("Your kernel does not support oom kill disable.")
	}
	if daemon.SystemConfig().IPv4ForwardingDisabled {
		warnings = append(warnings, "IPv4 forwarding is disabled. Networking will not work.")
		logrus.Warnf("IPv4 forwarding is disabled. Networking will not work")
	}
	return warnings, nil
}

// checkConfigOptions checks for mutually incompatible config options
func checkConfigOptions(config *Config) error {
	// Check for mutually incompatible config options
	if config.Bridge.Iface != "" && config.Bridge.IP != "" {
		return fmt.Errorf("You specified -b & --bip, mutually exclusive options. Please specify only one.")
	}
	if !config.Bridge.EnableIPTables && !config.Bridge.InterContainerCommunication {
		return fmt.Errorf("You specified --iptables=false with --icc=false. ICC uses iptables to function. Please set --icc or --iptables to true.")
	}
	if !config.Bridge.EnableIPTables && config.Bridge.EnableIPMasq {
		config.Bridge.EnableIPMasq = false
	}
	return nil
}

// checkSystem validates platform-specific requirements
func checkSystem() error {
	if os.Geteuid() != 0 {
		return fmt.Errorf("The Docker daemon needs to be run as root")
	}
	if err := checkKernel(); err != nil {
		return err
	}
	return nil
}

// configureKernelSecuritySupport configures and validate security support for the kernel
func configureKernelSecuritySupport(config *Config, driverName string) error {
	if config.EnableSelinuxSupport {
		if selinuxEnabled() {
			// As Docker on btrfs and SELinux are incompatible at present, error on both being enabled
			if driverName == "btrfs" {
				return fmt.Errorf("SELinux is not supported with the BTRFS graph driver")
			}
			logrus.Debug("SELinux enabled successfully")
		} else {
			logrus.Warn("Docker could not enable SELinux on the host system")
		}
	} else {
		selinuxSetDisabled()
	}
	return nil
}

// MigrateIfDownlevel is a wrapper for AUFS migration for downlevel
func migrateIfDownlevel(driver graphdriver.Driver, root string) error {
	return migrateIfAufs(driver, root)
}

func configureVolumes(config *Config) error {
	volumesDriver, err := local.New(config.Root)
	if err != nil {
		return err
	}
	volumedrivers.Register(volumesDriver, volumesDriver.Name())
	return nil
}

func configureSysInit(config *Config) (string, error) {
	localCopy := filepath.Join(config.Root, "init", fmt.Sprintf("dockerinit-%s", dockerversion.VERSION))
	sysInitPath := utils.DockerInitPath(localCopy)
	if sysInitPath == "" {
		return "", fmt.Errorf("Could not locate dockerinit: This usually means docker was built incorrectly. See https://docs.docker.com/contributing/devenvironment for official build instructions.")
	}

	if sysInitPath != localCopy {
		// When we find a suitable dockerinit binary (even if it's our local binary), we copy it into config.Root at localCopy for future use (so that the original can go away without that being a problem, for example during a package upgrade).
		if err := os.Mkdir(filepath.Dir(localCopy), 0700); err != nil && !os.IsExist(err) {
			return "", err
		}
		if _, err := fileutils.CopyFile(sysInitPath, localCopy); err != nil {
			return "", err
		}
		if err := os.Chmod(localCopy, 0700); err != nil {
			return "", err
		}
		sysInitPath = localCopy
	}
	return sysInitPath, nil
}

func isBridgeNetworkDisabled(config *Config) bool {
	return config.Bridge.Iface == disableNetworkBridge
}

func networkOptions(dconfig *Config) ([]nwconfig.Option, error) {
	options := []nwconfig.Option{}
	if dconfig == nil {
		return options, nil
	}
	if strings.TrimSpace(dconfig.DefaultNetwork) != "" {
		dn := strings.Split(dconfig.DefaultNetwork, ":")
		if len(dn) < 2 {
			return nil, fmt.Errorf("default network daemon config must be of the form NETWORKDRIVER:NETWORKNAME")
		}
		options = append(options, nwconfig.OptionDefaultDriver(dn[0]))
		options = append(options, nwconfig.OptionDefaultNetwork(strings.Join(dn[1:], ":")))
	} else {
		dd := runconfig.DefaultDaemonNetworkMode()
		dn := runconfig.DefaultDaemonNetworkMode().NetworkName()
		options = append(options, nwconfig.OptionDefaultDriver(string(dd)))
		options = append(options, nwconfig.OptionDefaultNetwork(dn))
	}

	if strings.TrimSpace(dconfig.NetworkKVStore) != "" {
		kv := strings.Split(dconfig.NetworkKVStore, ":")
		if len(kv) < 2 {
			return nil, fmt.Errorf("kv store daemon config must be of the form KV-PROVIDER:KV-URL")
		}
		options = append(options, nwconfig.OptionKVProvider(kv[0]))
		options = append(options, nwconfig.OptionKVProviderURL(strings.Join(kv[1:], ":")))
	}

	options = append(options, nwconfig.OptionLabels(dconfig.Labels))
	return options, nil
}

func initNetworkController(config *Config) (libnetwork.NetworkController, error) {
	netOptions, err := networkOptions(config)
	if err != nil {
		return nil, err
	}

	controller, err := libnetwork.New(netOptions...)
	if err != nil {
		return nil, fmt.Errorf("error obtaining controller instance: %v", err)
	}

	// Initialize default driver "null"

	if err := controller.ConfigureNetworkDriver("null", options.Generic{}); err != nil {
		return nil, fmt.Errorf("Error initializing null driver: %v", err)
	}

	// Initialize default network on "null"
	if _, err := controller.NewNetwork("null", "none"); err != nil {
		return nil, fmt.Errorf("Error creating default \"null\" network: %v", err)
	}

	// Initialize default driver "host"
	if err := controller.ConfigureNetworkDriver("host", options.Generic{}); err != nil {
		return nil, fmt.Errorf("Error initializing host driver: %v", err)
	}

	// Initialize default network on "host"
	if _, err := controller.NewNetwork("host", "host"); err != nil {
		return nil, fmt.Errorf("Error creating default \"host\" network: %v", err)
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
	option := options.Generic{
		"EnableIPForwarding": config.Bridge.EnableIPForward}

	if err := controller.ConfigureNetworkDriver("bridge", options.Generic{netlabel.GenericData: option}); err != nil {
		return fmt.Errorf("Error initializing bridge driver: %v", err)
	}

	netOption := options.Generic{
		"BridgeName":          config.Bridge.Iface,
		"Mtu":                 config.Mtu,
		"EnableIPTables":      config.Bridge.EnableIPTables,
		"EnableIPMasquerade":  config.Bridge.EnableIPMasq,
		"EnableICC":           config.Bridge.InterContainerCommunication,
		"EnableUserlandProxy": config.Bridge.EnableUserlandProxy,
	}

	if config.Bridge.IP != "" {
		ip, bipNet, err := net.ParseCIDR(config.Bridge.IP)
		if err != nil {
			return err
		}

		bipNet.IP = ip
		netOption["AddressIPv4"] = bipNet
	}

	if config.Bridge.FixedCIDR != "" {
		_, fCIDR, err := net.ParseCIDR(config.Bridge.FixedCIDR)
		if err != nil {
			return err
		}

		netOption["FixedCIDR"] = fCIDR
	}

	if config.Bridge.FixedCIDRv6 != "" {
		_, fCIDRv6, err := net.ParseCIDR(config.Bridge.FixedCIDRv6)
		if err != nil {
			return err
		}

		netOption["FixedCIDRv6"] = fCIDRv6
	}

	if config.Bridge.DefaultGatewayIPv4 != nil {
		netOption["DefaultGatewayIPv4"] = config.Bridge.DefaultGatewayIPv4
	}

	if config.Bridge.DefaultGatewayIPv6 != nil {
		netOption["DefaultGatewayIPv6"] = config.Bridge.DefaultGatewayIPv6
	}

	// --ip processing
	if config.Bridge.DefaultIP != nil {
		netOption["DefaultBindingIP"] = config.Bridge.DefaultIP
	}

	// Initialize default network on "bridge" with the same name
	_, err := controller.NewNetwork("bridge", "bridge",
		libnetwork.NetworkOptionGeneric(options.Generic{
			netlabel.GenericData: netOption,
			netlabel.EnableIPv6:  config.Bridge.EnableIPv6,
		}))
	if err != nil {
		return fmt.Errorf("Error creating default \"bridge\" network: %v", err)
	}
	return nil
}

// setupInitLayer populates a directory with mountpoints suitable
// for bind-mounting dockerinit into the container. The mountpoint is simply an
// empty file at /.dockerinit
//
// This extra layer is used by all containers as the top-most ro layer. It protects
// the container from unwanted side-effects on the rw layer.
func setupInitLayer(initLayer string) error {
	for pth, typ := range map[string]string{
		"/dev/pts":         "dir",
		"/dev/shm":         "dir",
		"/proc":            "dir",
		"/sys":             "dir",
		"/.dockerinit":     "file",
		"/.dockerenv":      "file",
		"/etc/resolv.conf": "file",
		"/etc/hosts":       "file",
		"/etc/hostname":    "file",
		"/dev/console":     "file",
		"/etc/mtab":        "/proc/mounts",
	} {
		parts := strings.Split(pth, "/")
		prev := "/"
		for _, p := range parts[1:] {
			prev = filepath.Join(prev, p)
			syscall.Unlink(filepath.Join(initLayer, prev))
		}

		if _, err := os.Stat(filepath.Join(initLayer, pth)); err != nil {
			if os.IsNotExist(err) {
				if err := system.MkdirAll(filepath.Join(initLayer, filepath.Dir(pth)), 0755); err != nil {
					return err
				}
				switch typ {
				case "dir":
					if err := system.MkdirAll(filepath.Join(initLayer, pth), 0755); err != nil {
						return err
					}
				case "file":
					f, err := os.OpenFile(filepath.Join(initLayer, pth), os.O_CREATE, 0755)
					if err != nil {
						return err
					}
					f.Close()
				default:
					if err := os.Symlink(typ, filepath.Join(initLayer, pth)); err != nil {
						return err
					}
				}
			} else {
				return err
			}
		}
	}

	// Layer is ready to use, if it wasn't before.
	return nil
}

func (daemon *Daemon) NetworkApiRouter() func(w http.ResponseWriter, req *http.Request) {
	return nwapi.NewHTTPHandler(daemon.netController)
}

func (daemon *Daemon) RegisterLinks(container *Container, hostConfig *runconfig.HostConfig) error {
	if hostConfig == nil || hostConfig.Links == nil {
		return nil
	}

	for _, l := range hostConfig.Links {
		name, alias, err := parsers.ParseLink(l)
		if err != nil {
			return err
		}
		child, err := daemon.Get(name)
		if err != nil {
			//An error from daemon.Get() means this name could not be found
			return fmt.Errorf("Could not get container for %s", name)
		}
		for child.hostConfig.NetworkMode.IsContainer() {
			parts := strings.SplitN(string(child.hostConfig.NetworkMode), ":", 2)
			child, err = daemon.Get(parts[1])
			if err != nil {
				return fmt.Errorf("Could not get container for %s", parts[1])
			}
		}
		if child.hostConfig.NetworkMode.IsHost() {
			return runconfig.ErrConflictHostNetworkAndLinks
		}
		if err := daemon.RegisterLink(container, child, alias); err != nil {
			return err
		}
	}

	// After we load all the links into the daemon
	// set them to nil on the hostconfig
	hostConfig.Links = nil
	if err := container.WriteHostConfig(); err != nil {
		return err
	}

	return nil
}
