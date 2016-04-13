package daemon

import (
	"os"
	"runtime"
	"time"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/autogen/dockerversion"
	"github.com/docker/docker/pkg/fileutils"
	"github.com/docker/docker/pkg/parsers/kernel"
	"github.com/docker/docker/pkg/parsers/operatingsystem"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/docker/registry"
	"github.com/docker/docker/utils"
)

func (daemon *Daemon) SystemInfo() (*types.Info, error) {
	images := daemon.Graph().Map()
	var imgcount int
	if images == nil {
		imgcount = 0
	} else {
		imgcount = len(images)
	}
	kernelVersion := "<unknown>"
	if kv, err := kernel.GetKernelVersion(); err == nil {
		kernelVersion = kv.String()
	}

	operatingSystem := "<unknown>"
	if s, err := operatingsystem.GetOperatingSystem(); err == nil {
		operatingSystem = s
	}

	// Don't do containerized check on Windows
	if runtime.GOOS != "windows" {
		if inContainer, err := operatingsystem.IsContainerized(); err != nil {
			logrus.Errorf("Could not determine if daemon is containerized: %v", err)
			operatingSystem += " (error determining if containerized)"
		} else if inContainer {
			operatingSystem += " (containerized)"
		}
	}

	meminfo, err := system.ReadMemInfo()
	if err != nil {
		logrus.Errorf("Could not read system memory info: %v", err)
	}

	// if we still have the original dockerinit binary from before we copied it locally, let's return the path to that, since that's more intuitive (the copied path is trivial to derive by hand given VERSION)
	initPath := utils.DockerInitPath("")
	if initPath == "" {
		// if that fails, we'll just return the path from the daemon
		initPath = daemon.SystemInitPath()
	}

	v := &types.Info{
		ID:                 daemon.ID,
		Containers:         len(daemon.List()),
		Images:             imgcount,
		Driver:             daemon.GraphDriver().String(),
		DriverStatus:       daemon.GraphDriver().Status(),
		IPv4Forwarding:     !daemon.SystemConfig().IPv4ForwardingDisabled,
		BridgeNfIptables:   !daemon.SystemConfig().BridgeNfCallIptablesDisabled,
		BridgeNfIp6tables:  !daemon.SystemConfig().BridgeNfCallIp6tablesDisabled,
		Debug:              os.Getenv("DEBUG") != "",
		NFd:                fileutils.GetTotalUsedFds(),
		NGoroutines:        runtime.NumGoroutine(),
		SystemTime:         time.Now().Format(time.RFC3339Nano),
		ExecutionDriver:    daemon.ExecutionDriver().Name(),
		LoggingDriver:      daemon.defaultLogConfig.Type,
		NEventsListener:    daemon.EventsService.SubscribersCount(),
		KernelVersion:      kernelVersion,
		OperatingSystem:    operatingSystem,
		IndexServerAddress: registry.INDEXSERVER,
		RegistryConfig:     daemon.RegistryService.Config,
		InitSha1:           dockerversion.INITSHA1,
		InitPath:           initPath,
		NCPU:               runtime.NumCPU(),
		MemTotal:           meminfo.MemTotal,
		DockerRootDir:      daemon.Config().Root,
		Labels:             daemon.Config().Labels,
		ExperimentalBuild:  utils.ExperimentalBuild(),
	}

	// TODO Windows. Refactor this more once sysinfo is refactored into
	// platform specific code. On Windows, sysinfo.cgroupMemInfo and
	// sysinfo.cgroupCpuInfo will be nil otherwise and cause a SIGSEGV if
	// an attempt is made to access through them.
	if runtime.GOOS != "windows" {
		v.MemoryLimit = daemon.SystemConfig().MemoryLimit
		v.SwapLimit = daemon.SystemConfig().SwapLimit
		v.OomKillDisable = daemon.SystemConfig().OomKillDisable
		v.CpuCfsPeriod = daemon.SystemConfig().CpuCfsPeriod
		v.CpuCfsQuota = daemon.SystemConfig().CpuCfsQuota
	}

	if httpProxy := os.Getenv("http_proxy"); httpProxy != "" {
		v.HttpProxy = httpProxy
	}
	if httpsProxy := os.Getenv("https_proxy"); httpsProxy != "" {
		v.HttpsProxy = httpsProxy
	}
	if noProxy := os.Getenv("no_proxy"); noProxy != "" {
		v.NoProxy = noProxy
	}
	if hostname, err := os.Hostname(); err == nil {
		v.Name = hostname
	}

	return v, nil
}
