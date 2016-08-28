package types

import "github.com/akutz/gofig"
import log "github.com/Sirupsen/logrus"

const (
	//ConfigOldRoot is a config key.
	ConfigOldRoot = "volume"

	// ConfigOldIntegrationVolMountPreempt is a config key.
	ConfigOldIntegrationVolMountPreempt = ConfigOldRoot + ".mount.preempt"

	// ConfigOldIntegrationVolCreateDisable is a config key.
	ConfigOldIntegrationVolCreateDisable = ConfigOldRoot + ".create.disable"

	// ConfigOldIntegrationVolRemoveDisable is a config key.
	ConfigOldIntegrationVolRemoveDisable = ConfigOldRoot + ".remove.disable"

	// ConfigOldIntegrationVolUnmountIgnoreUsed is a config key.
	ConfigOldIntegrationVolUnmountIgnoreUsed = ConfigOldRoot + ".unmount.ignoreusedcount"

	// ConfigOldIntegrationVolPathCache is a config key.
	ConfigOldIntegrationVolPathCache = ConfigOldRoot + ".path.cache"

	//ConfigOldDocker is a config key.
	ConfigOldDocker = "docker"

	//ConfigOldDockerFsType is a config key.
	ConfigOldDockerFsType = ConfigOldDocker + ".fsType"

	//ConfigOldDockerVolumeType is a  config key.
	ConfigOldDockerVolumeType = ConfigOldDocker + ".volumeType"

	//ConfigOldDockerIOPS is a config key.
	ConfigOldDockerIOPS = ConfigOldDocker + ".iops"

	//ConfigOldDockerSize is a config key.
	ConfigOldDockerSize = ConfigOldDocker + ".size"

	//ConfigOldDockerAvailabilityZone is a config key.
	ConfigOldDockerAvailabilityZone = ConfigOldDocker + ".availabilityZone"

	//ConfigOldDockerMountDirPath is a config key.
	ConfigOldDockerMountDirPath = ConfigOldDocker + ".mountDirPath"

	//ConfigOldDockerLinuxVolumeRootPath is a config key.
	ConfigOldDockerLinuxVolumeRootPath = "linux.volume.rootpath"
)

// BackCompat ensures keys can be used from old configurations.
func BackCompat(config gofig.Config) {
	checks := [][]string{
		{ConfigIgVolOpsMountPreempt, ConfigOldIntegrationVolMountPreempt},
		{ConfigIgVolOpsCreateDisable, ConfigOldIntegrationVolCreateDisable},
		{ConfigIgVolOpsRemoveDisable, ConfigOldIntegrationVolRemoveDisable},
		{ConfigIgVolOpsUnmountIgnoreUsed, ConfigOldIntegrationVolUnmountIgnoreUsed},
		{ConfigIgVolOpsPathCache, ConfigOldIntegrationVolPathCache},
		{ConfigIgVolOpsCreateDefaultFsType, ConfigOldDockerFsType},
		{ConfigIgVolOpsCreateDefaultType, ConfigOldDockerVolumeType},
		{ConfigIgVolOpsCreateDefaultIOPS, ConfigOldDockerIOPS},
		{ConfigIgVolOpsCreateDefaultSize, ConfigOldDockerSize},
		{ConfigIgVolOpsCreateDefaultAZ, ConfigOldDockerAvailabilityZone},
		{ConfigIgVolOpsMountPath, ConfigOldDockerMountDirPath},
		{ConfigIgVolOpsMountRootPath, ConfigOldDockerLinuxVolumeRootPath},
	}
	for _, check := range checks {
		if !config.IsSet(check[0]) && config.IsSet(check[1]) {
			log.Debug(config.Get(check[1]))
			config.Set(check[0], config.Get(check[1]))
		}
	}
}
