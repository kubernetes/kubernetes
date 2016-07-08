package types

const (
	//ConfigIg is a config key.
	ConfigIg = ConfigRoot + ".integration"

	//ConfigIgVol is a config key.
	ConfigIgVol = ConfigIg + ".volume"

	//ConfigIgVolOps is a config key.
	ConfigIgVolOps = ConfigIgVol + ".operations"

	//ConfigIgVolOpsMount is a config key.
	ConfigIgVolOpsMount = ConfigIgVolOps + ".mount"

	//ConfigIgVolOpsMountPreempt is a config key.
	ConfigIgVolOpsMountPreempt = ConfigIgVolOpsMount + ".preempt"

	//ConfigIgVolOpsMountPath is a config key.
	ConfigIgVolOpsMountPath = ConfigIgVolOpsMount + ".path"

	//ConfigIgVolOpsMountRootPath is a config key.
	ConfigIgVolOpsMountRootPath = ConfigIgVolOpsMount + ".rootPath"

	//ConfigIgVolOpsUnmount is a config key.
	ConfigIgVolOpsUnmount = ConfigIgVolOps + ".unmount"

	//ConfigIgVolOpsUnmountIgnoreUsed is a config key.
	ConfigIgVolOpsUnmountIgnoreUsed = ConfigIgVolOpsUnmount + ".ignoreusedcount"

	// ConfigIgVolOpsPath is a config key.
	ConfigIgVolOpsPath = ConfigIgVolOps + ".path"

	// ConfigIgVolOpsPathCache is a config key.
	ConfigIgVolOpsPathCache = ConfigIgVolOpsPath + ".cache"

	// ConfigIgVolOpsCreate is a config key.
	ConfigIgVolOpsCreate = ConfigIgVolOps + ".create"

	// ConfigIgVolOpsCreateDisable is a config key.
	ConfigIgVolOpsCreateDisable = ConfigIgVolOpsCreate + ".disable"

	// ConfigIgVolOpsCreateImplicit is a config key.
	ConfigIgVolOpsCreateImplicit = ConfigIgVolOpsCreate + ".implicit"

	// ConfigIgVolOpsCreateDefault is a config key.
	ConfigIgVolOpsCreateDefault = ConfigIgVolOpsCreate + ".default"

	// ConfigIgVolOpsCreateDefaultSize is a config key.
	ConfigIgVolOpsCreateDefaultSize = ConfigIgVolOpsCreateDefault + ".size"

	// ConfigIgVolOpsCreateDefaultFsType is a config key.
	ConfigIgVolOpsCreateDefaultFsType = ConfigIgVolOpsCreateDefault + ".fsType"

	// ConfigIgVolOpsCreateDefaultAZ is a config key.
	ConfigIgVolOpsCreateDefaultAZ = ConfigIgVolOpsCreateDefault + ".availabilityZone"

	// ConfigIgVolOpsCreateDefaultType is a config key.
	ConfigIgVolOpsCreateDefaultType = ConfigIgVolOpsCreateDefault + ".type"

	// ConfigIgVolOpsCreateDefaultIOPS is a config key.
	ConfigIgVolOpsCreateDefaultIOPS = ConfigIgVolOpsCreateDefault + ".IOPS"

	// ConfigIgVolOpsRemove is a config key.
	ConfigIgVolOpsRemove = ConfigIgVolOps + ".remove"

	// ConfigIgVolOpsRemoveDisable is a config key.
	ConfigIgVolOpsRemoveDisable = ConfigIgVolOpsRemove + ".disable"
)
