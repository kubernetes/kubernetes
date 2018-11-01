package policy

// log policy event reason
const (
	LogPolicyConfigUpdateFailedReason = "LogPolicyConfigUpdateFailed"
	LogPolicyCreateSuccess            = "LogPolicyCreateSuccess"
	LogPolicyRemoveSuccess            = "LogPolicyRemoveSuccess"
)

const (
	PodLogPolicyLabelKey = "alpha.log.qiniu.com/log-policy"
)

type PodLogPolicy struct {
	// log plugin name, eg. logkit, logexporter
	LogPlugin string `json:"log_plugin"`
	// ensure all log been collected before pod are terminated
	// SafeDeletionEnabled == true, pod will keep terminating forever util log plugin says all log are collected.
	// SafeDeletionEnabled == false, pod will terminated before TerminationGracePeriodSeconds in PodSpec.
	SafeDeletionEnabled bool `json:"safe_deletion_enabled"`
	// container name -> ContainerLogPolicies
	ContainerLogPolicies map[string]ContainerLogPolicies `json:"container_log_policies"`
}

type ContainerLogPolicies []*ContainerLogPolicy

type ContainerLogPolicy struct {
	// log category, eg. std(stdout/stderr), app, audit
	// if category is "std", path and volume_name will make no sense
	Category string `json:"category"`
	// log volume mount path
	Path string `json:"path"`
	// volume(mount) name
	// volume for container log
	VolumeName string `json:"volume_name"`
	// configmap name of log plugin configs
	PluginConfigMap string `json:"plugin_configmap"`
}
