package v1alpha1

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
	// log category name, eg. std(stdout/stderr), app, audit
	Category string `json:"category"`
	// log volume mount path
	// if path is "-", that means this policy is dedicated for std(stdout/stderr) logs and VolumeName will make no sense.
	Path string `json:"path"`
	// volume(mount) name
	// volume for container file log
	VolumeName string `json:"volume_name"`
	// configmap name of log plugin configs
	PluginConfigMap string `json:"plugin_configmap"`
}
