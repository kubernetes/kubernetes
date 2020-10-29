package runonceduration

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RunOnceDurationConfig is the configuration for the RunOnceDuration plugin.
// It specifies a maximum value for ActiveDeadlineSeconds for a run-once pod.
// The project that contains the pod may specify a different setting. That setting will
// take precedence over the one configured for the plugin here.
type RunOnceDurationConfig struct {
	metav1.TypeMeta

	// ActiveDeadlineSecondsOverride is the maximum value to set on containers of run-once pods
	// Only a positive value is valid. Absence of a value means that the plugin
	// won't make any changes to the pod
	ActiveDeadlineSecondsOverride *int64
}

// ActiveDeadlineSecondsLimitAnnotation can be set on a project to limit the number of
// seconds that a run-once pod can be active in that project
// TODO: this label needs to change to reflect its function. It's a limit, not an override.
// It is kept this way for compatibility. Only change it in a new version of the API.
const ActiveDeadlineSecondsLimitAnnotation = "openshift.io/active-deadline-seconds-override"
