package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RunOnceDurationConfig is the configuration for the RunOnceDuration plugin.
// It specifies a maximum value for ActiveDeadlineSeconds for a run-once pod.
// The project that contains the pod may specify a different setting. That setting will
// take precedence over the one configured for the plugin here.
type RunOnceDurationConfig struct {
	metav1.TypeMeta `json:",inline"`

	// ActiveDeadlineSecondsOverride is the maximum value to set on containers of run-once pods
	// Only a positive value is valid. Absence of a value means that the plugin
	// won't make any changes to the pod
	// TODO: change the external name of this field to reflect that it is a limit, not an override
	// It is kept this way for compatibility. Only change it in a new version of the API.
	ActiveDeadlineSecondsOverride *int64 `json:"activeDeadlineSecondsOverride,omitempty" description:"maximum value for activeDeadlineSeconds in run-once pods"`
}
