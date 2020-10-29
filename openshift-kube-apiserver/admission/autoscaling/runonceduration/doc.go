/*
Package runonceduration contains the RunOnceDuration admission control plugin.
The plugin allows overriding the ActiveDeadlineSeconds for pods that have a
RestartPolicy of RestartPolicyNever (run once). If configured to allow a project
annotation override, and an annotation exists in the pod's namespace of:

	openshift.io/active-deadline-seconds-override

the value of the annotation will take precedence over the globally configured
value in the plugin's configuration.

# Configuration

The plugin is configured via a RunOnceDurationConfig object:

	apiVersion: v1
	kind: RunOnceDurationConfig
	enabled: true
	activeDeadlineSecondsOverride: 3600
*/
package runonceduration
