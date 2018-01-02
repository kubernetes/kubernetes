package dockerfile

import (
	"github.com/docker/go-metrics"
)

var (
	buildsTriggered metrics.Counter
	buildsFailed    metrics.LabeledCounter
)

// Build metrics prometheus messages, these values must be initialized before
// using them. See the example below in the "builds_failed" metric definition.
const (
	metricsDockerfileSyntaxError        = "dockerfile_syntax_error"
	metricsDockerfileEmptyError         = "dockerfile_empty_error"
	metricsCommandNotSupportedError     = "command_not_supported_error"
	metricsErrorProcessingCommandsError = "error_processing_commands_error"
	metricsBuildTargetNotReachableError = "build_target_not_reachable_error"
	metricsMissingOnbuildArgumentsError = "missing_onbuild_arguments_error"
	metricsUnknownInstructionError      = "unknown_instruction_error"
	metricsBuildCanceled                = "build_canceled"
)

func init() {
	buildMetrics := metrics.NewNamespace("builder", "", nil)

	buildsTriggered = buildMetrics.NewCounter("builds_triggered", "Number of triggered image builds")
	buildsFailed = buildMetrics.NewLabeledCounter("builds_failed", "Number of failed image builds", "reason")
	for _, r := range []string{
		metricsDockerfileSyntaxError,
		metricsDockerfileEmptyError,
		metricsCommandNotSupportedError,
		metricsErrorProcessingCommandsError,
		metricsBuildTargetNotReachableError,
		metricsMissingOnbuildArgumentsError,
		metricsUnknownInstructionError,
		metricsBuildCanceled,
	} {
		buildsFailed.WithValues(r)
	}

	metrics.Register(buildMetrics)
}
