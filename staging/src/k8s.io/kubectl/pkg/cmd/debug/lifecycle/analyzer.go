/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package lifecycle

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// analyzer implements Analyzer
type analyzer struct{}

// NewAnalyzer creates a new diagnostic analyzer
func NewAnalyzer() Analyzer {
	return &analyzer{}
}

// Analyze performs comprehensive diagnostic analysis on the collected pod data
func (a *analyzer) Analyze(ctx context.Context, data *CollectedData) (*DiagnosticResult, error) {
	result := &DiagnosticResult{
		Pod:             data.Pod,
		ContainerIssues: []ContainerIssue{},
		VolumeIssues:    []VolumeIssue{},
		ResourceIssues:  []ResourceIssue{},
		ProbeIssues:     []ProbeIssue{},
		NetworkIssues:   []NetworkIssue{},
		Events:          []EventSummary{},
		Recommendations: []Recommendation{},
	}

	// 1. Determine the diagnostic phase
	result.Phase = a.determinePhase(data)

	// 2. Analyze based on phase
	switch result.Phase {
	case PhaseSchedulingFailed, PhasePending:
		a.analyzeSchedulingIssues(data, result)
	case PhaseImagePullFailed:
		a.analyzeImagePullIssues(data, result)
	case PhaseCrashLooping:
		a.analyzeCrashLoopIssues(data, result)
	case PhaseOOMKilled:
		a.analyzeOOMKilledIssues(data, result)
	case PhaseProbeFailed:
		a.analyzeProbeFailures(data, result)
	case PhaseContainerCreating:
		a.analyzeContainerCreatingIssues(data, result)
	case PhaseConfigError:
		a.analyzeConfigErrors(data, result)
	case PhaseInitializing:
		a.analyzeInitContainerIssues(data, result)
	case PhaseTerminated:
		a.analyzeTerminatedPod(data, result)
	case PhaseRunning:
		a.analyzeRunningPod(data, result)
	default:
		a.analyzeGenericIssues(data, result)
	}

	// 3. Analyze resource quotas and limits
	a.analyzeResourceConstraints(data, result)

	// 4. Analyze network policies
	a.analyzeNetworkPolicies(data, result)

	// 5. Extract relevant events
	result.Events = a.summarizeEvents(data.Events)

	// 6. Generate recommendations
	a.generateRecommendations(result)

	return result, nil
}

func (a *analyzer) determinePhase(data *CollectedData) DiagnosticPhase {
	pod := data.Pod

	// Check for OOMKilled first (highest priority) - check both current and last termination state
	for _, status := range pod.Status.ContainerStatuses {
		// Check current terminated state (for pods with restartPolicy: Never)
		if status.State.Terminated != nil {
			if status.State.Terminated.Reason == "OOMKilled" ||
				status.State.Terminated.ExitCode == 137 {
				return PhaseOOMKilled
			}
		}
		// Check last termination state (for pods that have restarted)
		if status.LastTerminationState.Terminated != nil {
			if status.LastTerminationState.Terminated.Reason == "OOMKilled" ||
				status.LastTerminationState.Terminated.ExitCode == 137 {
				return PhaseOOMKilled
			}
		}
	}

	// Check if pod is terminated (but not OOMKilled)
	if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
		return PhaseTerminated
	}

	// Check if pod is scheduled
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodScheduled {
			if condition.Status == corev1.ConditionFalse {
				if condition.Reason == "Unschedulable" {
					return PhaseSchedulingFailed
				}
			}
			break
		}
	}

	// Check for probe failures in events
	if a.hasProbeFailures(data.Events) {
		return PhaseProbeFailed
	}

	// Check init container states
	for _, status := range pod.Status.InitContainerStatuses {
		if status.State.Waiting != nil {
			switch status.State.Waiting.Reason {
			case "ImagePullBackOff", "ErrImagePull":
				return PhaseImagePullFailed
			case "CrashLoopBackOff":
				return PhaseCrashLooping
			case "CreateContainerConfigError":
				return PhaseConfigError
			}
			return PhaseInitializing
		}
		if status.State.Running != nil {
			return PhaseInitializing
		}
		// Check for OOMKilled in init containers
		if status.LastTerminationState.Terminated != nil {
			if status.LastTerminationState.Terminated.Reason == "OOMKilled" {
				return PhaseOOMKilled
			}
		}
	}

	// Check regular container states
	for _, status := range pod.Status.ContainerStatuses {
		if status.State.Waiting != nil {
			switch status.State.Waiting.Reason {
			case "ImagePullBackOff", "ErrImagePull":
				return PhaseImagePullFailed
			case "CrashLoopBackOff":
				return PhaseCrashLooping
			case "CreateContainerConfigError":
				return PhaseConfigError
			case "ContainerCreating":
				return PhaseContainerCreating
			case "PodInitializing":
				return PhaseInitializing
			}
		}
	}

	if pod.Status.Phase == corev1.PodPending {
		return PhasePending
	}
	if pod.Status.Phase == corev1.PodRunning {
		return PhaseRunning
	}

	return PhaseUnknown
}

func (a *analyzer) hasProbeFailures(events *corev1.EventList) bool {
	if events == nil {
		return false
	}
	for _, event := range events.Items {
		if event.Reason == "Unhealthy" {
			return true
		}
	}
	return false
}

func (a *analyzer) analyzeSchedulingIssues(data *CollectedData, result *DiagnosticResult) {
	result.SchedulingInfo = &SchedulingInfo{}

	// Parse FailedScheduling events for detailed reasons
	if data.Events != nil {
		for _, event := range data.Events.Items {
			if event.Reason == "FailedScheduling" {
				result.SchedulingInfo.Message = event.Message
				a.parseSchedulingMessage(event.Message, result.SchedulingInfo)

				result.RootCause = RootCause{
					Category:   "Scheduling",
					Summary:    "Pod cannot be scheduled to any node",
					Details:    event.Message,
					Confidence: 0.95,
				}
				break
			}
		}
	}

	// Check for specific scheduling gate
	if len(data.Pod.Spec.SchedulingGates) > 0 {
		gates := []string{}
		for _, gate := range data.Pod.Spec.SchedulingGates {
			gates = append(gates, gate.Name)
		}
		result.RootCause = RootCause{
			Category:   "Scheduling",
			Summary:    "Pod has scheduling gates preventing scheduling",
			Details:    "Scheduling gates: " + strings.Join(gates, ", "),
			Confidence: 1.0,
		}
	}
}

func (a *analyzer) parseSchedulingMessage(msg string, info *SchedulingInfo) {
	// Pattern: "0/X nodes are available: <reasons>"
	nodeCountPattern := regexp.MustCompile(`(\d+)/(\d+) nodes are available`)
	if matches := nodeCountPattern.FindStringSubmatch(msg); len(matches) == 3 {
		info.AvailableNodes = matches[1]
		info.TotalNodes = matches[2]
	}

	// Extract reasons histogram
	info.FailureReasons = []string{}

	// Pattern: "N reason" like "2 Insufficient cpu"
	reasonPattern := regexp.MustCompile(`(\d+) ([^,\.]+)`)
	for _, match := range reasonPattern.FindAllStringSubmatch(msg, -1) {
		if len(match) == 3 {
			info.FailureReasons = append(info.FailureReasons,
				match[1]+" "+strings.TrimSpace(match[2]))
		}
	}
}

func (a *analyzer) analyzeImagePullIssues(data *CollectedData, result *DiagnosticResult) {
	for _, status := range data.Pod.Status.ContainerStatuses {
		if status.State.Waiting != nil {
			reason := status.State.Waiting.Reason
			if reason == "ImagePullBackOff" || reason == "ErrImagePull" {
				issue := ContainerIssue{
					ContainerName: status.Name,
					State:         reason,
					Message:       status.State.Waiting.Message,
				}
				result.ContainerIssues = append(result.ContainerIssues, issue)

				// Determine specific image pull issue
				msg := status.State.Waiting.Message
				if strings.Contains(msg, "not found") || strings.Contains(msg, "manifest unknown") {
					result.RootCause = RootCause{
						Category:   "Image",
						Summary:    "Image or tag does not exist",
						Details:    msg,
						Confidence: 0.9,
					}
				} else if strings.Contains(msg, "unauthorized") || strings.Contains(msg, "denied") {
					result.RootCause = RootCause{
						Category:   "Image",
						Summary:    "Authentication failed - check imagePullSecrets",
						Details:    msg,
						Confidence: 0.85,
					}
				} else if strings.Contains(msg, "timeout") || strings.Contains(msg, "connection refused") {
					result.RootCause = RootCause{
						Category:   "Image",
						Summary:    "Registry unreachable - network or DNS issue",
						Details:    msg,
						Confidence: 0.8,
					}
				}
			}
		}
	}

	// Also check init containers
	for _, status := range data.Pod.Status.InitContainerStatuses {
		if status.State.Waiting != nil {
			reason := status.State.Waiting.Reason
			if reason == "ImagePullBackOff" || reason == "ErrImagePull" {
				issue := ContainerIssue{
					ContainerName: status.Name + " (init)",
					State:         reason,
					Message:       status.State.Waiting.Message,
				}
				result.ContainerIssues = append(result.ContainerIssues, issue)

				if result.RootCause.Category == "" {
					result.RootCause = RootCause{
						Category:   "Image",
						Summary:    "Init container image pull failed",
						Details:    status.State.Waiting.Message,
						Confidence: 0.85,
					}
				}
			}
		}
	}
}

func (a *analyzer) analyzeCrashLoopIssues(data *CollectedData, result *DiagnosticResult) {
	for _, status := range data.Pod.Status.ContainerStatuses {
		if status.State.Waiting != nil && status.State.Waiting.Reason == "CrashLoopBackOff" {
			issue := ContainerIssue{
				ContainerName: status.Name,
				State:         "CrashLoopBackOff",
				RestartCount:  status.RestartCount,
			}

			// Get termination details
			if status.LastTerminationState.Terminated != nil {
				term := status.LastTerminationState.Terminated
				issue.ExitCode = term.ExitCode
				issue.TerminationReason = term.Reason
				issue.TerminationMessage = term.Message
			}

			// Include previous logs if available
			if logs, ok := data.PreviousLogs[status.Name]; ok {
				issue.LastLogs = logs
			}

			result.ContainerIssues = append(result.ContainerIssues, issue)
		}
	}

	// Also check init containers
	for _, status := range data.Pod.Status.InitContainerStatuses {
		if status.State.Waiting != nil && status.State.Waiting.Reason == "CrashLoopBackOff" {
			issue := ContainerIssue{
				ContainerName: status.Name + " (init)",
				State:         "CrashLoopBackOff",
				RestartCount:  status.RestartCount,
			}

			if status.LastTerminationState.Terminated != nil {
				term := status.LastTerminationState.Terminated
				issue.ExitCode = term.ExitCode
				issue.TerminationReason = term.Reason
				issue.TerminationMessage = term.Message
			}

			result.ContainerIssues = append(result.ContainerIssues, issue)
		}
	}

	// Determine root cause based on exit code
	if len(result.ContainerIssues) > 0 {
		issue := result.ContainerIssues[0]
		switch issue.ExitCode {
		case 1:
			result.RootCause = RootCause{
				Category:   "Application",
				Summary:    "Application error (exit code 1)",
				Details:    "The container is exiting with code 1, indicating an application error. Check the container logs for details.",
				Confidence: 0.7,
			}
		case 137:
			result.RootCause = RootCause{
				Category:   "Resources",
				Summary:    "Container killed (likely OOMKilled, exit code 137)",
				Details:    "Exit code 137 typically indicates the container was killed by SIGKILL, often due to Out of Memory. Check memory limits.",
				Confidence: 0.8,
			}
		case 139:
			result.RootCause = RootCause{
				Category:   "Application",
				Summary:    "Segmentation fault (exit code 139)",
				Details:    "Exit code 139 indicates a segmentation fault in the application.",
				Confidence: 0.9,
			}
		case 143:
			result.RootCause = RootCause{
				Category:   "Lifecycle",
				Summary:    "Container terminated by SIGTERM (exit code 143)",
				Details:    "The container was gracefully terminated. This may indicate probe failures or external termination.",
				Confidence: 0.75,
			}
		default:
			result.RootCause = RootCause{
				Category:   "Application",
				Summary:    fmt.Sprintf("Container crashing with exit code %d", issue.ExitCode),
				Details:    "Check container logs for crash details.",
				Confidence: 0.6,
			}
		}
	}
}

func (a *analyzer) analyzeContainerCreatingIssues(data *CollectedData, result *DiagnosticResult) {
	// Check for volume issues
	for pvcName, pvc := range data.PVCs {
		if pvc == nil {
			result.VolumeIssues = append(result.VolumeIssues, VolumeIssue{
				VolumeName: pvcName,
				Type:       "PVCNotFound",
				Message:    "PersistentVolumeClaim not found",
			})
			result.RootCause = RootCause{
				Category:   "Volume",
				Summary:    "Referenced PVC does not exist",
				Details:    "PVC " + pvcName + " is referenced but not found in namespace",
				Confidence: 0.95,
			}
			continue
		}

		if pvc.Status.Phase == corev1.ClaimPending {
			issue := VolumeIssue{
				VolumeName: pvcName,
				Type:       "PVCPending",
				Message:    "PVC is pending, not bound to a PV",
			}

			// Check PVC events for details
			if events, ok := data.PVCEvents[pvcName]; ok {
				for _, event := range events.Items {
					if event.Type == "Warning" {
						issue.Message = event.Message
						break
					}
				}
			}

			result.VolumeIssues = append(result.VolumeIssues, issue)
			result.RootCause = RootCause{
				Category:   "Volume",
				Summary:    "PVC waiting for volume binding",
				Details:    issue.Message,
				Confidence: 0.9,
			}
		}
	}

	// Check for mount failures in events
	if data.Events != nil {
		for _, event := range data.Events.Items {
			if event.Reason == "FailedMount" || event.Reason == "FailedAttachVolume" {
				result.VolumeIssues = append(result.VolumeIssues, VolumeIssue{
					Type:    event.Reason,
					Message: event.Message,
				})
				result.RootCause = RootCause{
					Category:   "Volume",
					Summary:    "Volume mount/attach failed",
					Details:    event.Message,
					Confidence: 0.9,
				}
			}
		}
	}
}

func (a *analyzer) analyzeConfigErrors(data *CollectedData, result *DiagnosticResult) {
	// Check for missing ConfigMaps
	for name, cm := range data.ConfigMaps {
		if cm == nil {
			result.RootCause = RootCause{
				Category:   "Config",
				Summary:    "ConfigMap not found",
				Details:    "ConfigMap " + name + " is referenced but does not exist",
				Confidence: 0.95,
			}
			return
		}
	}

	// Check for missing Secrets
	for name, exists := range data.Secrets {
		if !exists {
			result.RootCause = RootCause{
				Category:   "Config",
				Summary:    "Secret not found",
				Details:    "Secret " + name + " is referenced but does not exist",
				Confidence: 0.95,
			}
			return
		}
	}

	// Check container error messages
	for _, status := range data.Pod.Status.ContainerStatuses {
		if status.State.Waiting != nil && status.State.Waiting.Reason == "CreateContainerConfigError" {
			result.RootCause = RootCause{
				Category:   "Config",
				Summary:    "Container configuration error",
				Details:    status.State.Waiting.Message,
				Confidence: 0.9,
			}
		}
	}
}

func (a *analyzer) analyzeInitContainerIssues(data *CollectedData, result *DiagnosticResult) {
	for _, status := range data.Pod.Status.InitContainerStatuses {
		if status.State.Running != nil {
			// Init container is running - might be slow or stuck
			result.RootCause = RootCause{
				Category:   "InitContainer",
				Summary:    "Init container '" + status.Name + "' is still running",
				Details:    "Pod is waiting for init containers to complete",
				Confidence: 0.8,
			}
			return
		}
		if status.State.Waiting != nil {
			result.ContainerIssues = append(result.ContainerIssues, ContainerIssue{
				ContainerName: status.Name + " (init)",
				State:         status.State.Waiting.Reason,
				Message:       status.State.Waiting.Message,
			})
		}
	}
}

func (a *analyzer) analyzeOOMKilledIssues(data *CollectedData, result *DiagnosticResult) {
	for _, status := range data.Pod.Status.ContainerStatuses {
		var term *corev1.ContainerStateTerminated

		// Check current state first (for restartPolicy: Never)
		if status.State.Terminated != nil &&
			(status.State.Terminated.Reason == "OOMKilled" || status.State.Terminated.ExitCode == 137) {
			term = status.State.Terminated
		} else if status.LastTerminationState.Terminated != nil &&
			(status.LastTerminationState.Terminated.Reason == "OOMKilled" || status.LastTerminationState.Terminated.ExitCode == 137) {
			term = status.LastTerminationState.Terminated
		}

		if term != nil {
			issue := ContainerIssue{
				ContainerName:     status.Name,
				State:             "OOMKilled",
				ExitCode:          term.ExitCode,
				RestartCount:      status.RestartCount,
				TerminationReason: term.Reason,
			}

			// Include previous logs if available
			if logs, ok := data.PreviousLogs[status.Name]; ok {
				issue.LastLogs = logs
			}

			result.ContainerIssues = append(result.ContainerIssues, issue)

			// Get resource limits for context
			for _, container := range data.Pod.Spec.Containers {
				if container.Name == status.Name {
					if memLimit := container.Resources.Limits.Memory(); memLimit != nil {
						result.ResourceIssues = append(result.ResourceIssues, ResourceIssue{
							ResourceType: "memory",
							Limit:        memLimit.String(),
							Message:      "Container was killed due to exceeding memory limit",
						})
					}
					break
				}
			}
		}
	}

	result.RootCause = RootCause{
		Category:   "Resources",
		Summary:    "Container killed due to Out of Memory (OOMKilled)",
		Details:    "The container exceeded its memory limit and was terminated by the kernel OOM killer. Increase memory limits or optimize application memory usage.",
		Confidence: 0.95,
	}
}

func (a *analyzer) analyzeProbeFailures(data *CollectedData, result *DiagnosticResult) {
	if data.Events == nil {
		return
	}

	// Parse Unhealthy events for probe details
	for _, event := range data.Events.Items {
		if event.Reason == "Unhealthy" {
			probeType := "unknown"
			if strings.Contains(event.Message, "Liveness") {
				probeType = "liveness"
			} else if strings.Contains(event.Message, "Readiness") {
				probeType = "readiness"
			} else if strings.Contains(event.Message, "Startup") {
				probeType = "startup"
			}

			result.ProbeIssues = append(result.ProbeIssues, ProbeIssue{
				ContainerName: "", // Event doesn't always specify container
				ProbeType:     probeType,
				FailureCount:  event.Count,
				Message:       event.Message,
			})
		}
	}

	if len(result.ProbeIssues) > 0 {
		probe := result.ProbeIssues[0]
		switch probe.ProbeType {
		case "liveness":
			result.RootCause = RootCause{
				Category:   "Probes",
				Summary:    "Liveness probe failing - container being restarted",
				Details:    probe.Message,
				Confidence: 0.9,
			}
		case "readiness":
			result.RootCause = RootCause{
				Category:   "Probes",
				Summary:    "Readiness probe failing - pod not receiving traffic",
				Details:    probe.Message,
				Confidence: 0.85,
			}
		case "startup":
			result.RootCause = RootCause{
				Category:   "Probes",
				Summary:    "Startup probe failing - container not starting properly",
				Details:    probe.Message,
				Confidence: 0.9,
			}
		default:
			result.RootCause = RootCause{
				Category:   "Probes",
				Summary:    "Health probe failing",
				Details:    probe.Message,
				Confidence: 0.8,
			}
		}
	}
}

func (a *analyzer) analyzeTerminatedPod(data *CollectedData, result *DiagnosticResult) {
	pod := data.Pod

	if pod.Status.Phase == corev1.PodSucceeded {
		result.RootCause = RootCause{
			Category:   "Lifecycle",
			Summary:    "Pod completed successfully",
			Details:    "All containers in the pod have terminated with exit code 0",
			Confidence: 1.0,
		}
		return
	}

	// Pod failed - find out why
	for _, status := range pod.Status.ContainerStatuses {
		if status.State.Terminated != nil {
			term := status.State.Terminated
			issue := ContainerIssue{
				ContainerName:      status.Name,
				State:              "Terminated",
				ExitCode:           term.ExitCode,
				TerminationReason:  term.Reason,
				TerminationMessage: term.Message,
			}

			if logs, ok := data.PreviousLogs[status.Name]; ok {
				issue.LastLogs = logs
			}

			result.ContainerIssues = append(result.ContainerIssues, issue)
		}
	}

	if len(result.ContainerIssues) > 0 {
		issue := result.ContainerIssues[0]
		result.RootCause = RootCause{
			Category:   "Application",
			Summary:    fmt.Sprintf("Container '%s' exited with code %d", issue.ContainerName, issue.ExitCode),
			Details:    issue.TerminationMessage,
			Confidence: 0.9,
		}
	} else {
		result.RootCause = RootCause{
			Category:   "Lifecycle",
			Summary:    "Pod failed",
			Details:    string(pod.Status.Phase),
			Confidence: 0.7,
		}
	}
}

func (a *analyzer) analyzeRunningPod(data *CollectedData, result *DiagnosticResult) {
	// Pod is running - check for any issues that might indicate problems
	pod := data.Pod

	// Check if all containers are ready
	allReady := true
	for _, status := range pod.Status.ContainerStatuses {
		if !status.Ready {
			allReady = false
			result.ContainerIssues = append(result.ContainerIssues, ContainerIssue{
				ContainerName: status.Name,
				State:         "NotReady",
				RestartCount:  status.RestartCount,
			})
		}
	}

	// Check for high restart counts
	for _, status := range pod.Status.ContainerStatuses {
		if status.RestartCount > 5 {
			result.ContainerIssues = append(result.ContainerIssues, ContainerIssue{
				ContainerName: status.Name,
				State:         "HighRestarts",
				RestartCount:  status.RestartCount,
				Message:       fmt.Sprintf("Container has restarted %d times", status.RestartCount),
			})
		}
	}

	if allReady && len(result.ContainerIssues) == 0 {
		result.RootCause = RootCause{
			Category:   "Healthy",
			Summary:    "Pod is running and all containers are ready",
			Details:    "No issues detected",
			Confidence: 1.0,
		}
	} else if !allReady {
		result.RootCause = RootCause{
			Category:   "Readiness",
			Summary:    "Pod is running but some containers are not ready",
			Details:    "Check readiness probes and container logs",
			Confidence: 0.8,
		}
	}
}

func (a *analyzer) analyzeResourceConstraints(data *CollectedData, result *DiagnosticResult) {
	// Check for ResourceQuota violations in events
	if data.Events != nil {
		for _, event := range data.Events.Items {
			if event.Reason == "FailedCreate" && strings.Contains(event.Message, "exceeded quota") {
				result.ResourceIssues = append(result.ResourceIssues, ResourceIssue{
					ResourceType: "quota",
					Message:      event.Message,
				})
				// Only override root cause if not already set
				if result.RootCause.Category == "" {
					result.RootCause = RootCause{
						Category:   "Resources",
						Summary:    "Resource quota exceeded",
						Details:    event.Message,
						Confidence: 0.95,
					}
				}
			}
		}
	}

	// Check LimitRange constraints
	if len(data.LimitRanges) > 0 {
		for _, container := range data.Pod.Spec.Containers {
			for _, lr := range data.LimitRanges {
				for _, item := range lr.Spec.Limits {
					if item.Type == corev1.LimitTypeContainer {
						// Check if requests are below minimum
						if item.Min != nil {
							if minCPU := item.Min.Cpu(); minCPU != nil && !minCPU.IsZero() {
								if reqCPU := container.Resources.Requests.Cpu(); reqCPU != nil && reqCPU.Cmp(*minCPU) < 0 {
									result.ResourceIssues = append(result.ResourceIssues, ResourceIssue{
										ResourceType: "cpu",
										Requested:    reqCPU.String(),
										Available:    "min: " + minCPU.String(),
										Message:      fmt.Sprintf("Container %s CPU request below LimitRange minimum", container.Name),
									})
								}
							}
						}
					}
				}
			}
		}
	}
}

func (a *analyzer) analyzeNetworkPolicies(data *CollectedData, result *DiagnosticResult) {
	if len(data.NetworkPolicies) > 0 {
		result.NetworkIssues = append(result.NetworkIssues, NetworkIssue{
			Type:    "NetworkPolicy",
			Message: fmt.Sprintf("Pod is affected by %d network policy(ies): %s", len(data.NetworkPolicies), strings.Join(data.NetworkPolicies, ", ")),
		})
	}

	// Check ServiceAccount issues
	if data.ServiceAccount == nil && data.Pod.Spec.ServiceAccountName != "" && data.Pod.Spec.ServiceAccountName != "default" {
		result.NetworkIssues = append(result.NetworkIssues, NetworkIssue{
			Type:    "ServiceAccount",
			Message: fmt.Sprintf("ServiceAccount '%s' not found", data.Pod.Spec.ServiceAccountName),
		})
	}
}

func (a *analyzer) analyzeGenericIssues(data *CollectedData, result *DiagnosticResult) {
	// Fallback analysis - extract any warning events
	if data.Events != nil {
		for _, event := range data.Events.Items {
			if event.Type == "Warning" {
				result.RootCause = RootCause{
					Category:   "Unknown",
					Summary:    event.Reason + ": " + truncate(event.Message, 80),
					Details:    event.Message,
					Confidence: 0.5,
				}
				return
			}
		}
	}

	result.RootCause = RootCause{
		Category:   "Unknown",
		Summary:    "Unable to determine specific issue",
		Details:    "Pod phase: " + string(data.Pod.Status.Phase),
		Confidence: 0.3,
	}
}

func (a *analyzer) summarizeEvents(events *corev1.EventList) []EventSummary {
	summaries := []EventSummary{}
	if events == nil {
		return summaries
	}

	for _, e := range events.Items {
		summaries = append(summaries, EventSummary{
			Type:    e.Type,
			Reason:  e.Reason,
			Message: e.Message,
			Count:   e.Count,
			Age:     e.LastTimestamp.Time,
			Source:  e.Source.Component,
		})
	}
	return summaries
}

func (a *analyzer) generateRecommendations(result *DiagnosticResult) {
	priority := 1

	switch result.Phase {
	case PhaseSchedulingFailed:
		if result.SchedulingInfo != nil {
			for _, reason := range result.SchedulingInfo.FailureReasons {
				if strings.Contains(reason, "Insufficient cpu") || strings.Contains(reason, "Insufficient memory") {
					result.Recommendations = append(result.Recommendations, Recommendation{
						Priority:    priority,
						Action:      "Reduce resource requests or add more nodes",
						Command:     "kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory",
						Explanation: "The cluster does not have enough resources. Check node capacity vs pod requests.",
					})
					priority++
				}
				if strings.Contains(reason, "untolerated taint") {
					result.Recommendations = append(result.Recommendations, Recommendation{
						Priority:    priority,
						Action:      "Add tolerations or remove node taints",
						Command:     "kubectl describe nodes | grep -A5 Taints",
						Explanation: "Nodes have taints that the pod doesn't tolerate.",
					})
					priority++
				}
				if strings.Contains(reason, "node selector") || strings.Contains(reason, "node affinity") {
					result.Recommendations = append(result.Recommendations, Recommendation{
						Priority:    priority,
						Action:      "Check nodeSelector/nodeAffinity matches node labels",
						Command:     "kubectl get nodes --show-labels",
						Explanation: "Pod's node selection constraints don't match any available nodes.",
					})
					priority++
				}
			}
		}

	case PhaseImagePullFailed:
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Verify image name and tag exist",
			Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o jsonpath='{.spec.containers[*].image}'",
			Explanation: "Ensure the image reference is correct and the tag exists in the registry.",
		})
		priority++
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Check imagePullSecrets configuration",
			Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o jsonpath='{.spec.imagePullSecrets}'",
			Explanation: "If using a private registry, ensure imagePullSecrets are configured correctly.",
		})

	case PhaseCrashLooping:
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Check container logs",
			Command:     "kubectl logs " + result.Pod.Name + " -n " + result.Pod.Namespace + " --previous",
			Explanation: "Container logs from the previous crash can reveal the error.",
		})
		priority++

		// Check for OOMKilled
		for _, issue := range result.ContainerIssues {
			if issue.ExitCode == 137 {
				result.Recommendations = append(result.Recommendations, Recommendation{
					Priority:    priority,
					Action:      "Increase memory limits",
					Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o jsonpath='{.spec.containers[*].resources}'",
					Explanation: "Exit code 137 often indicates OOMKilled. Consider increasing memory limits.",
				})
				break
			}
		}

	case PhaseOOMKilled:
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Increase container memory limits",
			Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o jsonpath='{.spec.containers[*].resources}'",
			Explanation: "Container was killed for exceeding memory limits. Increase limits or optimize memory usage.",
		})
		priority++
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Check application memory usage patterns",
			Command:     "kubectl logs " + result.Pod.Name + " -n " + result.Pod.Namespace + " --previous",
			Explanation: "Review logs for memory leaks or excessive allocations before the crash.",
		})
		priority++
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Profile application memory",
			Command:     "kubectl top pod " + result.Pod.Name + " -n " + result.Pod.Namespace,
			Explanation: "Monitor real-time memory consumption to understand usage patterns.",
		})

	case PhaseProbeFailed:
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Check probe configuration",
			Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o jsonpath='{.spec.containers[*].livenessProbe}'",
			Explanation: "Verify probe endpoints, timeouts, and thresholds are appropriate for your application.",
		})
		priority++
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Check application health endpoint",
			Command:     "kubectl exec " + result.Pod.Name + " -n " + result.Pod.Namespace + " -- curl -v localhost:<port>/<path>",
			Explanation: "Manually test the health endpoint from inside the container.",
		})
		priority++
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Increase probe timeouts/thresholds",
			Explanation: "If the application is slow to start or respond, consider increasing initialDelaySeconds, timeoutSeconds, or failureThreshold.",
		})

	case PhaseContainerCreating:
		if len(result.VolumeIssues) > 0 {
			result.Recommendations = append(result.Recommendations, Recommendation{
				Priority:    priority,
				Action:      "Check PVC status",
				Command:     "kubectl get pvc -n " + result.Pod.Namespace,
				Explanation: "Volumes may be pending or failing to attach.",
			})
			priority++
			result.Recommendations = append(result.Recommendations, Recommendation{
				Priority:    priority,
				Action:      "Check storage class provisioner",
				Command:     "kubectl get sc",
				Explanation: "Ensure the storage class exists and the provisioner is working.",
			})
		}

	case PhaseConfigError:
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Verify referenced ConfigMaps and Secrets exist",
			Command:     "kubectl get configmaps,secrets -n " + result.Pod.Namespace,
			Explanation: "Container configuration errors often indicate missing ConfigMaps or Secrets.",
		})
		priority++
		result.Recommendations = append(result.Recommendations, Recommendation{
			Priority:    priority,
			Action:      "Check for key mismatches",
			Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o yaml | grep -A5 configMapKeyRef",
			Explanation: "Ensure all referenced keys exist in the ConfigMaps/Secrets.",
		})

	case PhaseTerminated:
		if result.RootCause.Category != "Lifecycle" {
			result.Recommendations = append(result.Recommendations, Recommendation{
				Priority:    priority,
				Action:      "Check container logs for errors",
				Command:     "kubectl logs " + result.Pod.Name + " -n " + result.Pod.Namespace,
				Explanation: "Review logs to understand why the container terminated.",
			})
		}

	case PhaseRunning:
		if result.RootCause.Category == "Readiness" {
			result.Recommendations = append(result.Recommendations, Recommendation{
				Priority:    priority,
				Action:      "Check readiness probe configuration",
				Command:     "kubectl get pod " + result.Pod.Name + " -n " + result.Pod.Namespace + " -o jsonpath='{.spec.containers[*].readinessProbe}'",
				Explanation: "Container is running but not ready. Check readiness probe settings.",
			})
		}
	}

	// Add recommendations for resource issues
	if len(result.ResourceIssues) > 0 {
		for _, issue := range result.ResourceIssues {
			if issue.ResourceType == "quota" {
				result.Recommendations = append(result.Recommendations, Recommendation{
					Priority:    priority,
					Action:      "Check namespace resource quotas",
					Command:     "kubectl get resourcequota -n " + result.Pod.Namespace,
					Explanation: "Resource quota exceeded. Request quota increase or reduce resource requests.",
				})
				priority++
				break
			}
		}
	}

	// Add recommendations for network issues
	if len(result.NetworkIssues) > 0 {
		for _, issue := range result.NetworkIssues {
			if issue.Type == "NetworkPolicy" {
				result.Recommendations = append(result.Recommendations, Recommendation{
					Priority:    priority,
					Action:      "Review network policies affecting this pod",
					Command:     "kubectl get networkpolicy -n " + result.Pod.Namespace,
					Explanation: "Network policies may be blocking traffic to/from this pod.",
				})
				priority++
				break
			}
			if issue.Type == "ServiceAccount" {
				result.Recommendations = append(result.Recommendations, Recommendation{
					Priority:    priority,
					Action:      "Create missing ServiceAccount",
					Command:     "kubectl create serviceaccount " + result.Pod.Spec.ServiceAccountName + " -n " + result.Pod.Namespace,
					Explanation: "Referenced ServiceAccount does not exist.",
				})
				priority++
			}
		}
	}

	// Always recommend describe for more details
	result.Recommendations = append(result.Recommendations, Recommendation{
		Priority:    10, // Low priority catch-all
		Action:      "Get full pod details",
		Command:     "kubectl describe pod " + result.Pod.Name + " -n " + result.Pod.Namespace,
		Explanation: "kubectl describe provides comprehensive pod information.",
	})
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
