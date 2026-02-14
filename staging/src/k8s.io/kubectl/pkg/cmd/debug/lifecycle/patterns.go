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
	"time"

	corev1 "k8s.io/api/core/v1"
)

// DiagnosticPhase represents the diagnosed lifecycle phase
type DiagnosticPhase string

const (
	PhasePending           DiagnosticPhase = "Pending"
	PhaseSchedulingFailed  DiagnosticPhase = "SchedulingFailed"
	PhaseImagePullFailed   DiagnosticPhase = "ImagePullFailed"
	PhaseCrashLooping      DiagnosticPhase = "CrashLooping"
	PhaseContainerCreating DiagnosticPhase = "ContainerCreating"
	PhaseInitializing      DiagnosticPhase = "Initializing"
	PhaseConfigError       DiagnosticPhase = "ConfigError"
	PhaseOOMKilled         DiagnosticPhase = "OOMKilled"
	PhaseProbeFailed       DiagnosticPhase = "ProbeFailed"
	PhaseRunning           DiagnosticPhase = "Running"
	PhaseTerminated        DiagnosticPhase = "Terminated"
	PhaseUnknown           DiagnosticPhase = "Unknown"
)

// DiagnosticResult represents the complete diagnostic output
type DiagnosticResult struct {
	Pod             *corev1.Pod      `json:"pod,omitempty"`
	Phase           DiagnosticPhase  `json:"phase"`
	RootCause       RootCause        `json:"rootCause"`
	ContainerIssues []ContainerIssue `json:"containerIssues,omitempty"`
	SchedulingInfo  *SchedulingInfo  `json:"schedulingInfo,omitempty"`
	VolumeIssues    []VolumeIssue    `json:"volumeIssues,omitempty"`
	ResourceIssues  []ResourceIssue  `json:"resourceIssues,omitempty"`
	ProbeIssues     []ProbeIssue     `json:"probeIssues,omitempty"`
	NetworkIssues   []NetworkIssue   `json:"networkIssues,omitempty"`
	Events          []EventSummary   `json:"events,omitempty"`
	Recommendations []Recommendation `json:"recommendations,omitempty"`
}

// RootCause identifies the primary reason for the pod issue
type RootCause struct {
	Category   string  `json:"category"`
	Summary    string  `json:"summary"`
	Details    string  `json:"details,omitempty"`
	Confidence float64 `json:"confidence"`
}

// Recommendation provides actionable guidance
type Recommendation struct {
	Priority    int    `json:"priority"`
	Action      string `json:"action"`
	Command     string `json:"command,omitempty"`
	Explanation string `json:"explanation,omitempty"`
}

// SchedulingInfo contains parsed scheduling failure information
type SchedulingInfo struct {
	Message        string   `json:"message,omitempty"`
	AvailableNodes string   `json:"availableNodes,omitempty"`
	TotalNodes     string   `json:"totalNodes,omitempty"`
	FailureReasons []string `json:"failureReasons,omitempty"`
}

// ContainerIssue describes a problem with a specific container
type ContainerIssue struct {
	ContainerName      string `json:"containerName"`
	State              string `json:"state"`
	Message            string `json:"message,omitempty"`
	ExitCode           int32  `json:"exitCode,omitempty"`
	RestartCount       int32  `json:"restartCount,omitempty"`
	TerminationReason  string `json:"terminationReason,omitempty"`
	TerminationMessage string `json:"terminationMessage,omitempty"`
	LastLogs           string `json:"lastLogs,omitempty"`
}

// VolumeIssue describes a problem with a volume
type VolumeIssue struct {
	VolumeName string `json:"volumeName,omitempty"`
	Type       string `json:"type"`
	Message    string `json:"message"`
}

// ResourceIssue describes a resource-related problem
type ResourceIssue struct {
	ResourceType string `json:"resourceType"`
	Requested    string `json:"requested,omitempty"`
	Limit        string `json:"limit,omitempty"`
	Available    string `json:"available,omitempty"`
	Message      string `json:"message"`
}

// ProbeIssue describes a probe failure
type ProbeIssue struct {
	ContainerName string `json:"containerName"`
	ProbeType     string `json:"probeType"` // liveness, readiness, startup
	FailureCount  int32  `json:"failureCount,omitempty"`
	Message       string `json:"message,omitempty"`
}

// NetworkIssue describes a network-related problem
type NetworkIssue struct {
	Type    string `json:"type"` // DNS, NetworkPolicy, ServiceAccount
	Message string `json:"message"`
}

// EventSummary provides a condensed view of an event
type EventSummary struct {
	Type    string    `json:"type"`
	Reason  string    `json:"reason"`
	Message string    `json:"message"`
	Count   int32     `json:"count,omitempty"`
	Age     time.Time `json:"age,omitempty"`
	Source  string    `json:"source,omitempty"`
}

// CollectedData holds all gathered diagnostic information
type CollectedData struct {
	Pod             *corev1.Pod
	Events          *corev1.EventList
	Node            *corev1.Node
	NodeEvents      *corev1.EventList
	PVCs            map[string]*corev1.PersistentVolumeClaim
	PVCEvents       map[string]*corev1.EventList
	ConfigMaps      map[string]*corev1.ConfigMap
	Secrets         map[string]bool // existence check only, no content
	ServiceAccount  *corev1.ServiceAccount
	PreviousLogs    map[string]string // container name -> previous logs (truncated)
	ResourceQuotas  []corev1.ResourceQuota
	LimitRanges     []corev1.LimitRange
	NetworkPolicies []string // names of network policies affecting the pod
}

// Analyzer performs diagnostic analysis on pod data
type Analyzer interface {
	Analyze(ctx context.Context, data *CollectedData) (*DiagnosticResult, error)
}

// DataCollector gathers data from various Kubernetes resources
type DataCollector interface {
	Collect(ctx context.Context, namespace, podName string) (*CollectedData, error)
}
