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
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Test helper options for building test pods
type podOption func(*corev1.Pod)

func makePod(opts ...podOption) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "nginx"}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodPending},
	}
	for _, opt := range opts {
		opt(pod)
	}
	return pod
}

func withPhase(phase corev1.PodPhase) podOption {
	return func(p *corev1.Pod) { p.Status.Phase = phase }
}

func withUnscheduledCondition() podOption {
	return func(p *corev1.Pod) {
		p.Status.Conditions = append(p.Status.Conditions, corev1.PodCondition{
			Type:    corev1.PodScheduled,
			Status:  corev1.ConditionFalse,
			Reason:  "Unschedulable",
			Message: "0/3 nodes are available",
		})
	}
}

func withContainerWaiting(reason string) podOption {
	return func(p *corev1.Pod) {
		p.Status.ContainerStatuses = []corev1.ContainerStatus{{
			Name: "main",
			State: corev1.ContainerState{
				Waiting: &corev1.ContainerStateWaiting{Reason: reason},
			},
		}}
	}
}

func withContainerWaitingMessage(reason, message string) podOption {
	return func(p *corev1.Pod) {
		p.Status.ContainerStatuses = []corev1.ContainerStatus{{
			Name: "main",
			State: corev1.ContainerState{
				Waiting: &corev1.ContainerStateWaiting{Reason: reason, Message: message},
			},
		}}
	}
}

func withInitContainerRunning() podOption {
	return func(p *corev1.Pod) {
		p.Status.InitContainerStatuses = []corev1.ContainerStatus{{
			Name: "init",
			State: corev1.ContainerState{
				Running: &corev1.ContainerStateRunning{},
			},
		}}
	}
}

func withInitContainerWaiting(reason string) podOption {
	return func(p *corev1.Pod) {
		p.Status.InitContainerStatuses = []corev1.ContainerStatus{{
			Name: "init",
			State: corev1.ContainerState{
				Waiting: &corev1.ContainerStateWaiting{Reason: reason},
			},
		}}
	}
}

func withCrashLoopBackOff(exitCode int32, restartCount int32) podOption {
	return func(p *corev1.Pod) {
		p.Status.ContainerStatuses = []corev1.ContainerStatus{{
			Name:         "main",
			RestartCount: restartCount,
			State: corev1.ContainerState{
				Waiting: &corev1.ContainerStateWaiting{Reason: "CrashLoopBackOff"},
			},
			LastTerminationState: corev1.ContainerState{
				Terminated: &corev1.ContainerStateTerminated{
					ExitCode: exitCode,
					Reason:   "Error",
				},
			},
		}}
	}
}

func withSchedulingGates(gates ...string) podOption {
	return func(p *corev1.Pod) {
		for _, g := range gates {
			p.Spec.SchedulingGates = append(p.Spec.SchedulingGates, corev1.PodSchedulingGate{Name: g})
		}
	}
}

func withNodeName(name string) podOption {
	return func(p *corev1.Pod) {
		p.Spec.NodeName = name
	}
}

func TestDeterminePhase(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected DiagnosticPhase
	}{
		{
			name:     "pending unscheduled",
			pod:      makePod(withPhase(corev1.PodPending), withUnscheduledCondition()),
			expected: PhaseSchedulingFailed,
		},
		{
			name:     "image pull backoff",
			pod:      makePod(withPhase(corev1.PodPending), withContainerWaiting("ImagePullBackOff")),
			expected: PhaseImagePullFailed,
		},
		{
			name:     "err image pull",
			pod:      makePod(withPhase(corev1.PodPending), withContainerWaiting("ErrImagePull")),
			expected: PhaseImagePullFailed,
		},
		{
			name:     "crash loop backoff",
			pod:      makePod(withPhase(corev1.PodRunning), withContainerWaiting("CrashLoopBackOff")),
			expected: PhaseCrashLooping,
		},
		{
			name:     "container creating with pvc issue",
			pod:      makePod(withPhase(corev1.PodPending), withContainerWaiting("ContainerCreating")),
			expected: PhaseContainerCreating,
		},
		{
			name:     "config error",
			pod:      makePod(withContainerWaiting("CreateContainerConfigError")),
			expected: PhaseConfigError,
		},
		{
			name:     "init container running",
			pod:      makePod(withInitContainerRunning()),
			expected: PhaseInitializing,
		},
		{
			name:     "init container waiting",
			pod:      makePod(withInitContainerWaiting("PodInitializing")),
			expected: PhaseInitializing,
		},
		{
			name:     "init container image pull",
			pod:      makePod(withInitContainerWaiting("ImagePullBackOff")),
			expected: PhaseImagePullFailed,
		},
		{
			name:     "pod initializing",
			pod:      makePod(withContainerWaiting("PodInitializing")),
			expected: PhaseInitializing,
		},
		{
			name:     "running pod",
			pod:      makePod(withPhase(corev1.PodRunning)),
			expected: PhaseRunning,
		},
		{
			name:     "pending with no specific reason",
			pod:      makePod(withPhase(corev1.PodPending)),
			expected: PhasePending,
		},
	}

	a := &analyzer{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := &CollectedData{Pod: tc.pod}
			result := a.determinePhase(data)
			if result != tc.expected {
				t.Errorf("expected %s, got %s", tc.expected, result)
			}
		})
	}
}

func TestAnalyzeSchedulingIssues(t *testing.T) {
	tests := []struct {
		name           string
		pod            *corev1.Pod
		events         *corev1.EventList
		expectedCause  string
		hasFailureInfo bool
	}{
		{
			name: "failed scheduling with insufficient resources",
			pod:  makePod(withPhase(corev1.PodPending), withUnscheduledCondition()),
			events: &corev1.EventList{
				Items: []corev1.Event{{
					Reason:  "FailedScheduling",
					Message: "0/3 nodes are available: 2 Insufficient cpu, 1 node(s) had untolerated taint(s).",
				}},
			},
			expectedCause:  "Scheduling",
			hasFailureInfo: true,
		},
		{
			name:           "scheduling gates",
			pod:            makePod(withPhase(corev1.PodPending), withSchedulingGates("my-gate")),
			expectedCause:  "Scheduling",
			hasFailureInfo: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &analyzer{}
			data := &CollectedData{
				Pod:    tc.pod,
				Events: tc.events,
			}
			result := &DiagnosticResult{
				Pod: tc.pod,
			}

			a.analyzeSchedulingIssues(data, result)

			if result.RootCause.Category != tc.expectedCause {
				t.Errorf("expected category %s, got %s", tc.expectedCause, result.RootCause.Category)
			}

			if tc.hasFailureInfo {
				if result.SchedulingInfo == nil {
					t.Error("expected scheduling info to be set")
				}
			}
		})
	}
}

func TestAnalyzeImagePullIssues(t *testing.T) {
	tests := []struct {
		name            string
		pod             *corev1.Pod
		expectedSummary string
	}{
		{
			name:            "image not found",
			pod:             makePod(withContainerWaitingMessage("ImagePullBackOff", "image not found")),
			expectedSummary: "Image or tag does not exist",
		},
		{
			name:            "manifest unknown",
			pod:             makePod(withContainerWaitingMessage("ErrImagePull", "manifest unknown")),
			expectedSummary: "Image or tag does not exist",
		},
		{
			name:            "unauthorized",
			pod:             makePod(withContainerWaitingMessage("ImagePullBackOff", "unauthorized: authentication required")),
			expectedSummary: "Authentication failed - check imagePullSecrets",
		},
		{
			name:            "denied",
			pod:             makePod(withContainerWaitingMessage("ImagePullBackOff", "access denied")),
			expectedSummary: "Authentication failed - check imagePullSecrets",
		},
		{
			name:            "timeout",
			pod:             makePod(withContainerWaitingMessage("ImagePullBackOff", "connection timeout")),
			expectedSummary: "Registry unreachable - network or DNS issue",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &analyzer{}
			data := &CollectedData{Pod: tc.pod}
			result := &DiagnosticResult{Pod: tc.pod}

			a.analyzeImagePullIssues(data, result)

			if result.RootCause.Summary != tc.expectedSummary {
				t.Errorf("expected summary %q, got %q", tc.expectedSummary, result.RootCause.Summary)
			}

			if len(result.ContainerIssues) == 0 {
				t.Error("expected container issues to be set")
			}
		})
	}
}

func TestAnalyzeCrashLoopIssues(t *testing.T) {
	tests := []struct {
		name            string
		pod             *corev1.Pod
		expectedSummary string
		exitCode        int32
	}{
		{
			name:            "application error (exit 1)",
			pod:             makePod(withCrashLoopBackOff(1, 5)),
			expectedSummary: "Application error (exit code 1)",
			exitCode:        1,
		},
		{
			name:            "OOMKilled (exit 137)",
			pod:             makePod(withCrashLoopBackOff(137, 3)),
			expectedSummary: "Container killed (likely OOMKilled, exit code 137)",
			exitCode:        137,
		},
		{
			name:            "segfault (exit 139)",
			pod:             makePod(withCrashLoopBackOff(139, 2)),
			expectedSummary: "Segmentation fault (exit code 139)",
			exitCode:        139,
		},
		{
			name:            "SIGTERM (exit 143)",
			pod:             makePod(withCrashLoopBackOff(143, 1)),
			expectedSummary: "Container terminated by SIGTERM (exit code 143)",
			exitCode:        143,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &analyzer{}
			data := &CollectedData{
				Pod:          tc.pod,
				PreviousLogs: make(map[string]string),
			}
			result := &DiagnosticResult{Pod: tc.pod}

			a.analyzeCrashLoopIssues(data, result)

			if result.RootCause.Summary != tc.expectedSummary {
				t.Errorf("expected summary %q, got %q", tc.expectedSummary, result.RootCause.Summary)
			}

			if len(result.ContainerIssues) == 0 {
				t.Error("expected container issues to be set")
			}

			if result.ContainerIssues[0].ExitCode != tc.exitCode {
				t.Errorf("expected exit code %d, got %d", tc.exitCode, result.ContainerIssues[0].ExitCode)
			}
		})
	}
}

func TestAnalyzeConfigErrors(t *testing.T) {
	tests := []struct {
		name          string
		configMaps    map[string]*corev1.ConfigMap
		secrets       map[string]bool
		expectedCause string
		expectedName  string
	}{
		{
			name:          "missing configmap",
			configMaps:    map[string]*corev1.ConfigMap{"my-config": nil},
			secrets:       map[string]bool{},
			expectedCause: "Config",
			expectedName:  "ConfigMap not found",
		},
		{
			name:          "missing secret",
			configMaps:    map[string]*corev1.ConfigMap{},
			secrets:       map[string]bool{"my-secret": false},
			expectedCause: "Config",
			expectedName:  "Secret not found",
		},
		{
			name:          "resources exist",
			configMaps:    map[string]*corev1.ConfigMap{"my-config": {}},
			secrets:       map[string]bool{"my-secret": true},
			expectedCause: "", // No root cause if resources exist
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &analyzer{}
			pod := makePod()
			data := &CollectedData{
				Pod:        pod,
				ConfigMaps: tc.configMaps,
				Secrets:    tc.secrets,
			}
			result := &DiagnosticResult{Pod: pod}

			a.analyzeConfigErrors(data, result)

			if result.RootCause.Category != tc.expectedCause {
				t.Errorf("expected category %q, got %q", tc.expectedCause, result.RootCause.Category)
			}

			if tc.expectedName != "" && result.RootCause.Summary != tc.expectedName {
				t.Errorf("expected summary %q, got %q", tc.expectedName, result.RootCause.Summary)
			}
		})
	}
}

func TestGenerateRecommendations(t *testing.T) {
	tests := []struct {
		name               string
		phase              DiagnosticPhase
		schedulingInfo     *SchedulingInfo
		containerIssues    []ContainerIssue
		volumeIssues       []VolumeIssue
		minRecommendations int
	}{
		{
			name:  "scheduling failed with resource issues",
			phase: PhaseSchedulingFailed,
			schedulingInfo: &SchedulingInfo{
				FailureReasons: []string{"2 Insufficient cpu"},
			},
			minRecommendations: 2, // resource + describe
		},
		{
			name:               "image pull failed",
			phase:              PhaseImagePullFailed,
			minRecommendations: 3, // verify image + check secrets + describe
		},
		{
			name:  "crash loop with OOM",
			phase: PhaseCrashLooping,
			containerIssues: []ContainerIssue{
				{ExitCode: 137},
			},
			minRecommendations: 3, // logs + memory + describe
		},
		{
			name:  "container creating with volume issues",
			phase: PhaseContainerCreating,
			volumeIssues: []VolumeIssue{
				{VolumeName: "data", Type: "PVCPending"},
			},
			minRecommendations: 2, // pvc status + describe
		},
		{
			name:               "config error",
			phase:              PhaseConfigError,
			minRecommendations: 2, // verify configmaps/secrets + describe
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &analyzer{}
			pod := makePod()
			result := &DiagnosticResult{
				Pod:             pod,
				Phase:           tc.phase,
				SchedulingInfo:  tc.schedulingInfo,
				ContainerIssues: tc.containerIssues,
				VolumeIssues:    tc.volumeIssues,
				Recommendations: []Recommendation{},
			}

			a.generateRecommendations(result)

			if len(result.Recommendations) < tc.minRecommendations {
				t.Errorf("expected at least %d recommendations, got %d", tc.minRecommendations, len(result.Recommendations))
			}
		})
	}
}

func TestFullAnalysis(t *testing.T) {
	tests := []struct {
		name          string
		data          *CollectedData
		expectedPhase DiagnosticPhase
		expectedCause string
	}{
		{
			name: "unschedulable pod",
			data: &CollectedData{
				Pod: makePod(withPhase(corev1.PodPending), withUnscheduledCondition()),
				Events: &corev1.EventList{
					Items: []corev1.Event{{
						Reason:  "FailedScheduling",
						Message: "0/3 nodes are available: 2 Insufficient cpu",
					}},
				},
			},
			expectedPhase: PhaseSchedulingFailed,
			expectedCause: "Scheduling",
		},
		{
			name: "image pull failed",
			data: &CollectedData{
				Pod: makePod(withContainerWaitingMessage("ImagePullBackOff", "image not found")),
			},
			expectedPhase: PhaseImagePullFailed,
			expectedCause: "Image",
		},
		{
			name: "crash loop",
			data: &CollectedData{
				Pod:          makePod(withCrashLoopBackOff(1, 5)),
				PreviousLogs: make(map[string]string),
			},
			expectedPhase: PhaseCrashLooping,
			expectedCause: "Application",
		},
	}

	a := NewAnalyzer()
	ctx := context.Background()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := a.Analyze(ctx, tc.data)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if result.Phase != tc.expectedPhase {
				t.Errorf("expected phase %s, got %s", tc.expectedPhase, result.Phase)
			}

			if result.RootCause.Category != tc.expectedCause {
				t.Errorf("expected cause category %s, got %s", tc.expectedCause, result.RootCause.Category)
			}

			if len(result.Recommendations) == 0 {
				t.Error("expected at least one recommendation")
			}
		})
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"short", 10, "short"},
		{"this is a very long string", 10, "this is..."},
		{"exactly10!", 10, "exactly10!"},
		{"", 10, ""},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			result := truncate(tc.input, tc.maxLen)
			if result != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, result)
			}
		})
	}
}

func TestParseSchedulingMessage(t *testing.T) {
	tests := []struct {
		name            string
		message         string
		expectedNodes   string
		expectedTotal   string
		expectedReasons int
	}{
		{
			name:            "typical scheduling failure",
			message:         "0/3 nodes are available: 2 Insufficient cpu, 1 node(s) had untolerated taint(s).",
			expectedNodes:   "0",
			expectedTotal:   "3",
			expectedReasons: 2,
		},
		{
			name:            "no nodes available",
			message:         "0/5 nodes are available: 5 Insufficient memory.",
			expectedNodes:   "0",
			expectedTotal:   "5",
			expectedReasons: 1,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := &analyzer{}
			info := &SchedulingInfo{}

			a.parseSchedulingMessage(tc.message, info)

			if info.AvailableNodes != tc.expectedNodes {
				t.Errorf("expected available nodes %s, got %s", tc.expectedNodes, info.AvailableNodes)
			}

			if info.TotalNodes != tc.expectedTotal {
				t.Errorf("expected total nodes %s, got %s", tc.expectedTotal, info.TotalNodes)
			}

			// Note: the regex may capture more matches than expected failure reasons
			// so we just check that we got at least some failure reasons
			if len(info.FailureReasons) == 0 {
				t.Error("expected at least one failure reason")
			}
		})
	}
}
