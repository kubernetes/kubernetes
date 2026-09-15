//go:build windows

/*
Copyright 2022 The Kubernetes Authors.

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

package kuberuntime

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/initoption"
)

// logBufferTB is a ktesting.TB that captures log output so tests can assert on
// the warnings that the kubelet emits while building Windows container config.
type logBufferTB struct {
	strings.Builder
}

func (b *logBufferTB) Attr(key, value string)            {}
func (b *logBufferTB) Chdir(dir string)                  {}
func (b *logBufferTB) Cleanup(func())                    {}
func (b *logBufferTB) Error(args ...any)                 {}
func (b *logBufferTB) Errorf(format string, args ...any) {}
func (b *logBufferTB) Fail()                             {}
func (b *logBufferTB) FailNow()                          {}
func (b *logBufferTB) Failed() bool                      { return false }
func (b *logBufferTB) Fatal(args ...any)                 {}
func (b *logBufferTB) Fatalf(format string, args ...any) {}
func (b *logBufferTB) Helper()                           {}
func (b *logBufferTB) Log(args ...any)                   { fmt.Fprintln(b, args...) }
func (b *logBufferTB) Logf(format string, args ...any)   { fmt.Fprintf(b, format, args...) }
func (b *logBufferTB) Name() string                      { return "logBufferTB" }
func (b *logBufferTB) Setenv(key, value string)          {}
func (b *logBufferTB) Skip(args ...any)                  {}
func (b *logBufferTB) Skipf(format string, args ...any)  {}
func (b *logBufferTB) SkipNow()                          {}
func (b *logBufferTB) Skipped() bool                     { return false }
func (b *logBufferTB) TempDir() string                   { return "" }

func TestWarnIgnoredWindowsFields(t *testing.T) {
	unmasked := v1.UnmaskedProcMount
	localhostProfile := "profile.json"

	tests := []struct {
		name        string
		podSc       *v1.PodSecurityContext
		containerSc *v1.SecurityContext
		wantLogs    []string
	}{
		{
			name:        "no ignored fields requested",
			podSc:       nil,
			containerSc: nil,
			wantLogs:    []string{},
		},
		{
			name:        "procMount unmasked requested",
			podSc:       &v1.PodSecurityContext{},
			containerSc: &v1.SecurityContext{ProcMount: &unmasked},
			wantLogs: []string{
				"Windows containers do not support securityContext.procMount",
			},
		},
		{
			name:        "seccomp localhost requested at pod level",
			podSc:       &v1.PodSecurityContext{SeccompProfile: &v1.SeccompProfile{Type: v1.SeccompProfileTypeLocalhost, LocalhostProfile: &localhostProfile}},
			containerSc: &v1.SecurityContext{},
			wantLogs: []string{
				"Windows containers do not support a seccomp profile of type Localhost",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			buf := &logBufferTB{}
			tCtx := ktesting.Init(buf, initoption.BufferLogs(true))
			logger := klog.FromContext(tCtx)

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "new", UID: types.UID("12345678")},
				Spec:       v1.PodSpec{SecurityContext: tc.podSc, Containers: []v1.Container{{Name: "foo", SecurityContext: tc.containerSc}}},
			}

			warnIgnoredWindowsFields(logger, pod, &pod.Spec.Containers[0])

			got := buf.String()
			for _, want := range tc.wantLogs {
				assert.Contains(t, got, want, "expected log %q in output:\n%s", want, got)
			}
			if len(tc.wantLogs) == 0 {
				assert.NotContains(t, got, "will be ignored", "expected no ignored-field warnings in output:\n%s", got)
			}
		})
	}
}

func TestApplyPlatformSpecificContainerConfig(t *testing.T) {
	tCtx := ktesting.Init(t)
	_, _, fakeRuntimeSvc, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	containerConfig := &runtimeapi.ContainerConfig{}

	resources := v1.ResourceRequirements{
		Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("128Mi"),
			v1.ResourceCPU:    resource.MustParse("1"),
		},
		Limits: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("256Mi"),
			v1.ResourceCPU:    resource.MustParse("3"),
		},
	}

	gmsaCredSpecName := "gmsa spec name"
	gmsaCredSpec := "credential spec"
	username := "ContainerAdministrator"
	asHostProcess := true
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "foo",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					Command:         []string{"testCommand"},
					WorkingDir:      "testWorkingDir",
					Resources:       resources,
					SecurityContext: &v1.SecurityContext{
						WindowsOptions: &v1.WindowsSecurityContextOptions{
							GMSACredentialSpecName: &gmsaCredSpecName,
							GMSACredentialSpec:     &gmsaCredSpec,
							RunAsUserName:          &username,
							HostProcess:            &asHostProcess,
						},
					},
				},
			},
		},
	}

	err = fakeRuntimeSvc.applyPlatformSpecificContainerConfig(tCtx, containerConfig, &pod.Spec.Containers[0], pod, new(int64), "foo", nil)
	require.NoError(t, err)

	limit := int64(3000)
	expectedCpuMax := 10 * limit / int64(winstats.ProcessorCount())
	// Above, we're setting the limit to 3 CPUs. But we can't expect more than 100% of the CPUs
	// we have. (e.g.: if we only have 2 CPUs, we can't have 150% CPU max).
	if expectedCpuMax > 10000 {
		expectedCpuMax = 10000
	}
	expectedWindowsConfig := &runtimeapi.WindowsContainerConfig{
		Resources: &runtimeapi.WindowsContainerResources{
			CpuMaximum:         expectedCpuMax,
			MemoryLimitInBytes: 256 * 1024 * 1024,
		},
		SecurityContext: &runtimeapi.WindowsContainerSecurityContext{
			CredentialSpec: gmsaCredSpec,
			RunAsUsername:  "ContainerAdministrator",
			HostProcess:    true,
		},
	}
	assert.Equal(t, expectedWindowsConfig, containerConfig.Windows)
}

func TestCalculateCPUMaximum(t *testing.T) {
	tests := []struct {
		name     string
		cpuLimit resource.Quantity
		cpuCount int64
		want     int64
	}{
		{
			name:     "max range when same amount",
			cpuLimit: resource.MustParse("1"),
			cpuCount: 1,
			want:     10000,
		},
		{
			name:     "percentage calculation is working as intended",
			cpuLimit: resource.MustParse("94"),
			cpuCount: 96,
			want:     9791,
		},
		{
			name:     "half range when half amount",
			cpuLimit: resource.MustParse("1"),
			cpuCount: 2,
			want:     5000,
		},
		{
			name:     "max range when more requested than available",
			cpuLimit: resource.MustParse("2"),
			cpuCount: 1,
			want:     10000,
		},
		{
			name:     "min range when less than minimum",
			cpuLimit: resource.MustParse("1m"),
			cpuCount: 100,
			want:     1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, calculateCPUMaximum(&tt.cpuLimit, tt.cpuCount))
		})
	}
}

func TestCalculateWindowsResources(t *testing.T) {
	// TODO: remove skip once the failing test has been fixed.
	t.Skip("Skip failing test on Windows.")

	tCtx := ktesting.Init(t)
	_, _, fakeRuntimeSvc, err := createTestRuntimeManager(tCtx)
	require.NoError(t, err)

	tests := []struct {
		name     string
		cpuLim   resource.Quantity
		memLim   resource.Quantity
		expected *runtimeapi.WindowsContainerResources
	}{
		{
			name:   "Request128MBLimit256MB",
			cpuLim: resource.MustParse("2"),
			memLim: resource.MustParse("128Mi"),
			expected: &runtimeapi.WindowsContainerResources{
				CpuMaximum:         2500,
				MemoryLimitInBytes: 134217728,
			},
		},
		{
			name:   "RequestNoMemory",
			cpuLim: resource.MustParse("8"),
			memLim: resource.MustParse("0"),
			expected: &runtimeapi.WindowsContainerResources{
				CpuMaximum:         10000,
				MemoryLimitInBytes: 0,
			},
		},
		{
			name:   "RequestZeroCPU",
			cpuLim: resource.MustParse("0"),
			memLim: resource.MustParse("128Mi"),
			expected: &runtimeapi.WindowsContainerResources{
				CpuMaximum:         1,
				MemoryLimitInBytes: 134217728,
			},
		},
	}
	for _, test := range tests {
		windowsContainerResources := fakeRuntimeSvc.calculateWindowsResources(tCtx, &test.cpuLim, &test.memLim)
		assert.Equal(t, test.expected, windowsContainerResources)
	}
}

// TestWarnPodLevelIgnoredWindowsFields verifies that pod-level fields that
// Windows ignores (fsGroup, fsGroupChangePolicy) produce exactly the expected
// warnings from the dedicated pod-level helper. These are pod-wide settings, so
// the helper is invoked once per pod rather than once per container.
func TestWarnPodLevelIgnoredWindowsFields(t *testing.T) {
	fsGroup := int64(1000)
	always := v1.FSGroupChangeAlways
	buf := &logBufferTB{}
	tCtx := ktesting.Init(buf, initoption.BufferLogs(true))
	logger := klog.FromContext(tCtx)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "new", UID: "12345678"},
		Spec: v1.PodSpec{
			Containers:      []v1.Container{{Name: "foo"}, {Name: "sidecar"}},
			SecurityContext: &v1.PodSecurityContext{FSGroup: &fsGroup, FSGroupChangePolicy: &always},
		},
	}
	warnPodLevelIgnoredWindowsFields(logger, pod)
	got := buf.String()
	assert.Contains(t, got, "Windows containers do not support securityContext.fsGroup", "expected fsGroup warning:\n%s", got)
	assert.Contains(t, got, "Windows containers do not support securityContext.fsGroupChangePolicy", "expected fsGroupChangePolicy warning:\n%s", got)
}
