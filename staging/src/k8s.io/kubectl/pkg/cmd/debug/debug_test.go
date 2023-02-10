/*
Copyright 2020 The Kubernetes Authors.

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

package debug

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"k8s.io/utils/pointer"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestGenerateDebugContainer(t *testing.T) {
	// Slightly less randomness for testing.
	defer func(old func(int) string) { nameSuffixFunc = old }(nameSuffixFunc)
	var suffixCounter int
	nameSuffixFunc = func(int) string {
		suffixCounter++
		return fmt.Sprint(suffixCounter)
	}

	for _, tc := range []struct {
		name     string
		opts     *DebugOptions
		pod      *corev1.Pod
		expected *corev1.EphemeralContainer
	}{
		{
			name: "minimum fields",
			opts: &DebugOptions{
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger",
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "namespace targeting",
			opts: &DebugOptions{
				Container:       "debugger",
				Image:           "busybox",
				PullPolicy:      corev1.PullIfNotPresent,
				TargetContainer: "myapp",
				Profile:         ProfileLegacy,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger",
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
				TargetContainerName: "myapp",
			},
		},
		{
			name: "debug args as container command",
			opts: &DebugOptions{
				Args:       []string{"/bin/echo", "one", "two", "three"},
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger",
					Command:                  []string{"/bin/echo", "one", "two", "three"},
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "debug args as container args",
			opts: &DebugOptions{
				ArgsOnly:   true,
				Container:  "debugger",
				Args:       []string{"echo", "one", "two", "three"},
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger",
					Args:                     []string{"echo", "one", "two", "three"},
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "random name generation",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-1",
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "random name collision",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
					},
				},
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-2",
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "pod with init containers",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Name: "init-container-1",
						},
						{
							Name: "init-container-2",
						},
					},
					Containers: []corev1.Container{
						{
							Name: "debugger",
						},
					},
				},
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-1",
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "pod with ephemeral containers",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileLegacy,
			},
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger",
						},
					},
					EphemeralContainers: []corev1.EphemeralContainer{
						{
							EphemeralContainerCommon: corev1.EphemeralContainerCommon{
								Name: "ephemeral-container-1",
							},
						},
						{
							EphemeralContainerCommon: corev1.EphemeralContainerCommon{
								Name: "ephemeral-container-2",
							},
						},
					},
				},
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-1",
					Image:                    "busybox",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			},
		},
		{
			name: "general profile",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileGeneral,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-1",
					Image:                    "busybox",
					ImagePullPolicy:          corev1.PullIfNotPresent,
					TerminationMessagePolicy: corev1.TerminationMessageReadFile,
					SecurityContext: &corev1.SecurityContext{
						Capabilities: &corev1.Capabilities{
							Add: []corev1.Capability{"SYS_PTRACE"},
						},
					},
				},
			},
		},
		{
			name: "baseline profile",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileBaseline,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-1",
					Image:                    "busybox",
					ImagePullPolicy:          corev1.PullIfNotPresent,
					TerminationMessagePolicy: corev1.TerminationMessageReadFile,
				},
			},
		},
		{
			name: "restricted profile",
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Profile:    ProfileRestricted,
			},
			expected: &corev1.EphemeralContainer{
				EphemeralContainerCommon: corev1.EphemeralContainerCommon{
					Name:                     "debugger-1",
					Image:                    "busybox",
					ImagePullPolicy:          corev1.PullIfNotPresent,
					TerminationMessagePolicy: corev1.TerminationMessageReadFile,
					SecurityContext: &corev1.SecurityContext{
						RunAsNonRoot: pointer.Bool(true),
						Capabilities: &corev1.Capabilities{
							Drop: []corev1.Capability{"ALL"},
						},
					},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.opts.IOStreams = genericclioptions.NewTestIOStreamsDiscard()
			suffixCounter = 0

			if tc.pod == nil {
				tc.pod = &corev1.Pod{}
			}

			applier, err := NewProfileApplier(tc.opts.Profile)
			if err != nil {
				t.Fatalf("failed to create profile applier: %s: %v", tc.opts.Profile, err)
			}
			tc.opts.Applier = applier

			_, debugContainer, err := tc.opts.generateDebugContainer(tc.pod)
			if err != nil {
				t.Fatalf("fail to generate debug container: %v", err)
			}
			if diff := cmp.Diff(tc.expected, debugContainer); diff != "" {
				t.Error("unexpected diff in generated object: (-want +got):\n", diff)
			}
		})
	}
}

func TestGeneratePodCopyWithDebugContainer(t *testing.T) {
	defer func(old func(int) string) { nameSuffixFunc = old }(nameSuffixFunc)
	var suffixCounter int
	nameSuffixFunc = func(int) string {
		suffixCounter++
		return fmt.Sprint(suffixCounter)
	}

	for _, tc := range []struct {
		name             string
		opts             *DebugOptions
		havePod, wantPod *corev1.Pod
	}{
		{
			name: "basic",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger",
						},
					},
					NodeName: "node-1",
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            "debugger",
							Image:           "busybox",
							ImagePullPolicy: corev1.PullIfNotPresent,
						},
					},
				},
			},
		},
		{
			name: "same node",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				SameNode:   true,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger",
						},
					},
					NodeName: "node-1",
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            "debugger",
							Image:           "busybox",
							ImagePullPolicy: corev1.PullIfNotPresent,
						},
					},
					NodeName: "node-1",
				},
			},
		},
		{
			name: "metadata stripping",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
					Labels: map[string]string{
						"app": "business",
					},
					Annotations: map[string]string{
						"test": "test",
					},
					ResourceVersion:   "1",
					CreationTimestamp: metav1.Time{time.Now()},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
					Annotations: map[string]string{
						"test": "test",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            "debugger",
							Image:           "busybox",
							ImagePullPolicy: corev1.PullIfNotPresent,
						},
					},
				},
			},
		},
		{
			name: "add a debug container",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
						{
							Name:                     "debugger",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "customize envs",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Env: []corev1.EnvVar{{
					Name:  "TEST",
					Value: "test",
				}},
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
						{
							Name:                     "debugger",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							Env: []corev1.EnvVar{{
								Name:  "TEST",
								Value: "test",
							}},
						},
					},
				},
			},
		},
		{
			name: "debug args as container command",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Args:       []string{"/bin/echo", "one", "two", "three"},
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
						{
							Name:                     "debugger",
							Image:                    "busybox",
							Command:                  []string{"/bin/echo", "one", "two", "three"},
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "debug args as container command",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Args:       []string{"one", "two", "three"},
				ArgsOnly:   true,
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
						{
							Name:                     "debugger",
							Image:                    "busybox",
							Args:                     []string{"one", "two", "three"},
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "modify existing command to debug args",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Container:  "debugger",
				Args:       []string{"sleep", "1d"},
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "debugger",
							Command:                  []string{"echo"},
							Image:                    "app",
							Args:                     []string{"one", "two", "three"},
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "debugger",
							Image:                    "app",
							Command:                  []string{"sleep", "1d"},
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "random name",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "business",
						},
						{
							Name:                     "debugger-1",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "random name collision",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
						{
							Name:                     "debugger-2",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "pod with init containers",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Name: "init-container-1",
						},
						{
							Name: "init-container-2",
						},
					},
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					InitContainers: []corev1.Container{
						{
							Name: "init-container-1",
						},
						{
							Name: "init-container-2",
						},
					},
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
						{
							Name:                     "debugger-2",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "pod with ephemeral containers",
			opts: &DebugOptions{
				CopyTo:     "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
					},
					EphemeralContainers: []corev1.EphemeralContainer{
						{
							EphemeralContainerCommon: corev1.EphemeralContainerCommon{
								Name: "ephemeral-container-1",
							},
						},
						{
							EphemeralContainerCommon: corev1.EphemeralContainerCommon{
								Name: "ephemeral-container-2",
							},
						},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "debugger-1",
						},
						{
							Name:                     "debugger-2",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
				},
			},
		},
		{
			name: "shared process namespace",
			opts: &DebugOptions{
				CopyTo:                "debugger",
				Container:             "debugger",
				Image:                 "busybox",
				PullPolicy:            corev1.PullIfNotPresent,
				ShareProcesses:        true,
				shareProcessedChanged: true,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "target",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "debugger",
							ImagePullPolicy:          corev1.PullAlways,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
					NodeName: "node-1",
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "debugger",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "debugger",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
						},
					},
					ShareProcessNamespace: pointer.Bool(true),
				},
			},
		},
		{
			name: "Change image for a named container",
			opts: &DebugOptions{
				Args:        []string{},
				CopyTo:      "myapp-copy",
				Container:   "app",
				Image:       "busybox",
				TargetNames: []string{"myapp"},
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "myapp"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "myapp-copy"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "busybox"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
		},
		{
			name: "Change image for a named container with set-image",
			opts: &DebugOptions{
				CopyTo:    "myapp-copy",
				Container: "app",
				SetImages: map[string]string{"app": "busybox"},
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myapp",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myapp-copy",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "busybox"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
		},
		{
			name: "Change image for all containers with set-image",
			opts: &DebugOptions{
				CopyTo:    "myapp-copy",
				SetImages: map[string]string{"*": "busybox"},
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myapp",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myapp-copy",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "busybox"},
						{Name: "sidecar", Image: "busybox"},
					},
				},
			},
		},
		{
			name: "Change image for multiple containers with set-image",
			opts: &DebugOptions{
				CopyTo:    "myapp-copy",
				SetImages: map[string]string{"*": "busybox", "app": "app-debugger"},
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myapp",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myapp-copy",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "app-debugger"},
						{Name: "sidecar", Image: "busybox"},
					},
				},
			},
		},
		{
			name: "Add interactive debug container minimal args",
			opts: &DebugOptions{
				Args:        []string{},
				Attach:      true,
				CopyTo:      "my-debugger",
				Image:       "busybox",
				Interactive: true,
				TargetNames: []string{"mypod"},
				TTY:         true,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "my-debugger"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
						{
							Name:                     "debugger-1",
							Image:                    "busybox",
							Stdin:                    true,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							TTY:                      true,
						},
					},
				},
			},
		},
		{
			name: "Pod copy: add container and also mutate images",
			opts: &DebugOptions{
				Args:        []string{},
				Attach:      true,
				CopyTo:      "my-debugger",
				Image:       "debian",
				Interactive: true,
				Namespace:   "default",
				SetImages: map[string]string{
					"app":     "app:debug",
					"sidecar": "sidecar:debug",
				},
				ShareProcesses: true,
				TargetNames:    []string{"mypod"},
				TTY:            true,
			},
			havePod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "mypod"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "appimage"},
						{Name: "sidecar", Image: "sidecarimage"},
					},
				},
			},
			wantPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "my-debugger"},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{Name: "app", Image: "app:debug"},
						{Name: "sidecar", Image: "sidecar:debug"},
						{
							Name:                     "debugger-1",
							Image:                    "debian",
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							Stdin:                    true,
							TTY:                      true,
						},
					},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			tc.opts.Applier, err = NewProfileApplier(ProfileLegacy)
			if err != nil {
				t.Fatalf("Fail to create legacy profile: %v", err)
			}
			tc.opts.IOStreams = genericclioptions.NewTestIOStreamsDiscard()
			suffixCounter = 0

			if tc.havePod == nil {
				tc.havePod = &corev1.Pod{}
			}
			gotPod, _, _ := tc.opts.generatePodCopyWithDebugContainer(tc.havePod)
			if diff := cmp.Diff(tc.wantPod, gotPod); diff != "" {
				t.Error("TestGeneratePodCopyWithDebugContainer: diff in generated object: (-want +got):\n", diff)
			}
		})
	}
}

func TestGenerateNodeDebugPod(t *testing.T) {
	defer func(old func(int) string) { nameSuffixFunc = old }(nameSuffixFunc)
	var suffixCounter int
	nameSuffixFunc = func(int) string {
		suffixCounter++
		return fmt.Sprint(suffixCounter)
	}

	for _, tc := range []struct {
		name     string
		node     *corev1.Node
		opts     *DebugOptions
		expected *corev1.Pod
	}{
		{
			name: "minimum options",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-XXX",
				},
			},
			opts: &DebugOptions{
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-debugger-node-XXX-1",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "debugger",
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/host",
									Name:      "host-root",
								},
							},
						},
					},
					HostIPC:       true,
					HostNetwork:   true,
					HostPID:       true,
					NodeName:      "node-XXX",
					RestartPolicy: corev1.RestartPolicyNever,
					Volumes: []corev1.Volume{
						{
							Name: "host-root",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{Path: "/"},
							},
						},
					},
					Tolerations: []corev1.Toleration{
						{
							Operator: corev1.TolerationOpExists,
						},
					},
				},
			},
		},
		{
			name: "debug args as container command",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-XXX",
				},
			},
			opts: &DebugOptions{
				Args:       []string{"/bin/echo", "one", "two", "three"},
				Container:  "custom-debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-debugger-node-XXX-1",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "custom-debugger",
							Command:                  []string{"/bin/echo", "one", "two", "three"},
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/host",
									Name:      "host-root",
								},
							},
						},
					},
					HostIPC:       true,
					HostNetwork:   true,
					HostPID:       true,
					NodeName:      "node-XXX",
					RestartPolicy: corev1.RestartPolicyNever,
					Volumes: []corev1.Volume{
						{
							Name: "host-root",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{Path: "/"},
							},
						},
					},
					Tolerations: []corev1.Toleration{
						{
							Operator: corev1.TolerationOpExists,
						},
					},
				},
			},
		},
		{
			name: "debug args as container args",
			node: &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-XXX",
				},
			},
			opts: &DebugOptions{
				ArgsOnly:   true,
				Container:  "custom-debugger",
				Args:       []string{"echo", "one", "two", "three"},
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
			},
			expected: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-debugger-node-XXX-1",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:                     "custom-debugger",
							Args:                     []string{"echo", "one", "two", "three"},
							Image:                    "busybox",
							ImagePullPolicy:          corev1.PullIfNotPresent,
							TerminationMessagePolicy: corev1.TerminationMessageReadFile,
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/host",
									Name:      "host-root",
								},
							},
						},
					},
					HostIPC:       true,
					HostNetwork:   true,
					HostPID:       true,
					NodeName:      "node-XXX",
					RestartPolicy: corev1.RestartPolicyNever,
					Volumes: []corev1.Volume{
						{
							Name: "host-root",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{Path: "/"},
							},
						},
					},
					Tolerations: []corev1.Toleration{
						{
							Operator: corev1.TolerationOpExists,
						},
					},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var err error
			tc.opts.Applier, err = NewProfileApplier(ProfileLegacy)
			if err != nil {
				t.Fatalf("Fail to create legacy profile: %v", err)
			}
			tc.opts.IOStreams = genericclioptions.NewTestIOStreamsDiscard()
			suffixCounter = 0

			pod, err := tc.opts.generateNodeDebugPod(tc.node)
			if err != nil {
				t.Fatalf("Fail to generate node debug pod: %v", err)
			}
			if diff := cmp.Diff(tc.expected, pod); diff != "" {
				t.Error("unexpected diff in generated object: (-want +got):\n", diff)
			}
		})
	}
}

func TestCompleteAndValidate(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	ioStreams, _, _, _ := genericclioptions.NewTestIOStreams()
	cmpFilter := cmp.FilterPath(func(p cmp.Path) bool {
		switch p.String() {
		// IOStreams contains unexported fields
		case "IOStreams", "Applier":
			return true
		}
		return false
	}, cmp.Ignore())

	tests := []struct {
		name, args string
		wantOpts   *DebugOptions
		wantError  bool
	}{
		{
			name:      "No targets",
			args:      "--image=image",
			wantError: true,
		},
		{
			name:      "Invalid environment variables",
			args:      "--image=busybox --env=FOO mypod",
			wantError: true,
		},
		{
			name:      "Invalid image name",
			args:      "--image=image:label@deadbeef mypod",
			wantError: true,
		},
		{
			name:      "Invalid pull policy",
			args:      "--image=image --image-pull-policy=whenever-you-feel-like-it",
			wantError: true,
		},
		{
			name:      "TTY without stdin",
			args:      "--image=image --tty",
			wantError: true,
		},
		{
			name: "Set image pull policy",
			args: "--image=busybox --image-pull-policy=Always mypod",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Image:          "busybox",
				Namespace:      "test",
				PullPolicy:     corev1.PullPolicy("Always"),
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name: "Multiple targets",
			args: "--image=busybox mypod1 mypod2",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Image:          "busybox",
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod1", "mypod2"},
			},
		},
		{
			name: "Arguments with dash",
			args: "--image=busybox mypod1 mypod2 -- echo 1 2",
			wantOpts: &DebugOptions{
				Args:           []string{"echo", "1", "2"},
				Image:          "busybox",
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod1", "mypod2"},
			},
		},
		{
			name: "Interactive no attach",
			args: "-ti --image=busybox --attach=false mypod",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Attach:         false,
				Image:          "busybox",
				Interactive:    true,
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
				TTY:            true,
			},
		},
		{
			name: "Set environment variables",
			args: "--image=busybox --env=FOO=BAR mypod",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Env:            []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
				Image:          "busybox",
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name: "Ephemeral container: interactive session minimal args",
			args: "mypod -it --image=busybox",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Attach:         true,
				Image:          "busybox",
				Interactive:    true,
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
				TTY:            true,
			},
		},
		{
			name: "Ephemeral container: non-interactive debugger with image and name",
			args: "--image=myproj/debug-tools --image-pull-policy=Always -c debugger mypod",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Container:      "debugger",
				Image:          "myproj/debug-tools",
				Namespace:      "test",
				PullPolicy:     corev1.PullPolicy("Always"),
				Profile:        ProfileLegacy,
				ShareProcesses: true,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name:      "Ephemeral container: no image specified",
			args:      "mypod",
			wantError: true,
		},
		{
			name:      "Ephemeral container: no image but args",
			args:      "mypod -- echo 1 2",
			wantError: true,
		},
		{
			name:      "Ephemeral container: replace not allowed",
			args:      "--replace --image=busybox mypod",
			wantError: true,
		},
		{
			name:      "Ephemeral container: same-node not allowed",
			args:      "--same-node --image=busybox mypod",
			wantError: true,
		},
		{
			name:      "Ephemeral container: incompatible with --set-image",
			args:      "--set-image=*=busybox mypod",
			wantError: true,
		},
		{
			name: "Pod copy: interactive debug container minimal args",
			args: "mypod -it --image=busybox --copy-to=my-debugger",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Attach:         true,
				CopyTo:         "my-debugger",
				Image:          "busybox",
				Interactive:    true,
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
				TTY:            true,
			},
		},
		{
			name: "Pod copy: non-interactive with debug container, image name and command",
			args: "mypod --image=busybox --container=my-container --copy-to=my-debugger -- sleep 1d",
			wantOpts: &DebugOptions{
				Args:           []string{"sleep", "1d"},
				Container:      "my-container",
				CopyTo:         "my-debugger",
				Image:          "busybox",
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name: "Pod copy: explicit attach",
			args: "mypod --image=busybox --copy-to=my-debugger --attach -- sleep 1d",
			wantOpts: &DebugOptions{
				Args:           []string{"sleep", "1d"},
				Attach:         true,
				CopyTo:         "my-debugger",
				Image:          "busybox",
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name: "Pod copy: replace single image of existing container",
			args: "mypod --image=busybox --container=my-container --copy-to=my-debugger",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Container:      "my-container",
				CopyTo:         "my-debugger",
				Image:          "busybox",
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name: "Pod copy: mutate existing container images",
			args: "mypod --set-image=*=busybox,app=app-debugger --copy-to=my-debugger",
			wantOpts: &DebugOptions{
				Args:      []string{},
				CopyTo:    "my-debugger",
				Namespace: "test",
				SetImages: map[string]string{
					"*":   "busybox",
					"app": "app-debugger",
				},
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
			},
		},
		{
			name: "Pod copy: add container and also mutate images",
			args: "mypod -it --copy-to=my-debugger --image=debian --set-image=app=app:debug,sidecar=sidecar:debug",
			wantOpts: &DebugOptions{
				Args:        []string{},
				Attach:      true,
				CopyTo:      "my-debugger",
				Image:       "debian",
				Interactive: true,
				Namespace:   "test",
				SetImages: map[string]string{
					"app":     "app:debug",
					"sidecar": "sidecar:debug",
				},
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
				TTY:            true,
			},
		},
		{
			name: "Pod copy: change command",
			args: "mypod -it --copy-to=my-debugger --container=mycontainer -- sh",
			wantOpts: &DebugOptions{
				Attach:         true,
				Args:           []string{"sh"},
				Container:      "mycontainer",
				CopyTo:         "my-debugger",
				Interactive:    true,
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"mypod"},
				TTY:            true,
			},
		},
		{
			name:      "Pod copy: no image specified",
			args:      "mypod -it --copy-to=my-debugger",
			wantError: true,
		},
		{
			name:      "Pod copy: args but no image specified",
			args:      "mypod --copy-to=my-debugger -- echo milo",
			wantError: true,
		},
		{
			name:      "Pod copy: --target not allowed",
			args:      "mypod --target --image=busybox --copy-to=my-debugger",
			wantError: true,
		},
		{
			name:      "Pod copy: invalid --set-image",
			args:      "mypod --set-image=*=SUPERGOODIMAGE#1!!!! --copy-to=my-debugger",
			wantError: true,
		},
		{
			name:      "Pod copy: specifying attach without existing or newly created container",
			args:      "mypod --set-image=*=busybox --copy-to=my-debugger --attach",
			wantError: true,
		},
		{
			name: "Node: interactive session minimal args",
			args: "node/mynode -it --image=busybox",
			wantOpts: &DebugOptions{
				Args:           []string{},
				Attach:         true,
				Image:          "busybox",
				Interactive:    true,
				Namespace:      "test",
				ShareProcesses: true,
				Profile:        ProfileLegacy,
				TargetNames:    []string{"node/mynode"},
				TTY:            true,
			},
		},
		{
			name:      "Node: no image specified",
			args:      "node/mynode -it",
			wantError: true,
		},
		{
			name:      "Node: --replace not allowed",
			args:      "--image=busybox --replace node/mynode",
			wantError: true,
		},
		{
			name:      "Node: --same-node not allowed",
			args:      "--image=busybox --same-node node/mynode",
			wantError: true,
		},
		{
			name:      "Node: --set-image not allowed",
			args:      "--image=busybox --set-image=*=busybox node/mynode",
			wantError: true,
		},
		{
			name:      "Node: --target not allowed",
			args:      "node/mynode --target --image=busybox",
			wantError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opts := NewDebugOptions(ioStreams)
			var gotError error
			cmd := &cobra.Command{
				Run: func(cmd *cobra.Command, args []string) {
					gotError = opts.Complete(tf, cmd, args)
					if gotError != nil {
						return
					}
					gotError = opts.Validate()
				},
			}
			cmd.SetArgs(strings.Split(tc.args, " "))
			addDebugFlags(cmd, opts)

			cmdError := cmd.Execute()

			if tc.wantError {
				if cmdError != nil || gotError != nil {
					return
				}
				t.Fatalf("CompleteAndValidate got nil errors but wantError: %v", tc.wantError)
			} else if cmdError != nil {
				t.Fatalf("cmd.Execute got error '%v' executing test cobra.Command, wantError: %v", cmdError, tc.wantError)
			} else if gotError != nil {
				t.Fatalf("CompleteAndValidate got error: '%v', wantError: %v", gotError, tc.wantError)
			}

			if diff := cmp.Diff(tc.wantOpts, opts, cmpFilter, cmpopts.IgnoreFields(DebugOptions{},
				"attachChanged", "shareProcessedChanged", "podClient", "WarningPrinter", "Applier", "explicitNamespace")); diff != "" {
				t.Error("CompleteAndValidate unexpected diff in generated object: (-want +got):\n", diff)
			}
		})
	}
}
