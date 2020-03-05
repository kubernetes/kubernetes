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
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
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
				Container:  "debugger",
				Image:      "busybox",
				PullPolicy: corev1.PullIfNotPresent,
				Target:     "myapp",
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
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.opts.IOStreams = genericclioptions.NewTestIOStreamsDiscard()
			suffixCounter = 0

			if tc.pod == nil {
				tc.pod = &corev1.Pod{}
			}
			if diff := cmp.Diff(tc.expected, tc.opts.generateDebugContainer(tc.pod)); diff != "" {
				t.Error("unexpected diff in generated object: (-want +got):\n", diff)
			}
		})
	}
}
