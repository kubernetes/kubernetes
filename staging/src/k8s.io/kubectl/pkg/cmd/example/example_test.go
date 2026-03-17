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

package example

import (
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	yaml "sigs.k8s.io/yaml"
)

func runExample(t *testing.T, args []string, flags *Flags) (string, error) {
	t.Helper()

	tf := cmdtesting.NewTestFactory()
	t.Cleanup(tf.Cleanup)

	o, err := flags.ToOptions(tf, args)
	if err != nil {
		return "", err
	}
	if err := o.Validate(); err != nil {
		return "", err
	}
	if err := o.Run(); err != nil {
		return "", err
	}

	return strings.TrimSpace(flags.Out.(*strings.Builder).String()), nil
}

func newTestFlags() *Flags {
	out := &strings.Builder{}
	streams := genericiooptions.IOStreams{In: nil, Out: out, ErrOut: &strings.Builder{}}
	return NewFlags(streams)
}

func TestExampleInvalidArgs(t *testing.T) {
	tests := []struct {
		name string
		args []string
	}{
		{name: "missing arg", args: []string{}},
		{name: "too many args", args: []string{"pod", "extra"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags := newTestFlags()
			_, err := runExample(t, tt.args, flags)
			if err == nil {
				t.Fatalf("expected validation error")
			}
		})
	}
}

func TestExampleList(t *testing.T) {
	flags := newTestFlags()
	flags.List = true

	output, err := runExample(t, []string{}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, expected := range []string{"pod", "deployment", "service", "persistentvolumeclaim", "secret", "customresourcedefinition"} {
		if !strings.Contains(output, expected) {
			t.Fatalf("expected %q in --list output, got:\n%s", expected, output)
		}
	}
}

func TestExamplePodNameAndImageOverride(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "my-pod"
	flags.Image = "busybox:1.36"

	output, err := runExample(t, []string{"pod"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal([]byte(output), &pod); err != nil {
		t.Fatalf("failed to unmarshal pod yaml: %v", err)
	}

	if pod.Kind != "Pod" {
		t.Fatalf("expected kind Pod, got %q", pod.Kind)
	}
	if pod.Name != "my-pod" {
		t.Fatalf("expected metadata.name my-pod, got %q", pod.Name)
	}
	if got := pod.Spec.Containers[0].Image; got != "busybox:1.36" {
		t.Fatalf("expected image busybox:1.36, got %q", got)
	}
}

func TestExampleDeploymentReplicasAndImageOverride(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "web"
	flags.Image = "nginx:1.30"
	flags.Replicas = 3

	output, err := runExample(t, []string{"deployment"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var deployment appsv1.Deployment
	if err := yaml.Unmarshal([]byte(output), &deployment); err != nil {
		t.Fatalf("failed to unmarshal deployment yaml: %v", err)
	}

	if deployment.Name != "web" {
		t.Fatalf("expected metadata.name web, got %q", deployment.Name)
	}
	if deployment.Spec.Replicas == nil || *deployment.Spec.Replicas != 3 {
		t.Fatalf("expected replicas 3, got %v", deployment.Spec.Replicas)
	}
	if got := deployment.Spec.Template.Spec.Containers[0].Image; got != "nginx:1.30" {
		t.Fatalf("expected image nginx:1.30, got %q", got)
	}
}

func TestExampleFallbackAliases(t *testing.T) {
	tests := []struct {
		name       string
		alias      string
		assertKind func(t *testing.T, output string)
	}{
		{
			name:  "po alias",
			alias: "po",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var pod corev1.Pod
				if err := yaml.Unmarshal([]byte(output), &pod); err != nil {
					t.Fatalf("failed to unmarshal pod: %v", err)
				}
				if pod.Kind != "Pod" {
					t.Fatalf("expected Pod kind, got %q", pod.Kind)
				}
			},
		},
		{
			name:  "deploy alias",
			alias: "deploy",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var deployment appsv1.Deployment
				if err := yaml.Unmarshal([]byte(output), &deployment); err != nil {
					t.Fatalf("failed to unmarshal deployment: %v", err)
				}
				if deployment.Kind != "Deployment" {
					t.Fatalf("expected Deployment kind, got %q", deployment.Kind)
				}
			},
		},
		{
			name:  "svc alias",
			alias: "svc",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var service corev1.Service
				if err := yaml.Unmarshal([]byte(output), &service); err != nil {
					t.Fatalf("failed to unmarshal service: %v", err)
				}
				if service.Kind != "Service" {
					t.Fatalf("expected Service kind, got %q", service.Kind)
				}
			},
		},
		{
			name:  "pvc alias",
			alias: "pvc",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var pvc corev1.PersistentVolumeClaim
				if err := yaml.Unmarshal([]byte(output), &pvc); err != nil {
					t.Fatalf("failed to unmarshal pvc: %v", err)
				}
				if pvc.Kind != "PersistentVolumeClaim" {
					t.Fatalf("expected PersistentVolumeClaim kind, got %q", pvc.Kind)
				}
			},
		},
		{
			name:  "crd alias",
			alias: "crd",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var crd map[string]interface{}
				if err := yaml.Unmarshal([]byte(output), &crd); err != nil {
					t.Fatalf("failed to unmarshal crd: %v", err)
				}
				if kind, ok := crd["kind"].(string); !ok || kind != "CustomResourceDefinition" {
					t.Fatalf("expected CustomResourceDefinition kind, got %#v", crd["kind"])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flags := newTestFlags()
			output, err := runExample(t, []string{tt.alias}, flags)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tt.assertKind(t, output)
		})
	}
}
