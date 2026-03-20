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
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
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

	for _, expected := range []string{"pod", "deployment", "service", "persistentvolumeclaim", "secret", "customresourcedefinition", "configmap", "job", "cronjob", "ingress", "networkpolicy", "gateway", "httproute"} {
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
		{
			name:  "cm alias",
			alias: "cm",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var cm corev1.ConfigMap
				if err := yaml.Unmarshal([]byte(output), &cm); err != nil {
					t.Fatalf("failed to unmarshal configmap: %v", err)
				}
				if cm.Kind != "ConfigMap" {
					t.Fatalf("expected ConfigMap kind, got %q", cm.Kind)
				}
			},
		},
		{
			name:  "ing alias",
			alias: "ing",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var ing networkingv1.Ingress
				if err := yaml.Unmarshal([]byte(output), &ing); err != nil {
					t.Fatalf("failed to unmarshal ingress: %v", err)
				}
				if ing.Kind != "Ingress" {
					t.Fatalf("expected Ingress kind, got %q", ing.Kind)
				}
			},
		},
		{
			name:  "netpol alias",
			alias: "netpol",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var np networkingv1.NetworkPolicy
				if err := yaml.Unmarshal([]byte(output), &np); err != nil {
					t.Fatalf("failed to unmarshal networkpolicy: %v", err)
				}
				if np.Kind != "NetworkPolicy" {
					t.Fatalf("expected NetworkPolicy kind, got %q", np.Kind)
				}
			},
		},
		{
			name:  "gtw alias",
			alias: "gtw",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var gw map[string]interface{}
				if err := yaml.Unmarshal([]byte(output), &gw); err != nil {
					t.Fatalf("failed to unmarshal gateway: %v", err)
				}
				if kind, ok := gw["kind"].(string); !ok || kind != "Gateway" {
					t.Fatalf("expected Gateway kind, got %#v", gw["kind"])
				}
			},
		},
		{
			name:  "httproute alias",
			alias: "httproute",
			assertKind: func(t *testing.T, output string) {
				t.Helper()
				var hr map[string]interface{}
				if err := yaml.Unmarshal([]byte(output), &hr); err != nil {
					t.Fatalf("failed to unmarshal httproute: %v", err)
				}
				if kind, ok := hr["kind"].(string); !ok || kind != "HTTPRoute" {
					t.Fatalf("expected HTTPRoute kind, got %#v", hr["kind"])
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

func TestExampleConfigMap(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "app-config"

	output, err := runExample(t, []string{"configmap"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var cm corev1.ConfigMap
	if err := yaml.Unmarshal([]byte(output), &cm); err != nil {
		t.Fatalf("failed to unmarshal configmap yaml: %v", err)
	}

	if cm.Kind != "ConfigMap" {
		t.Fatalf("expected kind ConfigMap, got %q", cm.Kind)
	}
	if cm.Name != "app-config" {
		t.Fatalf("expected metadata.name app-config, got %q", cm.Name)
	}
	if _, ok := cm.Data["config.yaml"]; !ok {
		t.Fatalf("expected config.yaml key in data")
	}
	if _, ok := cm.Data["LOG_LEVEL"]; !ok {
		t.Fatalf("expected LOG_LEVEL key in data")
	}
}

func TestExampleJob(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "pi-calc"
	flags.Image = "perl:5.40"

	output, err := runExample(t, []string{"job"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var job batchv1.Job
	if err := yaml.Unmarshal([]byte(output), &job); err != nil {
		t.Fatalf("failed to unmarshal job yaml: %v", err)
	}

	if job.Kind != "Job" {
		t.Fatalf("expected kind Job, got %q", job.Kind)
	}
	if job.Name != "pi-calc" {
		t.Fatalf("expected metadata.name pi-calc, got %q", job.Name)
	}
	if got := job.Spec.Template.Spec.Containers[0].Image; got != "perl:5.40" {
		t.Fatalf("expected image perl:5.40, got %q", got)
	}
	if job.Spec.BackoffLimit == nil || *job.Spec.BackoffLimit != 4 {
		t.Fatalf("expected backoffLimit 4, got %v", job.Spec.BackoffLimit)
	}
	if job.Spec.Template.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected restartPolicy Never, got %q", job.Spec.Template.Spec.RestartPolicy)
	}
}

func TestExampleCronJob(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "hello-cron"

	output, err := runExample(t, []string{"cronjob"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var cj batchv1.CronJob
	if err := yaml.Unmarshal([]byte(output), &cj); err != nil {
		t.Fatalf("failed to unmarshal cronjob yaml: %v", err)
	}

	if cj.Kind != "CronJob" {
		t.Fatalf("expected kind CronJob, got %q", cj.Kind)
	}
	if cj.Name != "hello-cron" {
		t.Fatalf("expected metadata.name hello-cron, got %q", cj.Name)
	}
	if cj.Spec.Schedule != "*/5 * * * *" {
		t.Fatalf("expected schedule */5 * * * *, got %q", cj.Spec.Schedule)
	}
	if got := cj.Spec.JobTemplate.Spec.Template.Spec.Containers[0].Image; got != "busybox:1.36" {
		t.Fatalf("expected default image busybox:1.36, got %q", got)
	}
}

func TestExampleIngress(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "web-ingress"

	output, err := runExample(t, []string{"ingress"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var ing networkingv1.Ingress
	if err := yaml.Unmarshal([]byte(output), &ing); err != nil {
		t.Fatalf("failed to unmarshal ingress yaml: %v", err)
	}

	if ing.Kind != "Ingress" {
		t.Fatalf("expected kind Ingress, got %q", ing.Kind)
	}
	if ing.Name != "web-ingress" {
		t.Fatalf("expected metadata.name web-ingress, got %q", ing.Name)
	}
	if len(ing.Spec.Rules) == 0 {
		t.Fatalf("expected at least one ingress rule")
	}
	if ing.Spec.Rules[0].Host != "example.com" {
		t.Fatalf("expected host example.com, got %q", ing.Spec.Rules[0].Host)
	}
	if _, ok := ing.Annotations["nginx.ingress.kubernetes.io/rewrite-target"]; !ok {
		t.Fatalf("expected nginx rewrite-target annotation")
	}
}

func TestExampleNetworkPolicy(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "app-netpol"

	output, err := runExample(t, []string{"networkpolicy"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var np networkingv1.NetworkPolicy
	if err := yaml.Unmarshal([]byte(output), &np); err != nil {
		t.Fatalf("failed to unmarshal networkpolicy yaml: %v", err)
	}

	if np.Kind != "NetworkPolicy" {
		t.Fatalf("expected kind NetworkPolicy, got %q", np.Kind)
	}
	if np.Name != "app-netpol" {
		t.Fatalf("expected metadata.name app-netpol, got %q", np.Name)
	}
	if len(np.Spec.PolicyTypes) != 2 {
		t.Fatalf("expected 2 policy types, got %d", len(np.Spec.PolicyTypes))
	}
	if len(np.Spec.Ingress) == 0 {
		t.Fatalf("expected at least one ingress rule")
	}
	if len(np.Spec.Egress) == 0 {
		t.Fatalf("expected at least one egress rule")
	}
}

func TestExampleGateway(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "web-gateway"

	output, err := runExample(t, []string{"gateway"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var gw map[string]interface{}
	if err := yaml.Unmarshal([]byte(output), &gw); err != nil {
		t.Fatalf("failed to unmarshal gateway yaml: %v", err)
	}

	if kind, ok := gw["kind"].(string); !ok || kind != "Gateway" {
		t.Fatalf("expected kind Gateway, got %#v", gw["kind"])
	}
	if apiVersion, ok := gw["apiVersion"].(string); !ok || apiVersion != "gateway.networking.k8s.io/v1" {
		t.Fatalf("expected apiVersion gateway.networking.k8s.io/v1, got %#v", gw["apiVersion"])
	}

	metadata, ok := gw["metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected metadata map")
	}
	if name, ok := metadata["name"].(string); !ok || name != "web-gateway" {
		t.Fatalf("expected metadata.name web-gateway, got %#v", metadata["name"])
	}

	spec, ok := gw["spec"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected spec map")
	}
	if className, ok := spec["gatewayClassName"].(string); !ok || className != "example" {
		t.Fatalf("expected gatewayClassName example, got %#v", spec["gatewayClassName"])
	}

	listeners, ok := spec["listeners"].([]interface{})
	if !ok || len(listeners) < 2 {
		t.Fatalf("expected at least 2 listeners, got %d", len(listeners))
	}
}

func TestResourceGeneratorInterface(t *testing.T) {
	for key, gen := range buildersByKind {
		var _ ResourceGenerator = gen
		out, err := gen.Generate("test-"+key, "nginx:latest", 2)
		if err != nil {
			t.Fatalf("Generate failed for key %q: %v", key, err)
		}
		if len(out) == 0 {
			t.Fatalf("Generate returned empty output for key %q", key)
		}
	}
}

func TestExampleHTTPRoute(t *testing.T) {
	flags := newTestFlags()
	flags.Name = "web-route"

	output, err := runExample(t, []string{"httproute"}, flags)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var hr map[string]interface{}
	if err := yaml.Unmarshal([]byte(output), &hr); err != nil {
		t.Fatalf("failed to unmarshal httproute yaml: %v", err)
	}

	if kind, ok := hr["kind"].(string); !ok || kind != "HTTPRoute" {
		t.Fatalf("expected kind HTTPRoute, got %#v", hr["kind"])
	}
	if apiVersion, ok := hr["apiVersion"].(string); !ok || apiVersion != "gateway.networking.k8s.io/v1" {
		t.Fatalf("expected apiVersion gateway.networking.k8s.io/v1, got %#v", hr["apiVersion"])
	}

	metadata, ok := hr["metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected metadata map")
	}
	if name, ok := metadata["name"].(string); !ok || name != "web-route" {
		t.Fatalf("expected metadata.name web-route, got %#v", metadata["name"])
	}

	spec, ok := hr["spec"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected spec map")
	}

	parentRefs, ok := spec["parentRefs"].([]interface{})
	if !ok || len(parentRefs) == 0 {
		t.Fatalf("expected at least one parentRef")
	}

	rules, ok := spec["rules"].([]interface{})
	if !ok || len(rules) < 2 {
		t.Fatalf("expected at least 2 routing rules, got %d", len(rules))
	}

	hostnames, ok := spec["hostnames"].([]interface{})
	if !ok || len(hostnames) == 0 {
		t.Fatalf("expected at least one hostname")
	}
}
