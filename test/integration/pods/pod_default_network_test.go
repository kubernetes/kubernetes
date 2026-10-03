/*
Copyright The Kubernetes Authors.

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

package pods

import (
	"context"
	"fmt"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func isolatedContainer() v1.Container {
	return v1.Container{Name: "c", Image: "registry.k8s.io/pause:3.10"}
}

// TestPodDefaultNetworkCreate covers defaulting, hostNetwork mirroring and
// validation of spec.defaultNetwork on Pods with the gate enabled.
func TestPodDefaultNetworkCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDefaultNetwork, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	httpProbe := &v1.Probe{ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{Path: "/healthz", Port: intstr.FromInt32(8080)}}}
	execProbe := &v1.Probe{ProbeHandler: v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"true"}}}}

	testCases := []struct {
		name               string
		spec               v1.PodSpec
		wantErr            string
		wantDefaultNetwork v1.PodDefaultNetwork
		wantHostNetwork    bool
		wantDNSPolicy      v1.DNSPolicy
		wantServiceLinks   bool
	}{
		{
			name:               "unset defaults to Pod",
			spec:               v1.PodSpec{Containers: []v1.Container{isolatedContainer()}},
			wantDefaultNetwork: v1.PodDefaultNetworkPod,
			wantDNSPolicy:      v1.DNSClusterFirst,
			wantServiceLinks:   true,
		},
		{
			name:               "old client hostNetwork defaults to Host",
			spec:               v1.PodSpec{HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantDefaultNetwork: v1.PodDefaultNetworkHost,
			wantHostNetwork:    true,
			wantDNSPolicy:      v1.DNSClusterFirst,
			wantServiceLinks:   true,
		},
		{
			name:               "Host mirrors to hostNetwork",
			spec:               v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkHost), Containers: []v1.Container{isolatedContainer()}},
			wantDefaultNetwork: v1.PodDefaultNetworkHost,
			wantHostNetwork:    true,
			wantDNSPolicy:      v1.DNSClusterFirst,
			wantServiceLinks:   true,
		},
		{
			name:               "Pod with hostNetwork becomes Host",
			spec:               v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkPod), HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantDefaultNetwork: v1.PodDefaultNetworkHost,
			wantHostNetwork:    true,
			wantDNSPolicy:      v1.DNSClusterFirst,
			wantServiceLinks:   true,
		},
		{
			name: "None defaults dnsPolicy and enableServiceLinks",
			spec: v1.PodSpec{
				DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone),
				Containers: []v1.Container{{
					Name: "c", Image: "registry.k8s.io/pause:3.10", LivenessProbe: execProbe,
					Ports: []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}},
				}},
			},
			wantDefaultNetwork: v1.PodDefaultNetworkNone,
			wantDNSPolicy:      v1.DNSNone,
			wantServiceLinks:   false,
		},
		{
			name: "None keeps explicit dnsPolicy, dnsConfig and enableServiceLinks",
			spec: v1.PodSpec{
				DefaultNetwork:     ptr.To(v1.PodDefaultNetworkNone),
				DNSPolicy:          v1.DNSNone,
				DNSConfig:          &v1.PodDNSConfig{Nameservers: []string{"127.0.0.53"}},
				EnableServiceLinks: ptr.To(true),
				Containers:         []v1.Container{isolatedContainer()},
			},
			wantDefaultNetwork: v1.PodDefaultNetworkNone,
			wantDNSPolicy:      v1.DNSNone,
			wantServiceLinks:   true,
		},
		{
			name:    "None with hostNetwork is rejected",
			spec:    v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone), HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantErr: `must not be "None" when hostNetwork is true`,
		},
		{
			name: "None with HTTP probe is rejected",
			spec: v1.PodSpec{
				DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone),
				Containers:     []v1.Container{{Name: "c", Image: "registry.k8s.io/pause:3.10", LivenessProbe: httpProbe}},
			},
			wantErr: `spec.containers[0].livenessProbe.httpGet: Forbidden: may not be set when defaultNetwork is "None"`,
		},
		{
			name: "None with hostPort is rejected",
			spec: v1.PodSpec{
				DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone),
				Containers: []v1.Container{{
					Name: "c", Image: "registry.k8s.io/pause:3.10",
					Ports: []v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080, Protocol: v1.ProtocolTCP}},
				}},
			},
			wantErr: `spec.containers[0].ports[0].hostPort: Forbidden: may not be set when defaultNetwork is "None"`,
		},
		{
			name:    "unsupported value is rejected",
			spec:    v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetwork("Bridge")), Containers: []v1.Container{isolatedContainer()}},
			wantErr: `spec.defaultNetwork: Unsupported value: "Bridge"`,
		},
	}

	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", i), Namespace: ns.Name},
				Spec:       tc.spec,
			}
			created, err := client.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{})
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got none", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := ptr.Deref(created.Spec.DefaultNetwork, ""); got != tc.wantDefaultNetwork {
				t.Errorf("spec.defaultNetwork = %q, want %q", got, tc.wantDefaultNetwork)
			}
			if created.Spec.HostNetwork != tc.wantHostNetwork {
				t.Errorf("spec.hostNetwork = %v, want %v", created.Spec.HostNetwork, tc.wantHostNetwork)
			}
			if created.Spec.DNSPolicy != tc.wantDNSPolicy {
				t.Errorf("spec.dnsPolicy = %q, want %q", created.Spec.DNSPolicy, tc.wantDNSPolicy)
			}
			if got := ptr.Deref(created.Spec.EnableServiceLinks, false); got != tc.wantServiceLinks {
				t.Errorf("spec.enableServiceLinks = %v, want %v", got, tc.wantServiceLinks)
			}
		})
	}
}

// TestPodDefaultNetworkUpdate covers immutability of the field and the status
// invariant that isolated pods never report pod IPs.
func TestPodDefaultNetworkUpdate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDefaultNetwork, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network-update", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	pod, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "isolated", Namespace: ns.Name},
		Spec:       v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone), Containers: []v1.Container{isolatedContainer()}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}

	t.Run("defaultNetwork is immutable", func(t *testing.T) {
		update := pod.DeepCopy()
		update.Spec.DefaultNetwork = ptr.To(v1.PodDefaultNetworkPod)
		_, err := client.CoreV1().Pods(ns.Name).Update(ctx, update, metav1.UpdateOptions{})
		if err == nil || !strings.Contains(err.Error(), "pod updates may not change fields other than") {
			t.Fatalf("expected immutability error, got: %v", err)
		}
	})

	t.Run("status update with a pod IP is rejected", func(t *testing.T) {
		update := pod.DeepCopy()
		update.Status.PodIP = "10.0.0.1"
		update.Status.PodIPs = []v1.PodIP{{IP: "10.0.0.1"}}
		_, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, update, metav1.UpdateOptions{})
		if err == nil || !strings.Contains(err.Error(), `status.podIP: Forbidden: must be empty when spec.defaultNetwork is "None"`) {
			t.Fatalf("expected status validation error, got: %v", err)
		}
	})

	t.Run("status update with host IPs only is accepted", func(t *testing.T) {
		update := pod.DeepCopy()
		update.Status.Phase = v1.PodRunning
		update.Status.HostIP = "192.168.0.1"
		update.Status.HostIPs = []v1.HostIP{{IP: "192.168.0.1"}}
		if _, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, update, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("unexpected status update error: %v", err)
		}
	})
}

// TestPodDefaultNetworkWorkloadTemplates covers the field in every workload
// PodTemplateSpec: mirroring for explicit values, the None defaults, and an
// old-client PATCH of hostNetwork into a template stored with "Pod".
func TestPodDefaultNetworkWorkloadTemplates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDefaultNetwork, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network-workloads", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	labels := map[string]string{"app": "isolated"}
	template := func(defaultNetwork *v1.PodDefaultNetwork, restartPolicy v1.RestartPolicy) v1.PodTemplateSpec {
		return v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: labels},
			Spec: v1.PodSpec{
				DefaultNetwork: defaultNetwork,
				RestartPolicy:  restartPolicy,
				Containers:     []v1.Container{isolatedContainer()},
			},
		}
	}

	// create returns the stored template spec and patch applies a strategic merge
	// patch whose template spec is hostNetworkPatch.
	type workload struct {
		name             string
		create           func(name string, defaultNetwork *v1.PodDefaultNetwork) (*v1.PodSpec, error)
		patch            func(name string, patch string) (*v1.PodSpec, error)
		hostNetworkPatch string
		// immutableTemplate skips the PATCH subtests for kinds whose pod
		// template cannot be changed after creation.
		immutableTemplate bool
	}
	const templatePatch = `{"spec":{"template":{"spec":{"hostNetwork":true}}}}`
	workloads := []workload{
		{
			name:             "Deployment",
			hostNetworkPatch: templatePatch,
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().Deployments(ns.Name).Create(ctx, &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: appsv1.DeploymentSpec{
						Selector: &metav1.LabelSelector{MatchLabels: labels},
						Template: template(dn, v1.RestartPolicyAlways),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().Deployments(ns.Name).Patch(ctx, name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:             "StatefulSet",
			hostNetworkPatch: templatePatch,
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().StatefulSets(ns.Name).Create(ctx, &appsv1.StatefulSet{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: appsv1.StatefulSetSpec{
						Selector:    &metav1.LabelSelector{MatchLabels: labels},
						ServiceName: "headless",
						Template:    template(dn, v1.RestartPolicyAlways),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().StatefulSets(ns.Name).Patch(ctx, name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:             "DaemonSet",
			hostNetworkPatch: templatePatch,
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().DaemonSets(ns.Name).Create(ctx, &appsv1.DaemonSet{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: appsv1.DaemonSetSpec{
						Selector: &metav1.LabelSelector{MatchLabels: labels},
						Template: template(dn, v1.RestartPolicyAlways),
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().DaemonSets(ns.Name).Patch(ctx, name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:              "Job",
			hostNetworkPatch:  templatePatch,
			immutableTemplate: true,
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().Jobs(ns.Name).Create(ctx, &batchv1.Job{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       batchv1.JobSpec{Template: template(dn, v1.RestartPolicyNever)},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name, patch string) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().Jobs(ns.Name).Patch(ctx, name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:             "CronJob",
			hostNetworkPatch: `{"spec":{"jobTemplate":{"spec":{"template":{"spec":{"hostNetwork":true}}}}}}`,
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().CronJobs(ns.Name).Create(ctx, &batchv1.CronJob{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: batchv1.CronJobSpec{
						Schedule:    "* * * * *",
						JobTemplate: batchv1.JobTemplateSpec{Spec: batchv1.JobSpec{Template: template(dn, v1.RestartPolicyNever)}},
					},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.JobTemplate.Spec.Template.Spec, nil
			},
			patch: func(name, patch string) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().CronJobs(ns.Name).Patch(ctx, name, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.JobTemplate.Spec.Template.Spec, nil
			},
		},
	}

	for _, w := range workloads {
		prefix := strings.ToLower(w.name)
		t.Run(w.name, func(t *testing.T) {
			t.Run("unset stays unset", func(t *testing.T) {
				spec, err := w.create(prefix+"-unset", nil)
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				if spec.DefaultNetwork != nil {
					t.Errorf("template defaultNetwork = %q, want unset", *spec.DefaultNetwork)
				}
				if spec.HostNetwork {
					t.Errorf("template hostNetwork = true, want false")
				}
			})
			t.Run("Host mirrors to hostNetwork", func(t *testing.T) {
				spec, err := w.create(prefix+"-host", ptr.To(v1.PodDefaultNetworkHost))
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				if !spec.HostNetwork {
					t.Errorf("template hostNetwork = false, want true")
				}
			})
			t.Run("None defaults dnsPolicy", func(t *testing.T) {
				spec, err := w.create(prefix+"-none", ptr.To(v1.PodDefaultNetworkNone))
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				if spec.DNSPolicy != v1.DNSNone {
					t.Errorf("template dnsPolicy = %q, want %q", spec.DNSPolicy, v1.DNSNone)
				}
				if spec.EnableServiceLinks != nil {
					t.Errorf("template enableServiceLinks = %v, want unset (defaulted on Pods only)", *spec.EnableServiceLinks)
				}
			})
			t.Run("None with hostNetwork is rejected", func(t *testing.T) {
				if w.immutableTemplate {
					t.Skip("pod template is immutable")
				}
				if _, err := w.create(prefix+"-none-hostnet", ptr.To(v1.PodDefaultNetworkNone)); err != nil {
					t.Fatalf("create: %v", err)
				}
				_, err := w.patch(prefix+"-none-hostnet", w.hostNetworkPatch)
				if err == nil || !strings.Contains(err.Error(), `must not be "None" when hostNetwork is true`) {
					t.Fatalf("expected validation error, got: %v", err)
				}
			})
			t.Run("old client patches hostNetwork into a Pod template", func(t *testing.T) {
				if w.immutableTemplate {
					t.Skip("pod template is immutable")
				}
				if _, err := w.create(prefix+"-pod", ptr.To(v1.PodDefaultNetworkPod)); err != nil {
					t.Fatalf("create: %v", err)
				}
				spec, err := w.patch(prefix+"-pod", w.hostNetworkPatch)
				if err != nil {
					t.Fatalf("patch: %v", err)
				}
				if got := ptr.Deref(spec.DefaultNetwork, ""); got != v1.PodDefaultNetworkHost {
					t.Errorf("template defaultNetwork = %q, want %q", got, v1.PodDefaultNetworkHost)
				}
				if !spec.HostNetwork {
					t.Errorf("template hostNetwork = false, want true")
				}
			})
		})
	}
}

// TestPodDefaultNetworkGateDisabled covers the field being dropped on write
// while the gate is disabled and kept on objects that already carry it.
func TestPodDefaultNetworkGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDefaultNetwork, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network-gate", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	// Created while the gate is enabled.
	isolated, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "isolated", Namespace: ns.Name},
		Spec:       v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone), Containers: []v1.Container{isolatedContainer()}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDefaultNetwork, false)

	t.Run("field is dropped on create", func(t *testing.T) {
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "dropped", Namespace: ns.Name},
			Spec:       v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone), Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if created.Spec.DefaultNetwork != nil {
			t.Errorf("spec.defaultNetwork = %q, want dropped", *created.Spec.DefaultNetwork)
		}
		if created.Spec.DNSPolicy != v1.DNSClusterFirst {
			t.Errorf("spec.dnsPolicy = %q, want %q", created.Spec.DNSPolicy, v1.DNSClusterFirst)
		}
		if !ptr.Deref(created.Spec.EnableServiceLinks, false) {
			t.Errorf("spec.enableServiceLinks = %v, want true", created.Spec.EnableServiceLinks)
		}
	})

	t.Run("hostNetwork is not mirrored", func(t *testing.T) {
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "hostnet", Namespace: ns.Name},
			Spec:       v1.PodSpec{HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if created.Spec.DefaultNetwork != nil {
			t.Errorf("spec.defaultNetwork = %q, want unset", *created.Spec.DefaultNetwork)
		}
	})

	t.Run("stored field is kept and still validated", func(t *testing.T) {
		got, err := client.CoreV1().Pods(ns.Name).Get(ctx, isolated.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get pod: %v", err)
		}
		if ptr.Deref(got.Spec.DefaultNetwork, "") != v1.PodDefaultNetworkNone {
			t.Fatalf("spec.defaultNetwork = %v, want None", got.Spec.DefaultNetwork)
		}
		// Label update keeps the field.
		got.Labels = map[string]string{"updated": "true"}
		updated, err := client.CoreV1().Pods(ns.Name).Update(ctx, got, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to update pod: %v", err)
		}
		if ptr.Deref(updated.Spec.DefaultNetwork, "") != v1.PodDefaultNetworkNone {
			t.Errorf("spec.defaultNetwork = %v after update, want None", updated.Spec.DefaultNetwork)
		}
		// The status invariant does not depend on the gate.
		updated.Status.PodIP = "10.0.0.1"
		updated.Status.PodIPs = []v1.PodIP{{IP: "10.0.0.1"}}
		if _, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, updated, metav1.UpdateOptions{}); err == nil {
			t.Errorf("expected status update with a pod IP to be rejected")
		}
	})
}
