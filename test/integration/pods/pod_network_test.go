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
	"slices"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	appsv1ac "k8s.io/client-go/applyconfigurations/apps/v1"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

const gateDisabledErr = "spec.podNetwork: Forbidden: the NetworkIsolatedPods feature gate is disabled"

func isolatedContainer() v1.Container {
	return v1.Container{
		Name: "c", Image: "registry.k8s.io/pause:3.10",
		Ports: []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}},
	}
}

// TestPodNetworkCreate covers defaulting and validation of spec.podNetwork on
// Pods with the gate enabled.
func TestPodNetworkCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-network", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	httpProbe := &v1.Probe{ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{Path: "/healthz", Port: intstr.FromInt32(8080)}}}
	execProbe := &v1.Probe{ProbeHandler: v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"true"}}}}

	testCases := []struct {
		name             string
		spec             v1.PodSpec
		wantErr          string
		wantPodNetwork   *v1.PodNetwork
		wantDNSPolicy    v1.DNSPolicy
		wantServiceLinks bool
	}{
		{
			name:             "unset stays unset",
			spec:             v1.PodSpec{Containers: []v1.Container{isolatedContainer()}},
			wantDNSPolicy:    v1.DNSClusterFirst,
			wantServiceLinks: true,
		},
		{
			name:             "hostNetwork alone does not set the field",
			spec:             v1.PodSpec{HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantDNSPolicy:    v1.DNSClusterFirst,
			wantServiceLinks: true,
		},
		{
			name:             "Default stays Default",
			spec:             v1.PodSpec{PodNetwork: new(v1.PodNetworkDefault), Containers: []v1.Container{isolatedContainer()}},
			wantPodNetwork:   new(v1.PodNetworkDefault),
			wantDNSPolicy:    v1.DNSClusterFirst,
			wantServiceLinks: true,
		},
		{
			name:    "Default with hostNetwork is rejected",
			spec:    v1.PodSpec{PodNetwork: new(v1.PodNetworkDefault), HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantErr: "spec.podNetwork: Forbidden: may not be set when hostNetwork is true",
		},
		{
			name: "None defaults dnsPolicy and enableServiceLinks",
			spec: v1.PodSpec{
				PodNetwork: new(v1.PodNetworkNone),
				Containers: []v1.Container{{
					Name: "c", Image: "registry.k8s.io/pause:3.10", LivenessProbe: execProbe,
					Ports: []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}},
				}},
			},
			wantPodNetwork:   new(v1.PodNetworkNone),
			wantDNSPolicy:    v1.DNSNone,
			wantServiceLinks: false,
		},
		{
			name: "None keeps explicit dnsPolicy, dnsConfig and enableServiceLinks",
			spec: v1.PodSpec{
				PodNetwork:         new(v1.PodNetworkNone),
				DNSPolicy:          v1.DNSNone,
				DNSConfig:          &v1.PodDNSConfig{Nameservers: []string{"127.0.0.53"}},
				EnableServiceLinks: new(true),
				Containers:         []v1.Container{isolatedContainer()},
			},
			wantPodNetwork:   new(v1.PodNetworkNone),
			wantDNSPolicy:    v1.DNSNone,
			wantServiceLinks: true,
		},
		{
			name:    "None with hostNetwork is rejected",
			spec:    v1.PodSpec{PodNetwork: new(v1.PodNetworkNone), HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantErr: "spec.podNetwork: Forbidden: may not be set when hostNetwork is true",
		},
		{
			name: "None with HTTP probe is rejected",
			spec: v1.PodSpec{
				PodNetwork: new(v1.PodNetworkNone),
				Containers: []v1.Container{{Name: "c", Image: "registry.k8s.io/pause:3.10", LivenessProbe: httpProbe}},
			},
			wantErr: `spec.containers[0].livenessProbe.httpGet: Forbidden: may not be set when podNetwork is "None"`,
		},
		{
			name: "None with hostPort is rejected",
			spec: v1.PodSpec{
				PodNetwork: new(v1.PodNetworkNone),
				Containers: []v1.Container{{
					Name: "c", Image: "registry.k8s.io/pause:3.10",
					Ports: []v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080, Protocol: v1.ProtocolTCP}},
				}},
			},
			wantErr: `spec.containers[0].ports[0].hostPort: Forbidden: may not be set when podNetwork is "None"`,
		},
		{
			name:    "unsupported value is rejected",
			spec:    v1.PodSpec{PodNetwork: new(v1.PodNetwork("Bridge")), Containers: []v1.Container{isolatedContainer()}},
			wantErr: `spec.podNetwork: Unsupported value: "Bridge"`,
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
			if got, want := ptr.Deref(created.Spec.PodNetwork, "<unset>"), ptr.Deref(tc.wantPodNetwork, "<unset>"); got != want {
				t.Errorf("spec.podNetwork = %q, want %q", got, want)
			}
			if created.Spec.HostNetwork != tc.spec.HostNetwork {
				t.Errorf("spec.hostNetwork = %v, want %v", created.Spec.HostNetwork, tc.spec.HostNetwork)
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

// TestPodNetworkUpdate covers immutability of the field, full updates from
// clients that do not know the field, and the status invariant that isolated
// pods never report pod IPs.
func TestPodNetworkUpdate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-network-update", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	create := func(name string, podNetwork v1.PodNetwork) *v1.Pod {
		t.Helper()
		pod, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns.Name},
			Spec:       v1.PodSpec{PodNetwork: &podNetwork, Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod %s: %v", name, err)
		}
		return pod
	}
	isolated := create("isolated", v1.PodNetworkNone)
	attached := create("attached", v1.PodNetworkDefault)

	t.Run("podNetwork is immutable", func(t *testing.T) {
		update := isolated.DeepCopy()
		update.Spec.PodNetwork = new(v1.PodNetworkDefault)
		_, err := client.CoreV1().Pods(ns.Name).Update(ctx, update, metav1.UpdateOptions{})
		if err == nil || !strings.Contains(err.Error(), "pod updates may not change fields other than") {
			t.Fatalf("expected immutability error, got: %v", err)
		}
	})

	// A client built before the field existed sends the whole object back
	// without it. The stored value is kept instead of failing immutability.
	for _, tc := range []struct {
		name string
		pod  *v1.Pod
		want v1.PodNetwork
	}{
		{name: "old client full update keeps None", pod: isolated, want: v1.PodNetworkNone},
		{name: "old client full update keeps Default", pod: attached, want: v1.PodNetworkDefault},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := client.CoreV1().Pods(ns.Name).Get(ctx, tc.pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get: %v", err)
			}
			got.Spec.PodNetwork = nil
			got.Labels = map[string]string{"updated": "true"}
			updated, err := client.CoreV1().Pods(ns.Name).Update(ctx, got, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("update without the field: %v", err)
			}
			if got := ptr.Deref(updated.Spec.PodNetwork, "<unset>"); got != tc.want {
				t.Errorf("spec.podNetwork = %q after update, want %q", got, tc.want)
			}
		})
	}

	latest := func(t *testing.T) *v1.Pod {
		t.Helper()
		pod, err := client.CoreV1().Pods(ns.Name).Get(ctx, isolated.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("get: %v", err)
		}
		return pod
	}

	t.Run("status update with a pod IP is rejected", func(t *testing.T) {
		update := latest(t)
		update.Status.PodIP = "10.0.0.1"
		update.Status.PodIPs = []v1.PodIP{{IP: "10.0.0.1"}}
		_, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, update, metav1.UpdateOptions{})
		if err == nil || !strings.Contains(err.Error(), `status.podIP: Forbidden: must be unset when spec.podNetwork is "None"`) {
			t.Fatalf("expected status validation error, got: %v", err)
		}
	})

	t.Run("status update with host IPs only is accepted", func(t *testing.T) {
		update := latest(t)
		update.Status.Phase = v1.PodRunning
		update.Status.HostIP = "192.168.0.1"
		update.Status.HostIPs = []v1.HostIP{{IP: "192.168.0.1"}}
		if _, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, update, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("unexpected status update error: %v", err)
		}
	})
}

// templateWorkload drives one kind that embeds a PodTemplateSpec.
type templateWorkload struct {
	name string
	// create stores an object whose template spec has the given podNetwork and
	// returns the stored template spec.
	create func(name string, podNetwork *v1.PodNetwork) (*v1.PodSpec, error)
	// patch applies the patch of the given type to the object and returns the
	// stored template spec. nil for kinds whose template is immutable.
	patch func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error)
	// templatePath is the JSON path prefix of the pod template spec, used to
	// build patches.
	templatePath []string
}

// templatePatch nests body under the workload's template spec path.
func (w templateWorkload) templatePatch(body string) string {
	for _, key := range slices.Backward(w.templatePath) {
		body = fmt.Sprintf(`{%q:%s}`, key, body)
	}
	return body
}

func templateWorkloads(ctx context.Context, client clientset.Interface, ns string) []templateWorkload {
	labels := map[string]string{"app": "isolated"}
	template := func(podNetwork *v1.PodNetwork, restartPolicy v1.RestartPolicy) v1.PodTemplateSpec {
		return v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{Labels: labels},
			Spec: v1.PodSpec{
				PodNetwork:    podNetwork,
				RestartPolicy: restartPolicy,
				Containers:    []v1.Container{isolatedContainer()},
			},
		}
	}
	selector := &metav1.LabelSelector{MatchLabels: labels}
	specTemplatePath := []string{"spec", "template", "spec"}

	return []templateWorkload{
		{
			name:         "Deployment",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().Deployments(ns).Create(ctx, &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       appsv1.DeploymentSpec{Selector: selector, Template: template(dn, v1.RestartPolicyAlways)},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().Deployments(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:         "ReplicaSet",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().ReplicaSets(ns).Create(ctx, &appsv1.ReplicaSet{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       appsv1.ReplicaSetSpec{Selector: selector, Template: template(dn, v1.RestartPolicyAlways)},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().ReplicaSets(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:         "StatefulSet",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().StatefulSets(ns).Create(ctx, &appsv1.StatefulSet{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       appsv1.StatefulSetSpec{Selector: selector, ServiceName: "headless", Template: template(dn, v1.RestartPolicyAlways)},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().StatefulSets(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:         "DaemonSet",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().DaemonSets(ns).Create(ctx, &appsv1.DaemonSet{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       appsv1.DaemonSetSpec{Selector: selector, Template: template(dn, v1.RestartPolicyAlways)},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.AppsV1().DaemonSets(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:         "ReplicationController",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				tmpl := template(dn, v1.RestartPolicyAlways)
				obj, err := client.CoreV1().ReplicationControllers(ns).Create(ctx, &v1.ReplicationController{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       v1.ReplicationControllerSpec{Selector: labels, Template: &tmpl},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.CoreV1().ReplicationControllers(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:         "PodTemplate",
			templatePath: []string{"template", "spec"},
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.CoreV1().PodTemplates(ns).Create(ctx, &v1.PodTemplate{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Template:   template(dn, v1.RestartPolicyAlways),
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Template.Spec, nil
			},
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.CoreV1().PodTemplates(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Template.Spec, nil
			},
		},
		{
			// The Job pod template is immutable, so only creation is covered.
			name:         "Job",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().Jobs(ns).Create(ctx, &batchv1.Job{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec:       batchv1.JobSpec{Template: template(dn, v1.RestartPolicyNever)},
				}, metav1.CreateOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.Template.Spec, nil
			},
		},
		{
			name:         "CronJob",
			templatePath: []string{"spec", "jobTemplate", "spec", "template", "spec"},
			create: func(name string, dn *v1.PodNetwork) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().CronJobs(ns).Create(ctx, &batchv1.CronJob{
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
			patch: func(name string, pt types.PatchType, patch string) (*v1.PodSpec, error) {
				obj, err := client.BatchV1().CronJobs(ns).Patch(ctx, name, pt, []byte(patch), metav1.PatchOptions{})
				if err != nil {
					return nil, err
				}
				return &obj.Spec.JobTemplate.Spec.Template.Spec, nil
			},
		},
	}
}

func expectTemplate(t *testing.T, spec *v1.PodSpec, wantPodNetwork *v1.PodNetwork, wantHostNetwork bool) {
	t.Helper()
	if got, want := ptr.Deref(spec.PodNetwork, "<unset>"), ptr.Deref(wantPodNetwork, "<unset>"); got != want {
		t.Errorf("template podNetwork = %q, want %q", got, want)
	}
	if spec.HostNetwork != wantHostNetwork {
		t.Errorf("template hostNetwork = %v, want %v", spec.HostNetwork, wantHostNetwork)
	}
}

// TestPodNetworkWorkloadTemplates covers the field in every resource that
// embeds a PodTemplateSpec: the None defaults on create, and the registry
// strategies keeping the field for clients that do not know it, including the
// one that turns hostNetwork on for an isolated template.
func TestPodNetworkWorkloadTemplates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-network-workloads", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	const (
		hostNetworkTrue  = `{"hostNetwork":true}`
		hostNetworkFalse = `{"hostNetwork":false}`
		podNetworkNone   = `{"podNetwork":"None"}`
		podNetworkUnset  = `{"podNetwork":null}`
		// A full update from a client built before the field: it sends
		// hostNetwork and no podNetwork at all.
		oldClientHostNetwork = `{"podNetwork":null,"hostNetwork":true}`
	)

	for _, w := range templateWorkloads(ctx, client, ns.Name) {
		prefix := strings.ToLower(w.name)
		t.Run(w.name, func(t *testing.T) {
			t.Run("unset stays unset", func(t *testing.T) {
				spec, err := w.create(prefix+"-unset", nil)
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				expectTemplate(t, spec, nil, false)
			})
			t.Run("None defaults dnsPolicy", func(t *testing.T) {
				spec, err := w.create(prefix+"-none", new(v1.PodNetworkNone))
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				expectTemplate(t, spec, new(v1.PodNetworkNone), false)
				if spec.DNSPolicy != v1.DNSNone {
					t.Errorf("template dnsPolicy = %q, want %q", spec.DNSPolicy, v1.DNSNone)
				}
				if spec.EnableServiceLinks != nil {
					t.Errorf("template enableServiceLinks = %v, want unset (defaulted on Pods only)", *spec.EnableServiceLinks)
				}
			})
			if w.patch == nil {
				return
			}

			// The stored object is created with "from" (and hostNetwork
			// fromHostNetwork) and the strategic merge patch sets body on
			// the template spec.
			for i, tc := range []struct {
				name            string
				from            *v1.PodNetwork
				fromHostNetwork bool
				body            string
				wantErr         string
				wantPodNetwork  *v1.PodNetwork
				wantHostNetwork bool
			}{
				{
					name: "old client clears the field on a None template",
					from: new(v1.PodNetworkNone), body: podNetworkUnset,
					wantPodNetwork: new(v1.PodNetworkNone),
				},
				{
					name: "old client turns hostNetwork on for a None template",
					from: new(v1.PodNetworkNone), body: hostNetworkTrue,
					wantHostNetwork: true,
				},
				{
					name: "old client full update turns hostNetwork on for a None template",
					from: new(v1.PodNetworkNone), body: oldClientHostNetwork,
					wantHostNetwork: true,
				},
				{
					name: "old client turns hostNetwork on for a Default template",
					from: new(v1.PodNetworkDefault), body: hostNetworkTrue,
					wantHostNetwork: true,
				},
				{
					name: "old client turns hostNetwork off for a None template",
					from: new(v1.PodNetworkNone), fromHostNetwork: false, body: hostNetworkFalse,
					wantPodNetwork: new(v1.PodNetworkNone),
				},
				{
					name:            "new client sets None on a hostNetwork template",
					fromHostNetwork: true, body: podNetworkNone,
					wantErr: "may not be set when hostNetwork is true",
				},
				{
					name:            "new client sets Default on a hostNetwork template",
					fromHostNetwork: true, body: `{"podNetwork":"Default"}`,
					wantErr: "may not be set when hostNetwork is true",
				},
				{
					name:    "new client sets None and hostNetwork in one request",
					body:    `{"podNetwork":"None","hostNetwork":true}`,
					wantErr: "may not be set when hostNetwork is true",
				},
				{
					name:    "new client sets Default and hostNetwork in one request",
					body:    `{"podNetwork":"Default","hostNetwork":true}`,
					wantErr: "may not be set when hostNetwork is true",
				},
				{
					// Indistinguishable from an old-client PATCH of hostNetwork.
					name: "None sent again while turning hostNetwork on",
					from: new(v1.PodNetworkNone), body: `{"podNetwork":"None","hostNetwork":true}`,
					wantHostNetwork: true,
				},
				{
					name: "old client clears the field on a Default template",
					from: new(v1.PodNetworkDefault), body: podNetworkUnset,
					wantPodNetwork: new(v1.PodNetworkDefault),
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					name := fmt.Sprintf("%s-update-%d", prefix, i)
					if _, err := w.create(name, tc.from); err != nil {
						t.Fatalf("create: %v", err)
					}
					if tc.fromHostNetwork {
						if _, err := w.patch(name, types.StrategicMergePatchType, w.templatePatch(hostNetworkTrue)); err != nil {
							t.Fatalf("patch hostNetwork: %v", err)
						}
					}
					spec, err := w.patch(name, types.StrategicMergePatchType, w.templatePatch(tc.body))
					if tc.wantErr != "" {
						if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
							t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
						}
						return
					}
					if err != nil {
						t.Fatalf("patch: %v", err)
					}
					expectTemplate(t, spec, tc.wantPodNetwork, tc.wantHostNetwork)
					// Leaving "None" also undoes the dnsPolicy default it implied.
					wantDNSPolicy := v1.DNSClusterFirst
					if ptr.Deref(tc.wantPodNetwork, "") == v1.PodNetworkNone {
						wantDNSPolicy = v1.DNSNone
					}
					if spec.DNSPolicy != wantDNSPolicy {
						t.Errorf("template dnsPolicy = %q, want %q", spec.DNSPolicy, wantDNSPolicy)
					}
				})
			}
		})
	}

	// Server-side apply: a manager that applies podNetwork "None" and later
	// "Default" gets what it asked for. Dropping the field from the applied
	// configuration does not remove it, because an omitted field on update
	// cannot be told apart from a client that never knew it. Isolation is
	// turned off by applying "Default".
	t.Run("Deployment/server-side apply", func(t *testing.T) {
		const manager = "test-manager"
		apply := func(pn *v1.PodNetwork) *v1.PodSpec {
			t.Helper()
			podSpec := corev1ac.PodSpec().
				WithContainers(corev1ac.Container().WithName("c").WithImage("registry.k8s.io/pause:3.10"))
			if pn != nil {
				podSpec = podSpec.WithPodNetwork(*pn)
			}
			ac := appsv1ac.Deployment("ssa", ns.Name).WithSpec(appsv1ac.DeploymentSpec().
				WithSelector(metav1ac.LabelSelector().WithMatchLabels(map[string]string{"app": "ssa"})).
				WithTemplate(corev1ac.PodTemplateSpec().
					WithLabels(map[string]string{"app": "ssa"}).
					WithSpec(podSpec)))
			obj, err := client.AppsV1().Deployments(ns.Name).Apply(ctx, ac, metav1.ApplyOptions{FieldManager: manager, Force: true})
			if err != nil {
				t.Fatalf("apply %v: %v", ptr.Deref(pn, "<unset>"), err)
			}
			return &obj.Spec.Template.Spec
		}
		expectTemplate(t, apply(new(v1.PodNetworkNone)), new(v1.PodNetworkNone), false)
		expectTemplate(t, apply(new(v1.PodNetworkDefault)), new(v1.PodNetworkDefault), false)
		expectTemplate(t, apply(nil), new(v1.PodNetworkDefault), false)
		expectTemplate(t, apply(new(v1.PodNetworkNone)), new(v1.PodNetworkNone), false)
		expectTemplate(t, apply(nil), new(v1.PodNetworkNone), false)
	})
}

// TestPodNetworkGateDisabled covers the field being rejected on new objects
// while the gate is disabled, and kept and still validated on objects that
// already carry it.
func TestPodNetworkGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-network-gate", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	// Created while the gate is enabled.
	isolated, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "isolated", Namespace: ns.Name},
		Spec:       v1.PodSpec{PodNetwork: new(v1.PodNetworkNone), Containers: []v1.Container{isolatedContainer()}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}
	deployment := templateWorkloads(ctx, client, ns.Name)[0]
	if _, err := deployment.create("isolated", new(v1.PodNetworkNone)); err != nil {
		t.Fatalf("failed to create deployment: %v", err)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, false)

	// This is also what happens to pods that a controller creates from a
	// template stored with "None": they are rejected instead of being
	// attached to the pod network.
	t.Run("None is rejected on create", func(t *testing.T) {
		_, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "new-none", Namespace: ns.Name},
			Spec:       v1.PodSpec{PodNetwork: new(v1.PodNetworkNone), Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err == nil || !strings.Contains(err.Error(), gateDisabledErr) {
			t.Fatalf("expected error containing %q, got: %v", gateDisabledErr, err)
		}
		_, err = deployment.create("new-none", new(v1.PodNetworkNone))
		if err == nil || !strings.Contains(err.Error(), gateDisabledErr) {
			t.Fatalf("expected deployment error containing %q, got: %v", gateDisabledErr, err)
		}
	})

	// "Default" means the same as unset, so it is dropped like any other
	// disabled field. This is what lets a workload stored with "None" be
	// moved back to the pod network while the gate is off.
	t.Run("Default is dropped on create", func(t *testing.T) {
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "new-default", Namespace: ns.Name},
			Spec:       v1.PodSpec{PodNetwork: new(v1.PodNetworkDefault), Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if created.Spec.PodNetwork != nil {
			t.Errorf("spec.podNetwork = %q, want dropped", *created.Spec.PodNetwork)
		}
		spec, err := deployment.create("new-default", new(v1.PodNetworkDefault))
		if err != nil {
			t.Fatalf("failed to create deployment: %v", err)
		}
		expectTemplate(t, spec, nil, false)
	})

	t.Run("hostNetwork works without the field", func(t *testing.T) {
		created, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "hostnet", Namespace: ns.Name},
			Spec:       v1.PodSpec{HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod: %v", err)
		}
		if created.Spec.PodNetwork != nil {
			t.Errorf("spec.podNetwork = %q, want unset", *created.Spec.PodNetwork)
		}
	})

	t.Run("stored field is kept and still validated", func(t *testing.T) {
		got, err := client.CoreV1().Pods(ns.Name).Get(ctx, isolated.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to get pod: %v", err)
		}
		if ptr.Deref(got.Spec.PodNetwork, "") != v1.PodNetworkNone {
			t.Fatalf("spec.podNetwork = %v, want None", got.Spec.PodNetwork)
		}
		// Label update keeps the field.
		got.Labels = map[string]string{"updated": "true"}
		updated, err := client.CoreV1().Pods(ns.Name).Update(ctx, got, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to update pod: %v", err)
		}
		if ptr.Deref(updated.Spec.PodNetwork, "") != v1.PodNetworkNone {
			t.Errorf("spec.podNetwork = %v after update, want None", updated.Spec.PodNetwork)
		}
		// So does a full update from a client that does not know the field.
		updated.Spec.PodNetwork = nil
		updated.Labels["updated"] = "twice"
		updated, err = client.CoreV1().Pods(ns.Name).Update(ctx, updated, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to update pod without the field: %v", err)
		}
		if ptr.Deref(updated.Spec.PodNetwork, "") != v1.PodNetworkNone {
			t.Errorf("spec.podNetwork = %v after update without the field, want None", updated.Spec.PodNetwork)
		}
		// The status invariant does not depend on the gate.
		updated.Status.PodIP = "10.0.0.1"
		updated.Status.PodIPs = []v1.PodIP{{IP: "10.0.0.1"}}
		if _, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, updated, metav1.UpdateOptions{}); err == nil {
			t.Errorf("expected status update with a pod IP to be rejected")
		}
	})

	t.Run("stored template keeps the field and can be moved back to the pod network", func(t *testing.T) {
		spec, err := deployment.patch("isolated", types.StrategicMergePatchType, `{"metadata":{"labels":{"updated":"true"}}}`)
		if err != nil {
			t.Fatalf("patch: %v", err)
		}
		expectTemplate(t, spec, new(v1.PodNetworkNone), false)
		spec, err = deployment.patch("isolated", types.StrategicMergePatchType, deployment.templatePatch(`{"podNetwork":null}`))
		if err != nil {
			t.Fatalf("patch clearing the field: %v", err)
		}
		expectTemplate(t, spec, new(v1.PodNetworkNone), false)

		// The way out while the gate is off: "Default" is accepted on the
		// template and the pods the controller creates from it have the
		// field dropped.
		spec, err = deployment.patch("isolated", types.StrategicMergePatchType, deployment.templatePatch(`{"podNetwork":"Default"}`))
		if err != nil {
			t.Fatalf("patch to Default: %v", err)
		}
		expectTemplate(t, spec, new(v1.PodNetworkDefault), false)
		if spec.DNSPolicy != v1.DNSClusterFirst {
			t.Errorf("template dnsPolicy = %q, want %q", spec.DNSPolicy, v1.DNSClusterFirst)
		}
	})
}
