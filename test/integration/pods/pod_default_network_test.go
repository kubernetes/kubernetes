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

const gateDisabledErr = "spec.defaultNetwork: Forbidden: the NetworkIsolatedPods feature gate is disabled"

// isolatedContainer has a containerPort without a hostPort so that the tests
// can observe the hostPort defaulting that hostNetwork pods get.
func isolatedContainer() v1.Container {
	return v1.Container{
		Name: "c", Image: "registry.k8s.io/pause:3.10",
		Ports: []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}},
	}
}

// TestPodDefaultNetworkCreate covers defaulting and validation of
// spec.defaultNetwork on Pods with the gate enabled. The mirror between the
// two fields is one-way: "Host" sets hostNetwork, but hostNetwork alone does
// not set defaultNetwork, so pods from clients that do not know the field are
// stored unchanged.
func TestPodDefaultNetworkCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

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
		wantDefaultNetwork *v1.PodDefaultNetwork
		wantHostNetwork    bool
		wantDNSPolicy      v1.DNSPolicy
		wantServiceLinks   bool
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
			wantHostNetwork:  true,
			wantDNSPolicy:    v1.DNSClusterFirst,
			wantServiceLinks: true,
		},
		{
			name:               "Host mirrors to hostNetwork and defaults hostPort",
			spec:               v1.PodSpec{DefaultNetwork: new(v1.PodDefaultNetworkHost), Containers: []v1.Container{isolatedContainer()}},
			wantDefaultNetwork: new(v1.PodDefaultNetworkHost),
			wantHostNetwork:    true,
			wantDNSPolicy:      v1.DNSClusterFirst,
			wantServiceLinks:   true,
		},
		{
			name:               "Pod stays Pod",
			spec:               v1.PodSpec{DefaultNetwork: new(v1.PodDefaultNetworkPod), Containers: []v1.Container{isolatedContainer()}},
			wantDefaultNetwork: new(v1.PodDefaultNetworkPod),
			wantDNSPolicy:      v1.DNSClusterFirst,
			wantServiceLinks:   true,
		},
		{
			name:    "Pod with hostNetwork is rejected",
			spec:    v1.PodSpec{DefaultNetwork: new(v1.PodDefaultNetworkPod), HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantErr: `must be "Host" when hostNetwork is true`,
		},
		{
			name: "None defaults dnsPolicy and enableServiceLinks",
			spec: v1.PodSpec{
				DefaultNetwork: new(v1.PodDefaultNetworkNone),
				Containers: []v1.Container{{
					Name: "c", Image: "registry.k8s.io/pause:3.10", LivenessProbe: execProbe,
					Ports: []v1.ContainerPort{{ContainerPort: 8080, Protocol: v1.ProtocolTCP}},
				}},
			},
			wantDefaultNetwork: new(v1.PodDefaultNetworkNone),
			wantDNSPolicy:      v1.DNSNone,
			wantServiceLinks:   false,
		},
		{
			name: "None keeps explicit dnsPolicy, dnsConfig and enableServiceLinks",
			spec: v1.PodSpec{
				DefaultNetwork:     new(v1.PodDefaultNetworkNone),
				DNSPolicy:          v1.DNSNone,
				DNSConfig:          &v1.PodDNSConfig{Nameservers: []string{"127.0.0.53"}},
				EnableServiceLinks: new(true),
				Containers:         []v1.Container{isolatedContainer()},
			},
			wantDefaultNetwork: new(v1.PodDefaultNetworkNone),
			wantDNSPolicy:      v1.DNSNone,
			wantServiceLinks:   true,
		},
		{
			name:    "None with hostNetwork is rejected",
			spec:    v1.PodSpec{DefaultNetwork: new(v1.PodDefaultNetworkNone), HostNetwork: true, Containers: []v1.Container{isolatedContainer()}},
			wantErr: `must not be "None" when hostNetwork is true`,
		},
		{
			name: "None with HTTP probe is rejected",
			spec: v1.PodSpec{
				DefaultNetwork: new(v1.PodDefaultNetworkNone),
				Containers:     []v1.Container{{Name: "c", Image: "registry.k8s.io/pause:3.10", LivenessProbe: httpProbe}},
			},
			wantErr: `spec.containers[0].livenessProbe.httpGet: Forbidden: may not be set when defaultNetwork is "None"`,
		},
		{
			name: "None with hostPort is rejected",
			spec: v1.PodSpec{
				DefaultNetwork: new(v1.PodDefaultNetworkNone),
				Containers: []v1.Container{{
					Name: "c", Image: "registry.k8s.io/pause:3.10",
					Ports: []v1.ContainerPort{{ContainerPort: 8080, HostPort: 8080, Protocol: v1.ProtocolTCP}},
				}},
			},
			wantErr: `spec.containers[0].ports[0].hostPort: Forbidden: may not be set when defaultNetwork is "None"`,
		},
		{
			name:    "unsupported value is rejected",
			spec:    v1.PodSpec{DefaultNetwork: new(v1.PodDefaultNetwork("Bridge")), Containers: []v1.Container{isolatedContainer()}},
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
			if got, want := ptr.Deref(created.Spec.DefaultNetwork, "<unset>"), ptr.Deref(tc.wantDefaultNetwork, "<unset>"); got != want {
				t.Errorf("spec.defaultNetwork = %q, want %q", got, want)
			}
			if created.Spec.HostNetwork != tc.wantHostNetwork {
				t.Errorf("spec.hostNetwork = %v, want %v", created.Spec.HostNetwork, tc.wantHostNetwork)
			}
			wantHostPort := int32(0)
			if tc.wantHostNetwork {
				wantHostPort = 8080
			}
			if got := created.Spec.Containers[0].Ports[0].HostPort; got != wantHostPort {
				t.Errorf("spec.containers[0].ports[0].hostPort = %d, want %d", got, wantHostPort)
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

// TestPodDefaultNetworkUpdate covers immutability of the field, full updates
// from clients that do not know the field, and the status invariant that
// isolated pods never report pod IPs.
func TestPodDefaultNetworkUpdate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network-update", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	create := func(name string, defaultNetwork v1.PodDefaultNetwork) *v1.Pod {
		t.Helper()
		pod, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns.Name},
			Spec:       v1.PodSpec{DefaultNetwork: &defaultNetwork, Containers: []v1.Container{isolatedContainer()}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("failed to create pod %s: %v", name, err)
		}
		return pod
	}
	isolated := create("isolated", v1.PodDefaultNetworkNone)
	host := create("host", v1.PodDefaultNetworkHost)

	t.Run("defaultNetwork is immutable", func(t *testing.T) {
		update := isolated.DeepCopy()
		update.Spec.DefaultNetwork = new(v1.PodDefaultNetworkPod)
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
		want v1.PodDefaultNetwork
	}{
		{name: "old client full update keeps None", pod: isolated, want: v1.PodDefaultNetworkNone},
		{name: "old client full update keeps Host", pod: host, want: v1.PodDefaultNetworkHost},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := client.CoreV1().Pods(ns.Name).Get(ctx, tc.pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("get: %v", err)
			}
			got.Spec.DefaultNetwork = nil
			got.Labels = map[string]string{"updated": "true"}
			updated, err := client.CoreV1().Pods(ns.Name).Update(ctx, got, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("update without the field: %v", err)
			}
			if got := ptr.Deref(updated.Spec.DefaultNetwork, "<unset>"); got != tc.want {
				t.Errorf("spec.defaultNetwork = %q after update, want %q", got, tc.want)
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
		if err == nil || !strings.Contains(err.Error(), `status.podIP: Forbidden: must be unset when spec.defaultNetwork is "None"`) {
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
	// create stores an object whose template spec has the given defaultNetwork
	// and returns the stored template spec.
	create func(name string, defaultNetwork *v1.PodDefaultNetwork) (*v1.PodSpec, error)
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
	selector := &metav1.LabelSelector{MatchLabels: labels}
	specTemplatePath := []string{"spec", "template", "spec"}

	return []templateWorkload{
		{
			name:         "Deployment",
			templatePath: specTemplatePath,
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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
			create: func(name string, dn *v1.PodDefaultNetwork) (*v1.PodSpec, error) {
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

func expectTemplate(t *testing.T, spec *v1.PodSpec, wantDefaultNetwork *v1.PodDefaultNetwork, wantHostNetwork bool) {
	t.Helper()
	if got, want := ptr.Deref(spec.DefaultNetwork, "<unset>"), ptr.Deref(wantDefaultNetwork, "<unset>"); got != want {
		t.Errorf("template defaultNetwork = %q, want %q", got, want)
	}
	if spec.HostNetwork != wantHostNetwork {
		t.Errorf("template hostNetwork = %v, want %v", spec.HostNetwork, wantHostNetwork)
	}
}

// TestPodDefaultNetworkWorkloadTemplates covers the field in every resource
// that embeds a PodTemplateSpec: mirroring on create, the None defaults, and
// the registry strategies keeping defaultNetwork and hostNetwork consistent
// when an update changes only one of them.
func TestPodDefaultNetworkWorkloadTemplates(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network-workloads", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	const (
		hostNetworkTrue  = `{"hostNetwork":true}`
		hostNetworkFalse = `{"hostNetwork":false}`
		defaultNetPod    = `{"defaultNetwork":"Pod"}`
		defaultNetNone   = `{"defaultNetwork":"None"}`
		defaultNetUnset  = `{"defaultNetwork":null}`
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
			t.Run("Host mirrors to hostNetwork", func(t *testing.T) {
				spec, err := w.create(prefix+"-host", new(v1.PodDefaultNetworkHost))
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				expectTemplate(t, spec, new(v1.PodDefaultNetworkHost), true)
			})
			t.Run("None defaults dnsPolicy", func(t *testing.T) {
				spec, err := w.create(prefix+"-none", new(v1.PodDefaultNetworkNone))
				if err != nil {
					t.Fatalf("create: %v", err)
				}
				expectTemplate(t, spec, new(v1.PodDefaultNetworkNone), false)
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

			// Updates that change only one of the two fields. The stored
			// object is created with "from" and the strategic merge patch
			// sets body on the template spec.
			for i, tc := range []struct {
				name               string
				from               *v1.PodDefaultNetwork
				body               string
				wantErr            string
				wantDefaultNetwork *v1.PodDefaultNetwork
				wantHostNetwork    bool
			}{
				{
					name: "old client sets hostNetwork on an unset template",
					body: hostNetworkTrue, wantHostNetwork: true,
				},
				{
					name: "old client sets hostNetwork on a Pod template",
					from: new(v1.PodDefaultNetworkPod), body: hostNetworkTrue,
					wantDefaultNetwork: new(v1.PodDefaultNetworkHost), wantHostNetwork: true,
				},
				{
					name: "old client sets hostNetwork on a None template",
					from: new(v1.PodDefaultNetworkNone), body: hostNetworkTrue,
					wantErr: `must not be "None" when hostNetwork is true`,
				},
				{
					name: "new client sets Pod on a Host template",
					from: new(v1.PodDefaultNetworkHost), body: defaultNetPod,
					wantDefaultNetwork: new(v1.PodDefaultNetworkPod),
				},
				{
					name: "new client sets None on a Host template",
					from: new(v1.PodDefaultNetworkHost), body: defaultNetNone,
					wantDefaultNetwork: new(v1.PodDefaultNetworkNone),
				},
				{
					name: "old client clears the field on a None template",
					from: new(v1.PodDefaultNetworkNone), body: defaultNetUnset,
					wantDefaultNetwork: new(v1.PodDefaultNetworkNone),
				},
				{
					name: "old client clears the field on a Host template",
					from: new(v1.PodDefaultNetworkHost), body: defaultNetUnset,
					wantDefaultNetwork: new(v1.PodDefaultNetworkHost), wantHostNetwork: true,
				},
				{
					// Known limitation: defaulting mirrors the stored "Host"
					// back into hostNetwork before the strategy can see the
					// change, so the patch is a no-op. A full update works.
					name: "old client clears hostNetwork on a Host template is a no-op",
					from: new(v1.PodDefaultNetworkHost), body: hostNetworkFalse,
					wantDefaultNetwork: new(v1.PodDefaultNetworkHost), wantHostNetwork: true,
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					name := fmt.Sprintf("%s-update-%d", prefix, i)
					if _, err := w.create(name, tc.from); err != nil {
						t.Fatalf("create: %v", err)
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
					expectTemplate(t, spec, tc.wantDefaultNetwork, tc.wantHostNetwork)
				})
			}
		})
	}

	// Server-side apply: a manager that applies defaultNetwork "Host" and
	// later "Pod" without ever writing hostNetwork gets what it asked for.
	t.Run("Deployment/server-side apply Host then Pod", func(t *testing.T) {
		const manager = "test-manager"
		apply := func(dn v1.PodDefaultNetwork) *v1.PodSpec {
			t.Helper()
			ac := appsv1ac.Deployment("ssa", ns.Name).WithSpec(appsv1ac.DeploymentSpec().
				WithSelector(metav1ac.LabelSelector().WithMatchLabels(map[string]string{"app": "ssa"})).
				WithTemplate(corev1ac.PodTemplateSpec().
					WithLabels(map[string]string{"app": "ssa"}).
					WithSpec(corev1ac.PodSpec().
						WithDefaultNetwork(dn).
						WithContainers(corev1ac.Container().WithName("c").WithImage("registry.k8s.io/pause:3.10")))))
			obj, err := client.AppsV1().Deployments(ns.Name).Apply(ctx, ac, metav1.ApplyOptions{FieldManager: manager, Force: true})
			if err != nil {
				t.Fatalf("apply %q: %v", dn, err)
			}
			return &obj.Spec.Template.Spec
		}
		expectTemplate(t, apply(v1.PodDefaultNetworkHost), new(v1.PodDefaultNetworkHost), true)
		expectTemplate(t, apply(v1.PodDefaultNetworkPod), new(v1.PodDefaultNetworkPod), false)
		expectTemplate(t, apply(v1.PodDefaultNetworkNone), new(v1.PodDefaultNetworkNone), false)
	})
}

// TestPodDefaultNetworkGateDisabled covers the field being rejected on new
// objects while the gate is disabled, and kept and still validated on objects
// that already carry it.
func TestPodDefaultNetworkGateDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-default-network-gate", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)
	ctx := context.Background()

	// Created while the gate is enabled.
	isolated, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "isolated", Namespace: ns.Name},
		Spec:       v1.PodSpec{DefaultNetwork: new(v1.PodDefaultNetworkNone), Containers: []v1.Container{isolatedContainer()}},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}
	deployment := templateWorkloads(ctx, client, ns.Name)[0]
	if _, err := deployment.create("isolated", new(v1.PodDefaultNetworkNone)); err != nil {
		t.Fatalf("failed to create deployment: %v", err)
	}

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NetworkIsolatedPods, false)

	// This is also what happens to pods that a controller creates from a
	// template stored with "None": they are rejected instead of being
	// attached to the pod network.
	for _, dn := range []v1.PodDefaultNetwork{v1.PodDefaultNetworkPod, v1.PodDefaultNetworkHost, v1.PodDefaultNetworkNone} {
		t.Run(fmt.Sprintf("%s is rejected on create", dn), func(t *testing.T) {
			_, err := client.CoreV1().Pods(ns.Name).Create(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "new-" + strings.ToLower(string(dn)), Namespace: ns.Name},
				Spec:       v1.PodSpec{DefaultNetwork: &dn, Containers: []v1.Container{isolatedContainer()}},
			}, metav1.CreateOptions{})
			if err == nil || !strings.Contains(err.Error(), gateDisabledErr) {
				t.Fatalf("expected error containing %q, got: %v", gateDisabledErr, err)
			}
			_, err = deployment.create("new-"+strings.ToLower(string(dn)), &dn)
			if err == nil || !strings.Contains(err.Error(), gateDisabledErr) {
				t.Fatalf("expected deployment error containing %q, got: %v", gateDisabledErr, err)
			}
		})
	}

	t.Run("hostNetwork works without the field", func(t *testing.T) {
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
		// So does a full update from a client that does not know the field.
		updated.Spec.DefaultNetwork = nil
		updated.Labels["updated"] = "twice"
		updated, err = client.CoreV1().Pods(ns.Name).Update(ctx, updated, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("failed to update pod without the field: %v", err)
		}
		if ptr.Deref(updated.Spec.DefaultNetwork, "") != v1.PodDefaultNetworkNone {
			t.Errorf("spec.defaultNetwork = %v after update without the field, want None", updated.Spec.DefaultNetwork)
		}
		// The status invariant does not depend on the gate.
		updated.Status.PodIP = "10.0.0.1"
		updated.Status.PodIPs = []v1.PodIP{{IP: "10.0.0.1"}}
		if _, err := client.CoreV1().Pods(ns.Name).UpdateStatus(ctx, updated, metav1.UpdateOptions{}); err == nil {
			t.Errorf("expected status update with a pod IP to be rejected")
		}
	})

	t.Run("stored template keeps the field", func(t *testing.T) {
		spec, err := deployment.patch("isolated", types.StrategicMergePatchType, `{"metadata":{"labels":{"updated":"true"}}}`)
		if err != nil {
			t.Fatalf("patch: %v", err)
		}
		expectTemplate(t, spec, new(v1.PodDefaultNetworkNone), false)
		spec, err = deployment.patch("isolated", types.StrategicMergePatchType, deployment.templatePatch(`{"defaultNetwork":null}`))
		if err != nil {
			t.Fatalf("patch clearing the field: %v", err)
		}
		expectTemplate(t, spec, new(v1.PodDefaultNetworkNone), false)
	})
}
