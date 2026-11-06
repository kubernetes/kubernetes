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
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/dynamiccontainers"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
	psaapi "k8s.io/pod-security-admission/api"
)

func enableDynamicContainers(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.NodeDeclaredFeatures: true,
		features.DynamicContainers:    true,
	})
}

func dynamicContainersTestPod(name, nodeName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Containers: []v1.Container{{
				Name:  "main",
				Image: "fakeimage",
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m")},
				},
			}},
		},
	}
}

// createRunningPod creates the pod and marks it Running and Initialized, which dynamic container
// changes require.
func createRunningPod(ctx context.Context, t *testing.T, client clientset.Interface, ns string, pod *v1.Pod) *v1.Pod {
	t.Helper()
	pod, err := client.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create pod: %v", err)
	}
	pod.Status.Phase = v1.PodRunning
	pod.Status.Conditions = []v1.PodCondition{{Type: v1.PodInitialized, Status: v1.ConditionTrue}}
	pod, err = client.CoreV1().Pods(ns).UpdateStatus(ctx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("Failed to update pod status: %v", err)
	}
	return pod
}

func withAddedContainer(pod *v1.Pod, c v1.Container) *v1.Pod {
	pod = pod.DeepCopy()
	if c.Image == "" {
		c.Image = "fakeimage"
	}
	pod.Spec.Containers = append(pod.Spec.Containers, c)
	return pod
}

func createDynamicContainersNode(ctx context.Context, t *testing.T, client clientset.Interface, name string, declaredFeatures []string) {
	t.Helper()
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{v1.LabelOSStable: "linux"},
		},
	}
	node, err := client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}
	node.Status = v1.NodeStatus{
		DeclaredFeatures: declaredFeatures,
		Allocatable: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("4"),
			v1.ResourceMemory: resource.MustParse("8Gi"),
		},
		Capacity: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("4"),
			v1.ResourceMemory: resource.MustParse("8Gi"),
		},
	}
	if _, err := client.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Failed to update node status: %v", err)
	}
	t.Cleanup(func() {
		if err := client.CoreV1().Nodes().Delete(context.Background(), name, metav1.DeleteOptions{}); err != nil {
			t.Errorf("Failed to delete node %s: %v", name, err)
		}
	})
}

// TestDynamicContainersDefaultRoles verifies that the default edit and admin roles don't grant
// updates through pods/dynamic.
func TestDynamicContainersDefaultRoles(t *testing.T) {
	enableDynamicContainers(t)
	ctx := t.Context()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		append(framework.DefaultTestServerFlags(), "--authorization-mode=RBAC"), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(adminClient, "dynamic-containers-rbac", t)
	defer framework.DeleteNamespaceOrDie(adminClient, ns, t)

	const nodeName = "dynamic-containers-node"
	createDynamicContainersNode(ctx, t, adminClient, nodeName, []string{dynamiccontainers.DynamicContainersFeatureGate})

	// The edit and admin roles are filled in by the clusterrole aggregation controller, which
	// doesn't run here, so bind the roles that they aggregate instead.
	aggregatedRoles := map[string][]string{
		"edit":  {"system:aggregate-to-edit", "system:aggregate-to-view"},
		"admin": {"system:aggregate-to-admin", "system:aggregate-to-edit", "system:aggregate-to-view"},
	}
	for role, sources := range aggregatedRoles {
		t.Run(role, func(t *testing.T) {
			user := role + "-user"
			for _, source := range sources {
				if _, err := adminClient.RbacV1().RoleBindings(ns.Name).Create(ctx, &rbacv1.RoleBinding{
					ObjectMeta: metav1.ObjectMeta{Name: user + "-" + strings.TrimPrefix(source, "system:")},
					Subjects:   []rbacv1.Subject{{Kind: rbacv1.UserKind, APIGroup: rbacv1.GroupName, Name: user}},
					RoleRef:    rbacv1.RoleRef{Kind: "ClusterRole", APIGroup: rbacv1.GroupName, Name: source},
				}, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create role binding: %v", err)
				}
			}
			// Wait for the binding to take effect before checking what it doesn't grant.
			authutil.WaitForNamedAuthorizationUpdate(t, ctx, adminClient.AuthorizationV1(), user, ns.Name, "update", "", v1.Resource("pods"), true)

			userConfig := rest.CopyConfig(server.ClientConfig)
			userConfig.Impersonate = rest.ImpersonationConfig{UserName: user}
			userClient := clientset.NewForConfigOrDie(userConfig)

			pod := createRunningPod(ctx, t, adminClient, ns.Name, dynamicContainersTestPod(role, nodeName))
			added := withAddedContainer(pod, v1.Container{Name: "added"})
			_, err := userClient.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
			if !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), `cannot update resource "pods/dynamic"`) {
				t.Fatalf("Expected the %s role to be forbidden from updating pods/dynamic, got %v", role, err)
			}

			// The same update succeeds for a cluster admin.
			if _, err := adminClient.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("Unexpected error adding a container as cluster admin: %v", err)
			}
		})
	}
}

// TestDynamicContainersDisabled verifies that pods/dynamic isn't served with the DynamicContainers
// gate off, and that the main resource still rejects adding containers.
func TestDynamicContainersDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicContainers, false)
	ctx := t.Context()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "dynamic-containers-disabled", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	pod := createRunningPod(ctx, t, client, ns.Name, dynamicContainersTestPod("disabled", ""))
	added := withAddedContainer(pod, v1.Container{Name: "added"})
	if _, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{}); !apierrors.IsNotFound(err) {
		t.Errorf("Expected pods/dynamic to be not found, got %v", err)
	}
	if _, err := client.CoreV1().Pods(ns.Name).Update(ctx, added, metav1.UpdateOptions{}); !apierrors.IsInvalid(err) {
		t.Errorf("Expected adding a container through the main resource to be invalid, got %v", err)
	}
}

// TestDynamicContainersAdmission verifies that the first-party admission plugins are invoked on
// updates through pods/dynamic.
func TestDynamicContainersAdmission(t *testing.T) {
	enableDynamicContainers(t)
	ctx := t.Context()
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)
	client := clientset.NewForConfigOrDie(server.ClientConfig)

	const (
		nodeName         = "dynamic-containers-node"
		nodeWithoutFeats = "dynamic-containers-unsupported-node"
	)
	createDynamicContainersNode(ctx, t, client, nodeName, []string{dynamiccontainers.DynamicContainersFeatureGate})
	createDynamicContainersNode(ctx, t, client, nodeWithoutFeats, nil)

	newNamespace := func(t *testing.T, name string) *v1.Namespace {
		ns := framework.CreateNamespaceOrDie(client, name, t)
		t.Cleanup(func() { framework.DeleteNamespaceOrDie(client, ns, t) })
		return ns
	}
	cpu := func(request, limit string) v1.ResourceRequirements {
		r := v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(request)}}
		if limit != "" {
			r.Limits = v1.ResourceList{v1.ResourceCPU: resource.MustParse(limit)}
		}
		return r
	}
	expectError := func(t *testing.T, err error, contains string) {
		t.Helper()
		if err == nil {
			t.Fatalf("Expected an error containing %q, got none", contains)
		}
		if !strings.Contains(err.Error(), contains) {
			t.Fatalf("Expected an error containing %q, got %v", contains, err)
		}
	}

	t.Run("PodSecurity", func(t *testing.T) {
		ns := newNamespace(t, "dynamic-containers-psa")
		if ns.Labels == nil {
			ns.Labels = map[string]string{}
		}
		ns.Labels[psaapi.EnforceLevelLabel] = string(psaapi.LevelBaseline)
		if _, err := client.CoreV1().Namespaces().Update(ctx, ns, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to label namespace: %v", err)
		}
		pod := createRunningPod(ctx, t, client, ns.Name, dynamicContainersTestPod("psa", nodeName))

		added := withAddedContainer(pod, v1.Container{
			Name:            "added",
			Resources:       cpu("100m", ""),
			SecurityContext: &v1.SecurityContext{Capabilities: &v1.Capabilities{Add: []v1.Capability{"SYS_ADMIN"}}},
		})
		_, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		expectError(t, err, `violates PodSecurity "baseline:latest"`)

		added = withAddedContainer(pod, v1.Container{Name: "added", Resources: cpu("100m", "")})
		if _, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Unexpected error adding a baseline container: %v", err)
		}
	})

	t.Run("LimitRanger", func(t *testing.T) {
		ns := newNamespace(t, "dynamic-containers-limitranger")
		if _, err := client.CoreV1().LimitRanges(ns.Name).Create(ctx, &v1.LimitRange{
			ObjectMeta: metav1.ObjectMeta{Name: "limits"},
			Spec: v1.LimitRangeSpec{Limits: []v1.LimitRangeItem{{
				Type:           v1.LimitTypeContainer,
				Max:            v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
				Default:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("800m")},
				DefaultRequest: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
			}}},
		}, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create LimitRange: %v", err)
		}
		// LimitRanger reads LimitRanges from an informer.
		var pod *v1.Pod
		if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			p, err := client.CoreV1().Pods(ns.Name).Create(ctx, dynamicContainersTestPod("probe", nodeName), metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				return false, err
			}
			return p.Spec.Containers[0].Resources.Limits != nil, nil
		}); err != nil {
			t.Fatalf("LimitRange never took effect: %v", err)
		}
		pod = createRunningPod(ctx, t, client, ns.Name, dynamicContainersTestPod("limitranger", nodeName))

		added := withAddedContainer(pod, v1.Container{Name: "too-big", Resources: cpu("2", "2")})
		_, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		expectError(t, err, "maximum cpu usage per Container is 1")

		added = withAddedContainer(pod, v1.Container{Name: "defaulted"})
		updated, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error adding a container: %v", err)
		}
		if got := updated.Spec.Containers[1].Resources; !got.Requests.Cpu().Equal(resource.MustParse("100m")) || !got.Limits.Cpu().Equal(resource.MustParse("800m")) {
			t.Errorf("Added container wasn't defaulted by LimitRanger: %v", got)
		}
	})

	t.Run("ResourceQuota", func(t *testing.T) {
		ns := newNamespace(t, "dynamic-containers-quota")
		quota, err := client.CoreV1().ResourceQuotas(ns.Name).Create(ctx, &v1.ResourceQuota{
			ObjectMeta: metav1.ObjectMeta{Name: "quota"},
			Spec:       v1.ResourceQuotaSpec{Hard: v1.ResourceList{v1.ResourceRequestsCPU: resource.MustParse("1")}},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create ResourceQuota: %v", err)
		}
		// There is no quota controller, so seed the status it would have computed.
		quota.Status = v1.ResourceQuotaStatus{
			Hard: quota.Spec.Hard,
			Used: v1.ResourceList{v1.ResourceRequestsCPU: resource.MustParse("0")},
		}
		if _, err := client.CoreV1().ResourceQuotas(ns.Name).UpdateStatus(ctx, quota, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Failed to update ResourceQuota status: %v", err)
		}
		// The quota admission plugin reads quotas from an informer.
		if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			_, err := client.CoreV1().Pods(ns.Name).Create(ctx, withAddedContainer(dynamicContainersTestPod("probe", nodeName), v1.Container{Name: "no-requests"}), metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			return err != nil && strings.Contains(err.Error(), "must specify"), nil
		}); err != nil {
			t.Fatalf("ResourceQuota never took effect: %v", err)
		}
		pod := createRunningPod(ctx, t, client, ns.Name, dynamicContainersTestPod("quota", nodeName))

		added := withAddedContainer(pod, v1.Container{Name: "too-big", Resources: cpu("600m", "")})
		_, err = client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		expectError(t, err, "exceeded quota")

		added = withAddedContainer(pod, v1.Container{Name: "no-requests"})
		_, err = client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		expectError(t, err, "must specify requests.cpu")

		added = withAddedContainer(pod, v1.Container{Name: "fits", Resources: cpu("400m", "")})
		if _, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Unexpected error adding a container within quota: %v", err)
		}
		quota, err = client.CoreV1().ResourceQuotas(ns.Name).Get(ctx, quota.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Failed to get ResourceQuota: %v", err)
		}
		if used := quota.Status.Used[v1.ResourceRequestsCPU]; !used.Equal(resource.MustParse("900m")) {
			t.Errorf("Expected 900m requests.cpu used, got %s", used.String())
		}
	})

	t.Run("PodResizeValidator", func(t *testing.T) {
		ns := newNamespace(t, "dynamic-containers-resize")
		pod := createRunningPod(ctx, t, client, ns.Name, dynamicContainersTestPod("resize", nodeName))

		added := withAddedContainer(pod, v1.Container{Name: "too-big", Resources: cpu("4", "")})
		_, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		expectError(t, err, "node didn't have enough allocatable resources: cpu")
	})

	t.Run("NodeDeclaredFeatures", func(t *testing.T) {
		ns := newNamespace(t, "dynamic-containers-ndf")
		pod := createRunningPod(ctx, t, client, ns.Name, dynamicContainersTestPod("ndf", nodeWithoutFeats))

		added := withAddedContainer(pod, v1.Container{Name: "added", Resources: cpu("100m", "")})
		_, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, added, metav1.UpdateOptions{})
		expectError(t, err, "pod update requires features DynamicContainers which are not available on node")

		// Updates that don't add or remove containers don't need the feature.
		updated := pod.DeepCopy()
		updated.Spec.Containers[0].Image = "otherimage"
		if _, err := client.CoreV1().Pods(ns.Name).UpdateDynamic(ctx, pod.Name, updated, metav1.UpdateOptions{}); err != nil {
			t.Fatalf("Unexpected error updating an image: %v", err)
		}
	})
}
