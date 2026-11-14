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

package podcheckpoint

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admission/v1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	nodeapi "k8s.io/kubernetes/pkg/apis/node"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func preparedRestorePod(checkpoint *nodev1alpha1.PodCheckpoint, name string) *v1.Pod {
	spec := checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name, UID: checkpoint.UID}
	checkpointutil.AddRestoreNodeAffinity(spec, *checkpoint.Status.NodeName)
	return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: checkpoint.Namespace, Finalizers: []string{nodeapi.PodRestoreProtectionFinalizer}}, Spec: *spec}
}

func TestRestoreStorageCreatePaths(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(ctx, t, true, func(opts *options.ServerRunOptions) {
		opts.Authorization.Modes = []string{"RBAC"}
		opts.Admission.GenericAdmission.DisablePlugins = append(opts.Admission.GenericAdmission.DisablePlugins, "PodRestoreAuthorization")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-storage-paths", t)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	for _, allowed := range []bool{false, true} {
		username := fmt.Sprintf("restore-%t", allowed)
		rules := []rbacv1.PolicyRule{{APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"create", "patch"}}}
		if allowed {
			rules = append(rules, rbacv1.PolicyRule{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, ResourceNames: []string{checkpoint.Name}, Verbs: []string{"restore"}})
		}
		_, err := client.RbacV1().Roles(ns.Name).Create(ctx, &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: username}, Rules: rules}, metav1.CreateOptions{})
		require.NoError(t, err)
		_, err = client.RbacV1().RoleBindings(ns.Name).Create(ctx, &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: username}, Subjects: []rbacv1.Subject{{Kind: "User", APIGroup: rbacv1.GroupName, Name: username}}, RoleRef: rbacv1.RoleRef{Kind: "Role", APIGroup: rbacv1.GroupName, Name: username}}, metav1.CreateOptions{})
		require.NoError(t, err)
		for _, verb := range []string{"create", "patch"} {
			authutil.WaitForNamedAuthorizationUpdate(t, ctx, client.AuthorizationV1(), username, ns.Name, verb, "", schema.GroupResource{Resource: "pods"}, true)
		}
		authutil.WaitForNamedAuthorizationUpdate(t, ctx, client.AuthorizationV1(), username, ns.Name, "restore", checkpoint.Name, schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, allowed)
		userConfig := rest.CopyConfig(config)
		userConfig.Impersonate = rest.ImpersonationConfig{UserName: username}
		userClient := clientset.NewForConfigOrDie(userConfig)
		_, err = userClient.NodeV1alpha1().PodCheckpoints(ns.Name).Get(ctx, checkpoint.Name, metav1.GetOptions{})
		require.True(t, apierrors.IsForbidden(err), "restore must not require ordinary checkpoint read permission: %v", err)
		for _, method := range []string{"post", "apply"} {
			for _, dryRun := range []bool{false, true} {
				t.Run(fmt.Sprintf("allowed=%t/%s/dryrun=%t", allowed, method, dryRun), func(t *testing.T) {
					pod := preparedRestorePod(checkpoint, fmt.Sprintf("%s-%s-%t", username, method, dryRun))
					var dry []string
					if dryRun {
						dry = []string{metav1.DryRunAll}
					}
					if method == "post" {
						_, err = userClient.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{DryRun: dry})
					} else {
						pod.APIVersion, pod.Kind = "v1", "Pod"
						data, marshalErr := json.Marshal(pod)
						require.NoError(t, marshalErr)
						_, err = userClient.CoreV1().Pods(ns.Name).Patch(ctx, pod.Name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "restore-storage", DryRun: dry})
					}
					if allowed {
						require.NoError(t, err, "an authorized, prepared restore must work without the admission plugin")
					} else {
						require.True(t, apierrors.IsForbidden(err), "caller-supplied protection must not authorize restore: %v", err)
						require.ErrorContains(t, err, "not authorized")
					}
					_, getErr := client.CoreV1().Pods(ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
					if !allowed || dryRun {
						require.True(t, apierrors.IsNotFound(getErr), "rejected and dry-run restores must not be stored: %v", getErr)
					} else {
						require.NoError(t, getErr)
					}
				})
			}
		}
	}
}

func TestRestoreStorageProtocolDefaults(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.H2CContainerProbe, false)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true, func(opts *options.ServerRunOptions) {
		opts.Admission.GenericAdmission.DisablePlugins = append(opts.Admission.GenericAdmission.DisablePlugins, "PodRestoreAuthorization")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-protocol-defaults", t)
	pods := client.CoreV1().Pods(ns.Name)
	checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
	for _, phase := range []struct {
		name             string
		capture, restore bool
	}{
		{name: "off-off"},
		{name: "off-on", restore: true},
		{name: "on-on", capture: true, restore: true},
		{name: "on-off", capture: true},
	} {
		t.Run(phase.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.H2CContainerProbe, phase.capture)
			source, err := pods.Create(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "source-" + phase.name},
				Spec: v1.PodSpec{NodeName: "node-1", Containers: []v1.Container{{
					Name: "app", Image: "pause",
					ReadinessProbe: &v1.Probe{ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{Port: intstr.FromInt32(80)}}},
				}}},
			}, metav1.CreateOptions{})
			require.NoError(t, err)
			source = markCheckpointSourceRunning(ctx, t, client, source)
			checkpoint, err := checkpoints.Create(ctx, newPodCheckpoint(ns.Name, "checkpoint-"+phase.name, source.Name), metav1.CreateOptions{})
			require.NoError(t, err)
			checkpoint.Status.NodeName = new("node-1")
			checkpoint.Status.SourcePodUID = &source.UID
			checkpoint.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(source)
			checkpoint.Status.Conditions = []metav1.Condition{{Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue, Reason: "Completed", LastTransitionTime: metav1.Now()}}
			checkpoint, err = checkpoints.UpdateStatus(ctx, checkpoint, metav1.UpdateOptions{})
			require.NoError(t, err)
			captured := checkpoint.Status.CheckpointedPodTemplate.DeepCopy()
			if phase.capture {
				require.Equal(t, ptr.To(v1.HTTPProtocolHTTP1), captured.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol)
			} else {
				require.Nil(t, captured.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol)
			}

			// Only Pod defaulting materializes HTTP1 after the gate changes; decoding
			// the captured template must not determine whether an equivalent restore works.
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.H2CContainerProbe, phase.restore)
			for _, protocol := range []struct {
				name  string
				value *v1.HTTPProtocol
			}{
				{name: "omitted"},
				{name: "http1", value: ptr.To(v1.HTTPProtocolHTTP1)},
				{name: "http2", value: ptr.To(v1.HTTPProtocolHTTP2)},
			} {
				if protocol.name == "http2" && !phase.restore {
					continue
				}
				for _, method := range []string{"post", "apply"} {
					for _, dryRun := range []bool{false, true} {
						t.Run(fmt.Sprintf("%s/%s/dryrun=%t", protocol.name, method, dryRun), func(t *testing.T) {
							pod := preparedRestorePod(checkpoint, fmt.Sprintf("%s-%s-%s-%t", phase.name, protocol.name, method, dryRun))
							pod.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol = protocol.value
							var dry []string
							if dryRun {
								dry = []string{metav1.DryRunAll}
							}
							var restored *v1.Pod
							if method == "post" {
								restored, err = pods.Create(ctx, pod, metav1.CreateOptions{DryRun: dry})
							} else {
								pod.APIVersion, pod.Kind = "v1", "Pod"
								data, marshalErr := json.Marshal(pod)
								require.NoError(t, marshalErr)
								restored, err = pods.Patch(ctx, pod.Name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "restore-protocol", DryRun: dry})
							}
							if protocol.name == "http2" {
								require.True(t, apierrors.IsInvalid(err), "a different protocol must remain invalid: %v", err)
								require.ErrorContains(t, err, "spec.containers[0].readinessProbe.httpGet.protocol")
							} else {
								require.NoError(t, err, "equivalent HTTP/1.1 probes must restore across gate changes")
								if phase.restore {
									require.Equal(t, ptr.To(v1.HTTPProtocolHTTP1), restored.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol)
								} else {
									require.Nil(t, restored.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol)
								}
							}
							_, getErr := pods.Get(ctx, pod.Name, metav1.GetOptions{})
							if protocol.name == "http2" || dryRun {
								require.True(t, apierrors.IsNotFound(getErr), "rejected and dry-run restores must not persist: %v", getErr)
							} else {
								require.NoError(t, getErr)
							}
						})
					}
				}
			}
			stored, err := checkpoints.Get(ctx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, captured, stored.Status.CheckpointedPodTemplate, "restore requests must not rewrite the captured template")
		})
	}
}

func TestRestoreAfterSourceChangesBeforeCheckpointPublication(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	node, err := client.CoreV1().Nodes().Create(ctx, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}, metav1.CreateOptions{})
	require.NoError(t, err)
	node.Status.Capacity = v1.ResourceList{v1.ResourceCPU: resource.MustParse("4"), v1.ResourceMemory: resource.MustParse("8Gi")}
	node.Status.Allocatable = node.Status.Capacity.DeepCopy()
	_, err = client.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
	require.NoError(t, err)

	for _, change := range []string{"resize", "ephemeral-container"} {
		t.Run(change, func(t *testing.T) {
			ns := framework.CreateNamespaceOrDie(client, "restore-source-"+change, t)
			pods := client.CoreV1().Pods(ns.Name)
			checkpoints := client.NodeV1alpha1().PodCheckpoints(ns.Name)
			source, err := pods.Create(ctx, &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "source"},
				Spec: v1.PodSpec{NodeName: "node-1", Containers: []v1.Container{{
					Name: "app", Image: "pause",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m"), v1.ResourceMemory: resource.MustParse("64Mi")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m"), v1.ResourceMemory: resource.MustParse("64Mi")},
					},
				}}},
			}, metav1.CreateOptions{})
			require.NoError(t, err)
			source = markCheckpointSourceRunning(ctx, t, client, source)
			checkpoint, err := checkpoints.Create(ctx, newPodCheckpoint(ns.Name, "checkpoint", source.Name), metav1.CreateOptions{})
			require.NoError(t, err)
			checkpoint.Status.NodeName = new("node-1")
			checkpoint.Status.SourcePodUID = &source.UID
			checkpoint.Status.CheckpointedPodTemplate = checkpointutil.SanitizePodTemplate(source)
			checkpoint.Status.Conditions = []metav1.Condition{{
				Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionFalse,
				Reason: nodev1alpha1.PodCheckpointReasonInProgress, LastTransitionTime: metav1.Now(),
			}}
			checkpoint, err = checkpoints.UpdateStatus(ctx, checkpoint, metav1.UpdateOptions{})
			require.NoError(t, err)
			captured := checkpoint.Status.CheckpointedPodTemplate.DeepCopy()
			source.Status.Conditions = append(source.Status.Conditions, v1.PodCondition{
				Type: v1.PodCheckpointing, Status: v1.ConditionTrue, Reason: "CheckpointingPod", LastTransitionTime: metav1.Now(),
			})
			source, err = pods.UpdateStatus(ctx, source, metav1.UpdateOptions{})
			require.NoError(t, err)
			changed := source.DeepCopy()
			var update func(*v1.Pod) (*v1.Pod, error)
			if change == "resize" {
				changed.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("200m")
				changed.Spec.Containers[0].Resources.Limits[v1.ResourceCPU] = resource.MustParse("200m")
				update = func(pod *v1.Pod) (*v1.Pod, error) {
					return pods.UpdateResize(ctx, pod.Name, pod, metav1.UpdateOptions{})
				}
			} else {
				changed.Spec.EphemeralContainers = []v1.EphemeralContainer{{EphemeralContainerCommon: v1.EphemeralContainerCommon{Name: "debug", Image: "pause"}}}
				update = func(pod *v1.Pod) (*v1.Pod, error) {
					return pods.UpdateEphemeralContainers(ctx, pod.Name, pod, metav1.UpdateOptions{})
				}
			}
			_, err = update(changed)
			require.True(t, apierrors.IsInvalid(err), "source changes must be blocked during capture: %v", err)
			require.ErrorContains(t, err, "Checkpointing=True")

			// Runtime completion releases the source before the checkpoint outcome is published.
			source.Status.Conditions = slices.DeleteFunc(source.Status.Conditions, func(condition v1.PodCondition) bool {
				return condition.Type == v1.PodCheckpointing
			})
			source, err = pods.UpdateStatus(ctx, source, metav1.UpdateOptions{})
			require.NoError(t, err)
			changed.ResourceVersion = source.ResourceVersion
			source, err = update(changed)
			require.NoError(t, err, "source changes must be allowed after capture")
			pending, err := checkpoints.Get(ctx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.False(t, apimeta.IsStatusConditionTrue(pending.Status.Conditions, nodev1alpha1.PodCheckpointConditionReady))
			require.Equal(t, captured, pending.Status.CheckpointedPodTemplate)
			_, err = pods.Create(ctx, preparedRestorePod(pending, "too-early"), metav1.CreateOptions{})
			require.Error(t, err, "a captured template alone must not permit restore")
			require.ErrorContains(t, err, "not ready")

			apimeta.SetStatusCondition(&pending.Status.Conditions, metav1.Condition{
				Type: nodev1alpha1.PodCheckpointConditionReady, Status: metav1.ConditionTrue,
				Reason: nodev1alpha1.PodCheckpointReasonCompleted, LastTransitionTime: metav1.Now(),
			})
			checkpoint, err = checkpoints.UpdateStatus(ctx, pending, metav1.UpdateOptions{})
			require.NoError(t, err)
			restored, err := pods.Create(ctx, preparedRestorePod(checkpoint, "restore"), metav1.CreateOptions{})
			require.NoError(t, err, "restore must use the captured template after source changes")
			require.Equal(t, captured.Spec.Containers[0].Resources, restored.Spec.Containers[0].Resources)
			require.Empty(t, restored.Spec.EphemeralContainers)
			if change == "resize" {
				fromLiveSource := preparedRestorePod(checkpoint, "restore-live-source")
				fromLiveSource.Spec.Containers[0].Resources = source.Spec.Containers[0].Resources
				_, err = pods.Create(ctx, fromLiveSource, metav1.CreateOptions{})
				require.True(t, apierrors.IsInvalid(err), "restore must reject the resized source spec: %v", err)
				require.ErrorContains(t, err, "resources")
			}
			stored, err := checkpoints.Get(ctx, checkpoint.Name, metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, captured, stored.Status.CheckpointedPodTemplate)
		})
	}
}

func TestRestoreStorageRejectsUnpreparedPods(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true, func(opts *options.ServerRunOptions) {
		opts.Admission.GenericAdmission.DisablePlugins = append(opts.Admission.GenericAdmission.DisablePlugins, "PodRestoreAuthorization")
	})
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-storage-fields", t)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	for _, tc := range []struct {
		name    string
		change  func(*v1.Pod)
		invalid bool
	}{
		{name: "missing-protection", change: func(p *v1.Pod) { p.Finalizers = nil }},
		{name: "missing-affinity", change: func(p *v1.Pod) { p.Spec.Affinity = nil }, invalid: true},
		{name: "wrong-affinity", change: func(p *v1.Pod) {
			p.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[0].MatchFields[0].Values = []string{"node-2"}
		}, invalid: true},
		{name: "bound", change: func(p *v1.Pod) { p.Spec.NodeName = "node-1" }},
		{name: "workload", change: func(p *v1.Pod) { p.Spec.Containers[0].Image = "registry.example/private-changed-image:latest" }, invalid: true},
		{name: "missing-checkpoint", change: func(p *v1.Pod) { p.Spec.RestoreFrom.Name = "missing" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod := preparedRestorePod(checkpoint, tc.name)
			tc.change(pod)
			_, err := client.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{})
			if tc.invalid {
				require.True(t, apierrors.IsInvalid(err), "expected invalid restore: %v", err)
			} else {
				require.True(t, apierrors.IsForbidden(err), "expected forbidden restore: %v", err)
			}
			require.NotContains(t, err.Error(), "private-changed-image")
			_, getErr := client.CoreV1().Pods(ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			require.True(t, apierrors.IsNotFound(getErr), "rejected restore must not be stored: %v", getErr)
		})
	}
	checkpoint.Finalizers = []string{"example.com/retain"}
	checkpoint, err := client.NodeV1alpha1().PodCheckpoints(ns.Name).Update(ctx, checkpoint, metav1.UpdateOptions{})
	require.NoError(t, err)
	require.NoError(t, client.NodeV1alpha1().PodCheckpoints(ns.Name).Delete(ctx, checkpoint.Name, metav1.DeleteOptions{}))
	_, err = client.CoreV1().Pods(ns.Name).Create(ctx, preparedRestorePod(checkpoint, "deleting-checkpoint"), metav1.CreateOptions{})
	require.True(t, apierrors.IsForbidden(err), "a deleting checkpoint must not be restored: %v", err)
	_, err = client.CoreV1().Pods(ns.Name).Get(ctx, "deleting-checkpoint", metav1.GetOptions{})
	require.True(t, apierrors.IsNotFound(err))
}

func startRestoreMutationWebhook(t *testing.T, ctx ktesting.TContext, client clientset.Interface, namespace string, patches map[string][]byte) func() {
	t.Helper()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var review admissionv1.AdmissionReview
		if err := json.NewDecoder(r.Body).Decode(&review); err != nil || review.Request == nil {
			http.Error(w, "invalid AdmissionReview", http.StatusBadRequest)
			return
		}
		var pod v1.Pod
		if err := json.Unmarshal(review.Request.Object.Raw, &pod); err != nil {
			http.Error(w, "invalid Pod", http.StatusBadRequest)
			return
		}
		var patch []byte
		if r.URL.Path == "/barrier" {
			patch = []byte(`[{"op":"add","path":"/metadata/annotations","value":{"restore-mutation-ready":"true"}}]`)
		} else if pod.Annotations["restore-mutation-ready"] == "true" {
			patch = patches[pod.Labels["mutation"]]
			if pod.Labels["mutation"] == "probe" {
				patch = []byte(`[{"op":"add","path":"/metadata/annotations","value":{"webhook":"observed"}}]`)
			}
		}
		review.Response = &admissionv1.AdmissionResponse{UID: review.Request.UID, Allowed: true, Patch: patch}
		if len(patch) > 0 {
			review.Response.PatchType = ptr.To(admissionv1.PatchTypeJSONPatch)
		}
		review.Request = nil
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(review); err != nil {
			t.Errorf("encode admission response: %v", err)
		}
	}))
	ca := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw})
	mutationURL := server.URL + "/mutate"
	mutationWebhook := admissionregistrationv1.MutatingWebhook{
		Name:               "restore-mutation.integration.test",
		ClientConfig:       admissionregistrationv1.WebhookClientConfig{URL: &mutationURL, CABundle: ca},
		Rules:              []admissionregistrationv1.RuleWithOperations{{Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create}, Rule: admissionregistrationv1.Rule{APIGroups: []string{""}, APIVersions: []string{"v1"}, Resources: []string{"pods"}}}},
		ObjectSelector:     &metav1.LabelSelector{MatchLabels: map[string]string{"restore-mutation": "true"}},
		ReinvocationPolicy: ptr.To(admissionregistrationv1.IfNeededReinvocationPolicy),
		FailurePolicy:      ptr.To(admissionregistrationv1.Fail), AdmissionReviewVersions: []string{"v1"}, SideEffects: ptr.To(admissionregistrationv1.SideEffectClassNone),
	}
	// Trigger reinvocation so the tested mutation runs after the in-tree
	// plugins' final pass, which would otherwise repair missing protection.
	barrierURL := server.URL + "/barrier"
	barrierWebhook := mutationWebhook.DeepCopy()
	barrierWebhook.Name = "restore-mutation-barrier.integration.test"
	barrierWebhook.ClientConfig.URL = &barrierURL
	barrierWebhook.ReinvocationPolicy = ptr.To(admissionregistrationv1.NeverReinvocationPolicy)
	cfg, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(ctx, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: "restore-mutation.integration.test"},
		Webhooks:   []admissionregistrationv1.MutatingWebhook{mutationWebhook, *barrierWebhook},
	}, metav1.CreateOptions{})
	if err != nil {
		server.Close()
		t.Fatal(err)
	}
	closeWebhook := func() {
		require.NoError(t, client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(ctx, cfg.Name, metav1.DeleteOptions{}))
		server.Close()
	}
	err = wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		probe, err := client.CoreV1().Pods(namespace).Create(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "webhook-probe", Labels: map[string]string{"restore-mutation": "true", "mutation": "probe"}},
			Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "app", Image: "pause"}}},
		}, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
		return err == nil && probe.Annotations["webhook"] == "observed", err
	})
	if err != nil {
		closeWebhook()
		t.Fatalf("wait for mutating webhook: %v", err)
	}
	return closeWebhook
}

func TestRestoreStorageValidatesAfterMutation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, config, closeFn := startAPIServer(ctx, t, true, func(opts *options.ServerRunOptions) { opts.Authorization.Modes = []string{"RBAC"} })
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-storage-mutation", t)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	const username = "webhook-restorer"
	_, err := client.RbacV1().Roles(ns.Name).Create(ctx, &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: username}, Rules: []rbacv1.PolicyRule{
		{APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"create", "patch"}},
		{APIGroups: []string{"node.k8s.io"}, Resources: []string{"podcheckpoints"}, ResourceNames: []string{checkpoint.Name}, Verbs: []string{"restore"}},
	}}, metav1.CreateOptions{})
	require.NoError(t, err)
	_, err = client.RbacV1().RoleBindings(ns.Name).Create(ctx, &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: username}, Subjects: []rbacv1.Subject{{Kind: "User", APIGroup: rbacv1.GroupName, Name: username}}, RoleRef: rbacv1.RoleRef{Kind: "Role", APIGroup: rbacv1.GroupName, Name: username}}, metav1.CreateOptions{})
	require.NoError(t, err)
	for _, verb := range []string{"create", "patch"} {
		authutil.WaitForNamedAuthorizationUpdate(t, ctx, client.AuthorizationV1(), username, ns.Name, verb, "", schema.GroupResource{Resource: "pods"}, true)
	}
	authutil.WaitForNamedAuthorizationUpdate(t, ctx, client.AuthorizationV1(), username, ns.Name, "restore", checkpoint.Name, schema.GroupResource{Group: "node.k8s.io", Resource: "podcheckpoints"}, true)
	userConfig := rest.CopyConfig(config)
	userConfig.Impersonate = rest.ImpersonationConfig{UserName: username}
	userClient := clientset.NewForConfigOrDie(userConfig)
	closeWebhook := startRestoreMutationWebhook(t, ctx, client, ns.Name, map[string][]byte{
		"reference":  []byte(`[{"op":"replace","path":"/spec/restoreFrom/name","value":"unauthorized-checkpoint"}]`),
		"affinity":   []byte(`[{"op":"remove","path":"/spec/affinity"}]`),
		"protection": []byte(`[{"op":"remove","path":"/metadata/finalizers"}]`),
	})
	defer closeWebhook()
	for _, mutation := range []string{"unchanged", "reference", "affinity", "protection"} {
		for _, method := range []string{"post", "apply"} {
			t.Run(mutation+"/"+method, func(t *testing.T) {
				pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: mutation + "-" + method, Namespace: ns.Name, Labels: map[string]string{"restore-mutation": "true", "mutation": mutation}}, Spec: *checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()}
				pod.Spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name, UID: checkpoint.UID}
				if method == "post" {
					_, err = userClient.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{})
				} else {
					pod.APIVersion, pod.Kind = "v1", "Pod"
					data, marshalErr := json.Marshal(pod)
					require.NoError(t, marshalErr)
					_, err = userClient.CoreV1().Pods(ns.Name).Patch(ctx, pod.Name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "restore-mutation"})
				}
				switch mutation {
				case "unchanged":
					require.NoError(t, err)
				case "affinity":
					require.True(t, apierrors.IsInvalid(err), "final affinity must be validated: %v", err)
				default:
					require.True(t, apierrors.IsForbidden(err), "final reference and protection must be authorized: %v", err)
					if mutation == "reference" {
						require.ErrorContains(t, err, "not authorized")
						require.NotContains(t, err.Error(), "not found")
					}
				}
				stored, getErr := client.CoreV1().Pods(ns.Name).Get(ctx, pod.Name, metav1.GetOptions{})
				if mutation != "unchanged" {
					require.True(t, apierrors.IsNotFound(getErr), "mutated invalid restore must not persist: %v", getErr)
				} else {
					require.NoError(t, getErr)
					require.Equal(t, "true", stored.Annotations["restore-mutation-ready"])
					require.Contains(t, stored.Finalizers, nodeapi.PodRestoreProtectionFinalizer)
					expected := preparedRestorePod(checkpoint, pod.Name)
					require.Equal(t, expected.Spec.Affinity, stored.Spec.Affinity)
				}
			})
		}
	}
}

func TestRestoreNameOnlyApplyKeepsUID(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	ctx := ktesting.Init(t)
	client, _, closeFn := startAPIServer(ctx, t, true)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(client, "restore-apply-identity", t)
	checkpoint := createReadyRestoreCheckpoint(ctx, t, client, ns.Name, "checkpoint")
	spec := checkpoint.Status.CheckpointedPodTemplate.Spec.DeepCopy()
	spec.RestoreFrom = &v1.CheckpointReference{Name: checkpoint.Name}
	manifest := &v1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "restore", Namespace: ns.Name}, Spec: *spec}
	data, err := json.Marshal(manifest)
	require.NoError(t, err)
	for range 2 {
		stored, err := client.CoreV1().Pods(ns.Name).Patch(ctx, manifest.Name, types.ApplyPatchType, data, metav1.PatchOptions{FieldManager: "restore-apply", Force: new(true)})
		require.NoError(t, err, "reapplying the original name-only manifest must retain its injected checkpoint UID")
		require.Equal(t, checkpoint.UID, stored.Spec.RestoreFrom.UID)
	}
}
