/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"
	"time"

	checkpointv1alpha1 "k8s.io/api/checkpoint/v1alpha1"
	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	cliflag "k8s.io/component-base/cli/flag"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	podcheckpointcontroller "k8s.io/kubernetes/pkg/controller/podcheckpoint"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

var podCheckpointGVR = schema.GroupVersionResource{
	Group:    "checkpoint.k8s.io",
	Version:  "v1alpha1",
	Resource: "podcheckpoints",
}

// startAPIServer starts an in-process apiserver for the test. When serveCheckpointAPI
// is true it enables the alpha checkpoint.k8s.io/v1alpha1 group via --runtime-config;
// the PodLevelCheckpointRestore feature gate must be toggled separately by the caller
// (the gate also guards whether the group is served, see storage_checkpoint.go).
func startAPIServer(tCtx ktesting.TContext, t *testing.T, serveCheckpointAPI bool) (clientset.Interface, *restclient.Config, framework.TearDownFunc) {
	return framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable admission plugins that interfere with a minimal pod/node setup.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition", "Priority", "StorageObjectInUseProtection"}
			if serveCheckpointAPI {
				opts.APIEnablement.RuntimeConfig = cliflag.ConfigurationMap{
					checkpointv1alpha1.SchemeGroupVersion.String(): "true",
				}
			}
		},
	})
}

func newPodCheckpoint(namespace, name, sourcePodName string) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "checkpoint.k8s.io/v1alpha1",
		"kind":       "PodCheckpoint",
		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,
		},
		"spec": map[string]interface{}{
			"sourcePodName": sourcePodName,
		},
	}}
}

// TestPodCheckpointControllerReconciles verifies that, with the
// PodLevelCheckpointRestore feature gate enabled, the pod-snapshot-controller
// reconciles a PodCheckpoint: it records the source node, pins the source Pod's
// UID, captures the pod template, and sets the Ready condition. (No kubelet runs
// in integration, so the asynchronous trigger ultimately fails; the assertion
// only requires that the controller acted on the object.)
func TestPodCheckpointControllerReconciles(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)

	tCtx := ktesting.Init(t)
	clientSet, kubeConfig, closeFn := startAPIServer(tCtx, t, true)
	defer closeFn()

	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		t.Fatalf("failed to create dynamic client: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(clientSet, "podcheckpoint", t)

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	}
	if _, err := clientSet.CoreV1().Nodes().Create(tCtx, node, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create node: %v", err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "src-pod", Namespace: ns.Name},
		Spec: v1.PodSpec{
			NodeName:   node.Name,
			Containers: []v1.Container{{Name: "c", Image: "registry.example.com/app:v1"}},
		},
	}
	pod, err = clientSet.CoreV1().Pods(ns.Name).Create(tCtx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create pod: %v", err)
	}
	pod.Status.Phase = v1.PodRunning
	pod, err = clientSet.CoreV1().Pods(ns.Name).UpdateStatus(tCtx, pod, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("failed to set pod Running: %v", err)
	}

	// Start the controller.
	controller := podcheckpointcontroller.NewController(clientSet, dynamicClient)
	go controller.Run(tCtx, 1)

	if _, err := dynamicClient.Resource(podCheckpointGVR).Namespace(ns.Name).Create(tCtx, newPodCheckpoint(ns.Name, "cp-1", pod.Name), metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create PodCheckpoint: %v", err)
	}

	// The controller should reconcile: record node, pin the UID, capture the
	// template, and set a Ready condition.
	err = wait.PollUntilContextTimeout(tCtx, time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		obj, err := dynamicClient.Resource(podCheckpointGVR).Namespace(ns.Name).Get(ctx, "cp-1", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		var got checkpointv1alpha1.PodCheckpoint
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &got); err != nil {
			return false, err
		}
		if apimeta.FindStatusCondition(got.Status.Conditions, checkpointv1alpha1.PodCheckpointReady) == nil {
			return false, nil
		}
		if got.Status.NodeName != node.Name {
			return false, nil
		}
		if got.Status.CheckpointedPodTemplate == nil {
			return false, nil
		}
		if got.Status.SourcePodUID == nil || *got.Status.SourcePodUID != pod.UID {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("controller did not reconcile the PodCheckpoint with the feature gate enabled: %v", err)
	}
}

// TestPodCheckpointAPINotServedWhenDisabled verifies that with the
// PodLevelCheckpointRestore feature gate disabled the checkpoint.k8s.io API group
// is not served, so a PodCheckpoint cannot be created. This is the apiserver side
// of feature disablement; the controller itself is registered behind the gate via
// its ControllerDescriptor and is not started by kube-controller-manager when the
// gate is off.
func TestPodCheckpointAPINotServedWhenDisabled(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, false)

	tCtx := ktesting.Init(t)
	clientSet, kubeConfig, closeFn := startAPIServer(tCtx, t, false)
	defer closeFn()

	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		t.Fatalf("failed to create dynamic client: %v", err)
	}

	ns := framework.CreateNamespaceOrDie(clientSet, "podcheckpoint-disabled", t)

	if _, err := dynamicClient.Resource(podCheckpointGVR).Namespace(ns.Name).Create(tCtx, newPodCheckpoint(ns.Name, "cp-1", "src-pod"), metav1.CreateOptions{}); err == nil {
		t.Fatalf("expected PodCheckpoint create to fail when the feature gate is disabled (API not served), but it succeeded")
	}
}
