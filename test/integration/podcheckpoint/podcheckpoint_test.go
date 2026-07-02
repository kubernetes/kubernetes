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
	"slices"
	"testing"
	"time"

	checkpointv1alpha1 "k8s.io/api/checkpoint/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/informers"
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
			"sourcePod": map[string]interface{}{
				"name": sourcePodName,
			},
		},
	}}
}

// TestPodCheckpointControllerReconciles verifies that the lifecycle controller
// removes the restore lock when no Pod is actively restoring from the checkpoint.
// Checkpoint execution and status updates belong to the kubelet.
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

	// Start the controller.
	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	controller := podcheckpointcontroller.NewController(dynamicClient, informerFactory.Core().V1().Pods())
	informerFactory.Start(tCtx.Done())
	go controller.Run(tCtx, 1)

	checkpoint := newPodCheckpoint(ns.Name, "cp-1", "source-pod")
	checkpoint.SetFinalizers([]string{podcheckpointcontroller.RestoreLockFinalizer})
	if _, err := dynamicClient.Resource(podCheckpointGVR).Namespace(ns.Name).Create(tCtx, checkpoint, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create PodCheckpoint: %v", err)
	}

	err = wait.PollUntilContextTimeout(tCtx, time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		obj, err := dynamicClient.Resource(podCheckpointGVR).Namespace(ns.Name).Get(ctx, "cp-1", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return !slices.Contains(obj.GetFinalizers(), podcheckpointcontroller.RestoreLockFinalizer), nil
	})
	if err != nil {
		t.Fatalf("controller did not remove the inactive restore lock: %v", err)
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
